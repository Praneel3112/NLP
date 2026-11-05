"""Character-level GRU sequence-to-sequence transliteration correction."""

import random

import torch
from torch import Tensor, nn

from src.detection.edit_distance import EditDistanceDetector


class EncoderRNN(nn.Module):
    """Embed source characters and encode them with a three-layer GRU."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        pad_idx: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=3,
            dropout=dropout,
        )

    def forward(self, src: Tensor) -> tuple[Tensor, Tensor]:
        """Encode ``src`` shaped ``[source_length, batch_size]``."""
        embedded = self.embedding(src)
        return self.gru(embedded)


class Attention(nn.Module):
    """Bahdanau additive or Luong dot-product attention."""

    def __init__(self, hidden_dim: int, attention_type: str = "bahdanau") -> None:
        super().__init__()
        if attention_type not in {"bahdanau", "luong"}:
            raise ValueError("attention_type must be 'bahdanau' or 'luong'")
        self.attention_type = attention_type
        if attention_type == "bahdanau":
            self.encoder_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.decoder_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.energy = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: Tensor,
        encoder_outputs: Tensor,
        src_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return context ``[batch, hidden]`` and weights ``[batch, source]``."""
        encoder_batch_first = encoder_outputs.transpose(0, 1)
        if self.attention_type == "bahdanau":
            scores = self.energy(
                torch.tanh(
                    self.encoder_projection(encoder_batch_first)
                    + self.decoder_projection(decoder_hidden).unsqueeze(1)
                )
            ).squeeze(-1)
        else:
            scores = torch.bmm(
                encoder_batch_first,
                decoder_hidden.unsqueeze(2),
            ).squeeze(2)

        if src_mask is not None:
            source_first_shape = (
                encoder_outputs.size(0),
                encoder_outputs.size(1),
            )
            batch_first_shape = (
                encoder_outputs.size(1),
                encoder_outputs.size(0),
            )
            if (
                src_mask.shape == source_first_shape
                and src_mask.shape != batch_first_shape
            ):
                src_mask = src_mask.transpose(0, 1)
            if src_mask.shape != scores.shape:
                raise ValueError("src_mask must have shape [source, batch] or [batch, source]")
            scores = scores.masked_fill(~src_mask.bool(), torch.finfo(scores.dtype).min)

        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_batch_first).squeeze(1)
        return context, weights


class DecoderRNN(nn.Module):
    """Decode target characters while attending to source encoder states."""

    def __init__(
        self,
        output_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        pad_idx: int = 0,
        attention_type: str = "bahdanau",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, embedding_dim, padding_idx=pad_idx)
        self.attention = Attention(hidden_dim, attention_type)
        self.gru = nn.GRU(
            embedding_dim + hidden_dim,
            hidden_dim,
            num_layers=3,
            dropout=dropout,
        )
        self.output_projection = nn.Linear(
            embedding_dim + hidden_dim * 2,
            output_dim,
        )

    def forward(
        self,
        input_token: Tensor,
        hidden: Tensor,
        encoder_outputs: Tensor,
        src_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Decode one step from ``input_token`` shaped ``[batch_size]``."""
        if input_token.ndim == 2 and input_token.size(0) == 1:
            input_token = input_token.squeeze(0)
        embedded = self.embedding(input_token)
        context, attention_weights = self.attention(
            hidden[-1],
            encoder_outputs,
            src_mask,
        )
        gru_input = torch.cat((embedded, context), dim=1).unsqueeze(0)
        output, hidden = self.gru(gru_input, hidden)
        output = output.squeeze(0)
        logits = self.output_projection(
            torch.cat((output, context, embedded), dim=1)
        )
        return logits, hidden, attention_weights


class Seq2Seq(nn.Module):
    """Combine the character encoder and attention-based decoder."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        pad_idx: int = 0,
        sos_idx: int = 1,
        eos_idx: int = 2,
        attention_type: str = "bahdanau",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = EncoderRNN(
            input_dim,
            embedding_dim,
            hidden_dim,
            pad_idx,
            dropout,
        )
        self.decoder = DecoderRNN(
            output_dim,
            embedding_dim,
            hidden_dim,
            pad_idx,
            attention_type,
            dropout,
        )
        self.output_dim = output_dim
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def forward(
        self,
        src: Tensor,
        trg: Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> Tensor:
        """Return target logits shaped ``[target_length, batch, vocab_size]``.

        ``src`` and ``trg`` use sequence-first shapes. The first target row is
        the start-of-sequence token and is used to begin decoding.
        """
        if not 0.0 <= teacher_forcing_ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio must be between 0 and 1")
        if src.ndim != 2 or trg.ndim != 2:
            raise ValueError("src and trg must have shape [sequence, batch]")
        if src.size(0) == 0 or trg.size(0) == 0:
            raise ValueError("src and trg must include at least one sequence token")
        if src.size(1) != trg.size(1):
            raise ValueError("src and trg must have the same batch size")

        encoder_outputs, hidden = self.encoder(src)
        src_mask = src.ne(self.pad_idx)
        batch_size = src.size(1)
        outputs: list[Tensor] = []
        input_token = trg[0]

        for target_index in range(1, trg.size(0)):
            logits, hidden, _ = self.decoder(
                input_token,
                hidden,
                encoder_outputs,
                src_mask,
            )
            outputs.append(logits)
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            input_token = (
                trg[target_index]
                if use_teacher_forcing
                else logits.argmax(dim=1)
            )

        if outputs:
            return torch.stack(outputs, dim=0)
        return src.new_empty((0, batch_size, self.output_dim), dtype=torch.float)


class Seq2SeqRNN(Seq2Seq):
    """Trainable character-level seq2seq transliteration model."""

    def __init__(
        self,
        encoder: nn.Module | None = None,
        decoder: nn.Module | None = None,
        device: str = "cpu",
        input_dim: int = 50,
        output_dim: int = 50,
        embedding_dim: int = 16,
        hidden_dim: int = 32,
        lexicon: list[str] | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.device = torch.device(device)
        self.encoder = encoder
        self.decoder = decoder
        self.fallback_corrector = EditDistanceDetector(lexicon) if lexicon else None
        self.char2idx: dict[str, int] | None = None
        self.idx2char: dict[int, str] | None = None
        self.sos_idx = 1
        self.eos_idx = 2
        self.pad_idx = 0
        self.input_dim = input_dim
        self.output_dim = output_dim

        if encoder is not None and decoder is not None:
            Seq2Seq.__init__(
                self,
                input_dim=input_dim,
                output_dim=output_dim,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                pad_idx=self.pad_idx,
                sos_idx=self.sos_idx,
                eos_idx=self.eos_idx,
            )
            self.encoder = encoder
            self.decoder = decoder
            self.to(self.device)
            return

        if encoder is None and decoder is None:
            vocab = self._build_vocab_from_lexicon(lexicon)
            if not vocab:
                self.encoder = None
                self.decoder = None
                return
            self.char2idx, self.idx2char = self._make_char_maps(vocab)
            self.input_dim = len(self.char2idx)
            self.output_dim = len(self.char2idx)
            self.sos_idx = self.char2idx["<sos>"]
            self.eos_idx = self.char2idx["<eos>"]
            self.pad_idx = self.char2idx["<pad>"]
            Seq2Seq.__init__(
                self,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                embedding_dim=max(8, min(32, self.input_dim // 2)),
                hidden_dim=max(16, min(64, self.input_dim // 2)),
                pad_idx=self.pad_idx,
                sos_idx=self.sos_idx,
                eos_idx=self.eos_idx,
            )
            self.to(self.device)
            return

        raise ValueError("encoder and decoder must either both be provided or both be None")

    @staticmethod
    def _build_vocab_from_lexicon(lexicon: list[str] | None) -> list[str]:
        base_chars: set[str] = set()
        if lexicon:
            for word in lexicon:
                base_chars.update(word.lower())
        vocab = sorted(base_chars)
        if not vocab:
            return []
        tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]
        return tokens + [char for char in vocab if char not in tokens]

    @staticmethod
    def _make_char_maps(vocab: list[str]) -> tuple[dict[str, int], dict[int, str]]:
        char2idx = {token: idx for idx, token in enumerate(vocab)}
        idx2char = {idx: token for token, idx in char2idx.items()}
        return char2idx, idx2char

    def _reset_model_for_vocab(self, vocab: list[str]) -> None:
        self.char2idx, self.idx2char = self._make_char_maps(vocab)
        self.input_dim = len(self.char2idx)
        self.output_dim = len(self.char2idx)
        self.pad_idx = self.char2idx["<pad>"]
        self.sos_idx = self.char2idx["<sos>"]
        self.eos_idx = self.char2idx["<eos>"]
        Seq2Seq.__init__(
            self,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            embedding_dim=max(8, min(32, self.input_dim // 2)),
            hidden_dim=max(16, min(64, self.input_dim // 2)),
            pad_idx=self.pad_idx,
            sos_idx=self.sos_idx,
            eos_idx=self.eos_idx,
        )
        self.to(self.device)

    def train_on_pairs(
        self,
        source_sentences: list[str],
        target_sentences: list[str],
        epochs: int = 60,
        learning_rate: float = 1e-3,
        teacher_forcing_ratio: float = 0.5,
        verbose: bool = False,
    ) -> float:
        """Train a character-level seq2seq model on noisy-to-clean sentence pairs."""
        if len(source_sentences) != len(target_sentences):
            raise ValueError("source and target sentence counts must match")
        if not source_sentences:
            raise ValueError("at least one training pair is required")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= teacher_forcing_ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio must be between 0 and 1")

        training_pairs: list[tuple[str, str]] = []
        for source, target in zip(source_sentences, target_sentences):
            source_words = source.lower().split()
            target_words = target.lower().split()
            if len(source_words) != len(target_words):
                raise ValueError(
                    "each source and target pair must contain the same number of words"
                )
            training_pairs.extend(zip(source_words, target_words))

        if not training_pairs:
            raise ValueError("training pairs must contain at least one word")

        vocab_chars = [
            "<pad>",
            "<sos>",
            "<eos>",
            "<unk>",
            *sorted(
                set(
                    "".join(
                        source + target for source, target in training_pairs
                    )
                )
            ),
        ]
        expected_char2idx, _ = self._make_char_maps(vocab_chars)
        if (
            self.encoder is None
            or self.decoder is None
            or self.char2idx != expected_char2idx
        ):
            self._reset_model_for_vocab(vocab_chars)

        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

        last_loss = 0.0
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            shuffled_pairs = training_pairs.copy()
            random.shuffle(shuffled_pairs)
            for src_text, trg_text in shuffled_pairs:
                src_tensor = self._encode_sequence(src_text, self.char2idx, self.pad_idx)
                trg_tensor = self._encode_sequence(trg_text, self.char2idx, self.pad_idx)
                src = torch.tensor(src_tensor, dtype=torch.long, device=self.device).unsqueeze(1)
                trg = torch.tensor(trg_tensor, dtype=torch.long, device=self.device).unsqueeze(1)

                optimizer.zero_grad()
                logits = self(src, trg, teacher_forcing_ratio=teacher_forcing_ratio)
                if logits.size(0) == 0:
                    continue
                target = trg[1:].contiguous().view(-1)
                logits = logits.contiguous().view(-1, logits.size(-1))
                loss = criterion(logits, target)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += float(loss.item())

            last_loss = epoch_loss / len(training_pairs)
            if verbose:
                print(f"Epoch {epoch}/{epochs} - loss: {last_loss:.4f}")

        return last_loss

    @staticmethod
    def _encode_sequence(
        text: str,
        char2idx: dict[str, int] | None,
        pad_idx: int,
    ) -> list[int]:
        if char2idx is None:
            return []
        chars = [ch.lower() for ch in text if ch.isalnum()]
        if not chars:
            return [char2idx["<sos>"], char2idx["<eos>"]]
        encoded = [char2idx.get(ch, char2idx["<unk>"]) for ch in chars]
        return [char2idx["<sos>"]] + encoded + [char2idx["<eos>"]]

    def predict_word(
        self,
        word: str,
        char2idx: dict[str, int] | None = None,
        idx2char: dict[int, str] | None = None,
    ) -> str:
        """Correct a noisy token using the trained model or lexicon fallback."""
        if not word:
            return ""

        if self.encoder is None or self.decoder is None:
            if self.fallback_corrector is None:
                raise RuntimeError(
                    "RNN is not initialized; provide a trained model or lexicon"
                )
            return self.fallback_corrector.process_sentence(word)[0]

        if char2idx is None:
            char2idx = self.char2idx
        if idx2char is None:
            idx2char = self.idx2char
        if char2idx is None or idx2char is None:
            raise ValueError("char2idx and idx2char are required for RNN inference")

        return inference_correct(self, word, char2idx, idx2char)

    def correct_sentence(
        self,
        sentence: str,
        char2idx: dict[str, int] | None = None,
        idx2char: dict[int, str] | None = None,
    ) -> str:
        """Correct a sentence with the trained model or configured lexicon fallback."""
        if self.encoder is None or self.decoder is None:
            if self.fallback_corrector is None:
                raise RuntimeError(
                    "RNN is not initialized; provide a trained model or lexicon"
                )
            return self.fallback_corrector.process_sentence(sentence)[0]

        if char2idx is None:
            char2idx = self.char2idx
        if idx2char is None:
            idx2char = self.idx2char
        if char2idx is None or idx2char is None:
            raise ValueError("char2idx and idx2char are required for RNN inference")

        return " ".join(
            self.predict_word(token, char2idx, idx2char)
            for token in sentence.split()
        )


def inference_correct(
    model: Seq2Seq,
    word: str,
    char2idx: dict[str, int],
    idx2char: dict[int, str],
) -> str:
    """Greedily correct a word using a trained character-level seq2seq model."""
    if not word:
        return ""

    normalized_char2idx = {
        str(token).casefold(): int(index) for token, index in char2idx.items()
    }
    normalized_idx2char = {int(index): token for index, token in idx2char.items()}

    def find_index(names: tuple[str, ...], default: int | None = None) -> int | None:
        for name in names:
            if name in normalized_char2idx:
                return normalized_char2idx[name]
        return default

    sos_idx = find_index(("<sos>", "<bos>", "<start>", "sos", "bos"), getattr(model, "sos_idx", None))
    eos_idx = find_index(("<eos>", "<end>", "eos"), getattr(model, "eos_idx", None))
    pad_idx = find_index(("<pad>", "pad"), getattr(model, "pad_idx", 0))
    unk_idx = find_index(("<unk>", "unk"))
    if sos_idx is None or eos_idx is None:
        raise ValueError("model or vocabulary must define SOS and EOS indices")

    source_indices: list[int] = [sos_idx]
    for char in word.lower():
        if char in char2idx:
            source_indices.append(int(char2idx[char]))
        elif unk_idx is not None:
            source_indices.append(unk_idx)
        else:
            raise ValueError(f"character {char!r} is not present in char2idx")
    source_indices.append(eos_idx)

    device = next(model.parameters()).device
    src = torch.tensor(source_indices, dtype=torch.long, device=device).unsqueeze(1)
    src_mask = src.ne(pad_idx)
    max_length = max(2 * len(word) + 10, 20)
    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            encoder_outputs, hidden = model.encoder(src)
            input_token = torch.tensor([sos_idx], dtype=torch.long, device=device)
            output_chars: list[str] = []
            for _ in range(max_length):
                logits, hidden, _ = model.decoder(
                    input_token,
                    hidden,
                    encoder_outputs,
                    src_mask,
                )
                next_index = logits.argmax(dim=1)
                output_index = int(next_index.item())
                if output_index == eos_idx:
                    break
                if output_index in {sos_idx, pad_idx, unk_idx}:
                    input_token = next_index
                    continue
                if output_index in normalized_idx2char:
                    output_chars.append(str(normalized_idx2char[output_index]))
                input_token = next_index
    finally:
        model.train(was_training)

    return "".join(output_chars)