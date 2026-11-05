"""Trainable character-level Transformer for transliteration correction."""

import random

import torch
from torch import Tensor, nn

from src.detection.edit_distance import EditDistanceDetector

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"


class _CharacterTransformer(nn.Module):
    """Small encoder-decoder Transformer operating on character IDs."""

    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        model_dim: int = 32,
        max_length: int = 128,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, model_dim, padding_idx=pad_idx)
        self.position_embedding = nn.Embedding(max_length, model_dim)
        self.transformer = nn.Transformer(
            d_model=model_dim,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True,
        )
        self.output_projection = nn.Linear(model_dim, vocab_size)

    def _embed(self, tokens: Tensor) -> Tensor:
        length = tokens.size(1)
        if length > self.max_length:
            raise ValueError(f"sequence length exceeds maximum of {self.max_length}")
        positions = torch.arange(length, device=tokens.device).unsqueeze(0)
        return self.embedding(tokens) + self.position_embedding(positions)

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        target_length = target.size(1)
        causal_mask = torch.triu(
            torch.ones(
                (target_length, target_length),
                dtype=torch.bool,
                device=target.device,
            ),
            diagonal=1,
        )
        source_padding = source.eq(self.pad_idx)
        target_padding = target.eq(self.pad_idx)
        transformed = self.transformer(
            self._embed(source),
            self._embed(target),
            tgt_mask=causal_mask,
            src_key_padding_mask=source_padding,
            tgt_key_padding_mask=target_padding,
            memory_key_padding_mask=source_padding,
        )
        return self.output_projection(transformed)


class TransformerCorrector:
    """Train and run a character-level encoder-decoder correction Transformer."""

    def __init__(
        self,
        device: str = "cpu",
        lexicon: list[str] | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.lexicon = [word.lower() for word in lexicon or []]
        self.fallback_corrector = (
            EditDistanceDetector(self.lexicon) if self.lexicon else None
        )
        self.char2idx: dict[str, int] = {}
        self.idx2char: dict[int, str] = {}
        self.model: _CharacterTransformer | None = None
        self.last_trace: dict[str, str | bool | None] = {}
        self.fallback_count = 0

    def train_on_pairs(
        self,
        source_words: list[str],
        target_words: list[str],
        epochs: int = 100,
        learning_rate: float = 1e-3,
        batch_size: int = 16,
        seed: int = 42,
        verbose: bool = False,
    ) -> float:
        """Train on aligned noisy and clean words, returning the final mean loss."""
        if len(source_words) != len(target_words):
            raise ValueError("source and target word counts must match")
        if not source_words:
            raise ValueError("at least one training pair is required")
        if epochs <= 0 or batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        pairs = [
            (source.lower(), target.lower())
            for source, target in zip(source_words, target_words)
        ]
        vocabulary = [
            PAD_TOKEN,
            SOS_TOKEN,
            EOS_TOKEN,
            UNK_TOKEN,
            *sorted(set("".join(source + target for source, target in pairs))),
        ]
        self.char2idx = {token: index for index, token in enumerate(vocabulary)}
        self.idx2char = {index: token for token, index in self.char2idx.items()}
        pad_idx = self.char2idx[PAD_TOKEN]
        sos_idx = self.char2idx[SOS_TOKEN]
        eos_idx = self.char2idx[EOS_TOKEN]
        unk_idx = self.char2idx[UNK_TOKEN]

        torch.manual_seed(seed)
        self.model = _CharacterTransformer(
            vocab_size=len(vocabulary),
            pad_idx=pad_idx,
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        loss_function = nn.CrossEntropyLoss(ignore_index=pad_idx)
        random_generator = random.Random(seed)
        encoded_pairs = [
            (
                [self.char2idx.get(char, unk_idx) for char in source] + [eos_idx],
                [sos_idx]
                + [self.char2idx.get(char, unk_idx) for char in target]
                + [eos_idx],
            )
            for source, target in pairs
        ]

        self.model.train()
        final_loss = 0.0
        for epoch in range(epochs):
            random_generator.shuffle(encoded_pairs)
            epoch_loss = 0.0
            batch_count = 0
            for start in range(0, len(encoded_pairs), batch_size):
                batch = encoded_pairs[start : start + batch_size]
                source_batch = self._pad_batch(
                    [source for source, _ in batch],
                    pad_idx,
                )
                target_batch = self._pad_batch(
                    [target for _, target in batch],
                    pad_idx,
                )
                decoder_input = target_batch[:, :-1]
                expected = target_batch[:, 1:]

                optimizer.zero_grad()
                logits = self.model(source_batch, decoder_input)
                loss = loss_function(
                    logits.reshape(-1, logits.size(-1)),
                    expected.reshape(-1),
                )
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += float(loss.item())
                batch_count += 1

            final_loss = epoch_loss / max(1, batch_count)
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} - loss: {final_loss:.4f}")

        self.model.eval()
        return final_loss

    def _pad_batch(self, sequences: list[list[int]], pad_idx: int) -> Tensor:
        max_length = max(len(sequence) for sequence in sequences)
        batch = torch.full(
            (len(sequences), max_length),
            fill_value=pad_idx,
            dtype=torch.long,
            device=self.device,
        )
        for row, sequence in enumerate(sequences):
            batch[row, : len(sequence)] = torch.tensor(
                sequence,
                dtype=torch.long,
                device=self.device,
            )
        return batch

    def _predict_word(self, word: str) -> str:
        if self.model is None:
            raise RuntimeError("Transformer has not been trained")
        source_ids = [
            self.char2idx.get(char.lower(), self.char2idx[UNK_TOKEN])
            for char in word
        ]
        source_ids.append(self.char2idx[EOS_TOKEN])
        source = torch.tensor(
            [source_ids],
            dtype=torch.long,
            device=self.device,
        )
        decoder_ids = [self.char2idx[SOS_TOKEN]]
        output_chars: list[str] = []
        eos_idx = self.char2idx[EOS_TOKEN]

        with torch.no_grad():
            for _ in range(min(max(2 * len(word) + 4, 12), 64)):
                decoder_input = torch.tensor(
                    [decoder_ids],
                    dtype=torch.long,
                    device=self.device,
                )
                logits = self.model(source, decoder_input)
                next_idx = int(logits[0, -1].argmax().item())
                if next_idx == eos_idx:
                    break
                decoder_ids.append(next_idx)
                token = self.idx2char.get(next_idx, UNK_TOKEN)
                if token not in {PAD_TOKEN, SOS_TOKEN, UNK_TOKEN}:
                    output_chars.append(token)

        return "".join(output_chars)

    def correct_sentence(self, sentence: str) -> str:
        if self.model is None:
            raise RuntimeError("Transformer has not been trained")

        raw_words: list[str] = []
        output_words: list[str] = []
        used_fallback = False
        for word in sentence.split():
            candidate = self._predict_word(word)
            raw_words.append(candidate)
            if self.fallback_corrector is not None and candidate not in self.lexicon:
                corrected = self.fallback_corrector.process_sentence(word)[0]
                output_words.append(corrected)
                used_fallback = True
            else:
                output_words.append(candidate)

        result = " ".join(output_words)
        self.fallback_count += int(used_fallback)
        self.last_trace = {
            "input": sentence,
            "model_output": " ".join(raw_words),
            "decision": (
                "used lexicon fallback for out-of-lexicon word(s)"
                if used_fallback
                else "accepted trained Transformer output"
            ),
            "output": result,
        }
        return result

    def correct(self, text: str) -> str:
        return self.correct_sentence(text)
