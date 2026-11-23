"""Phase 3: context-aware correction with a pretrained multilingual transformer.

For every Indic word the ranker collects a short list of candidate spellings from
Phase 1 (edit distance + phonetic G2P matching) and Phase 2 (the GRU), then asks
XLM-RoBERTa how well each candidate fits the *rest of the sentence*. The score of
a candidate is the masked-language-model pseudo-log-likelihood of its sub-word
pieces in context (left-to-right within the word), combined with how close the
candidate is to what the user typed:

    score(c) = logP_xlmr(c | context) / pieces(c)**KAPPA
               - BETA * noise_cost(typed word, c)

Candidates are always lexicon words: Phase 1's matches (within two edits, or a
phonetic match) plus the GRU's answer when it is a lexicon word that is not too far
from what was typed. The transformer chooses between plausible fixes; it cannot
invent words.
"""

import re
from typing import Any

import torch

from src.detection import levenshtein as editdistance
from src.detection.edit_distance import EditDistanceDetector, gru_candidate_ok
from src.detection.noise_distance import noise_cost

MODEL_NAME = "xlm-roberta-base"
MAX_CANDIDATES = 4
MAX_DISTANCE = 2

# Weights of the scoring formula above (tuned on the "tune" split of data/eval_sentences.csv).
KAPPA = 1.0
BETA = 10.0  # one unit of typing-slip cost is worth 10 log-probability points: the transformer breaks near-ties

_EDGES = re.compile(r"^(\W*)(.*?)(\W*)$", re.UNICODE)


class ContextRanker:
    """Choose between candidate corrections using XLM-RoBERTa and sentence context."""

    def __init__(
        self,
        lexicon: list[str],
        variants: dict[str, str] | None = None,
        model_name: str = MODEL_NAME,
        device: str = "cpu",
        kappa: float = KAPPA,
        beta: float = BETA,
    ) -> None:
        self.lexicon = [word.lower() for word in lexicon]
        self._lexicon_set = set(self.lexicon)
        self.detector = EditDistanceDetector(self.lexicon, variants)
        self.model_name = model_name
        self.device = torch.device(device)
        self.kappa, self.beta = kappa, beta
        self.tokenizer: Any = None
        self.model: Any = None
        self.load_error: str | None = None
        self._cache: dict[Any, dict[str, tuple[float, int]]] = {}
        self.last_trace: dict[str, Any] = {}

    @property
    def available(self) -> bool:
        return self.model is not None

    def load(self) -> bool:
        """Load XLM-RoBERTa (downloaded once, then cached). Returns False on failure."""
        if self.available:
            return True
        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_name).to(self.device).eval()
            self.load_error = None
        except Exception as error:  # offline, missing package, corrupt cache...
            self.tokenizer = self.model = None
            self.load_error = f"{type(error).__name__}: {error}"
        return self.available

    # ------------------------------------------------------------------ scoring

    def _piece_logprobs(self, words: list[str], index: int, candidates: list[str]) -> dict[str, tuple[float, int]]:
        """Cached :meth:`_compute_piece_logprobs` (the same question is often asked twice)."""
        key = (tuple(words), index, tuple(candidates))
        if key not in self._cache:
            if len(self._cache) > 20_000:
                self._cache.clear()
            self._cache[key] = self._compute_piece_logprobs(words, index, candidates)
        return self._cache[key]

    def _compute_piece_logprobs(
        self, words: list[str], index: int, candidates: list[str]
    ) -> dict[str, tuple[float, int]]:
        """Return ``{candidate: (sum of log-probabilities, number of pieces)}`` in context."""
        sequences: list[tuple[list[int], int, int, str]] = []  # (ids, position, target id, candidate)
        pieces_per_candidate: dict[str, int] = {}
        prefix = " ".join(words[:index])
        start = len(prefix) + (1 if prefix else 0)

        for candidate in candidates:
            text = " ".join([*words[:index], candidate, *words[index + 1 :]])
            encoded = self.tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=128)
            ids = encoded["input_ids"]
            end = start + len(candidate)
            span = [
                position
                for position, (left, right) in enumerate(encoded["offset_mapping"])
                if right > left and left < end and right > start
            ]
            if not span:
                pieces_per_candidate[candidate] = 0
                continue
            pieces_per_candidate[candidate] = len(span)
            for step, position in enumerate(span):
                masked = list(ids)
                for later in span[step:]:  # hide this piece and the rest of the word
                    masked[later] = self.tokenizer.mask_token_id
                sequences.append((masked, position, ids[position], candidate))

        totals = {candidate: 0.0 for candidate in candidates}
        if sequences:
            width = max(len(item[0]) for item in sequences)
            pad = self.tokenizer.pad_token_id
            input_ids = torch.full((len(sequences), width), pad, dtype=torch.long)
            attention = torch.zeros((len(sequences), width), dtype=torch.long)
            for row, (ids, _, _, _) in enumerate(sequences):
                input_ids[row, : len(ids)] = torch.tensor(ids)
                attention[row, : len(ids)] = 1
            with torch.no_grad():
                logits = self.model(
                    input_ids=input_ids.to(self.device), attention_mask=attention.to(self.device)
                ).logits
            for row, (_, position, target, candidate) in enumerate(sequences):
                totals[candidate] += float(torch.log_softmax(logits[row, position], dim=-1)[target])
        return {candidate: (totals[candidate], pieces_per_candidate[candidate]) for candidate in candidates}

    def _score(self, typed: str, candidate: str, logprob: float, pieces: int) -> float:
        context_score = logprob / (max(pieces, 1) ** self.kappa) if self.available else 0.0
        return context_score - self.beta * noise_cost(typed, candidate)

    # ------------------------------------------------------------------ public API

    def candidates_for(self, typed: str, extra: list[str] | None = None) -> list[str]:
        """Lexicon words that could be what ``typed`` was meant to be (best first).

        Phase 1 suggestions come first; GRU suggestions are added when they are lexicon
        words within :func:`gru_distance_limit` of the typed word.
        """
        typed_lower = typed.lower()
        suggestions = list(self.detector.get_candidates(typed_lower, MAX_DISTANCE, MAX_CANDIDATES))
        for candidate in extra or []:
            candidate = candidate.lower()
            if candidate not in suggestions and candidate in self._lexicon_set and gru_candidate_ok(typed_lower, candidate):
                suggestions.append(candidate)
        return suggestions

    def correct_tokens(
        self,
        tokens: list[str],
        correctable: list[bool] | None = None,
        extra: dict[int, list[str]] | None = None,
    ) -> list[str]:
        """Correct ``tokens``; only positions flagged in ``correctable`` may change."""
        correctable = correctable or [True] * len(tokens)
        extra = extra or {}
        parts = [_EDGES.match(token).groups() for token in tokens]  # (prefix, core, suffix)
        cores = [core for _, core, _ in parts]

        candidate_lists: dict[int, list[str]] = {}
        for index, core in enumerate(cores):
            if not correctable[index] or not core or self.detector.is_known(core):
                continue
            options = self.candidates_for(core, extra.get(index))
            if options:
                candidate_lists[index] = options

        # Context for scoring: each word is first set to its best cheap guess
        # (the GRU's answer if it gave one, else the closest candidate).
        context = list(cores)
        for index, options in candidate_lists.items():
            context[index] = next((c for c in extra.get(index, []) if c.lower() in options), options[0])

        chosen = list(cores)
        rankings: list[dict[str, Any]] = []
        for index, options in candidate_lists.items():
            if len(options) == 1:  # nothing to choose between
                chosen[index] = options[0]
                context[index] = options[0]
                continue
            if self.available:
                stats = self._piece_logprobs(context, index, options)
            else:
                stats = {option: (0.0, 1) for option in options}
            scored = sorted(
                ((self._score(cores[index], option, *stats[option]), option) for option in options),
                reverse=True,
            )
            best = scored[0][1]
            chosen[index] = best
            rankings.append(
                {
                    "position": index,
                    "token": cores[index],
                    "chosen": best,
                    "candidates": [
                        {"word": option, "score": round(score, 3), "pieces": stats[option][1]}
                        for score, option in scored
                    ],
                }
            )
            context[index] = chosen[index]

        result = []
        for (prefix, core, suffix), final in zip(parts, chosen):
            if final != core and core[:1].isupper():
                final = final[:1].upper() + final[1:]
            result.append(f"{prefix}{final}{suffix}")

        changed = sum(1 for old, new in zip(cores, chosen) if old != new)
        output = " ".join(result)
        self.last_trace = {
            "input": " ".join(tokens),
            "model_output": output,
            "decision": (
                (
                    f"XLM-RoBERTa chose between candidates for {len(rankings)} ambiguous word(s); "
                    f"{changed} word(s) changed"
                    if rankings
                    else "No ambiguous words, so XLM-RoBERTa had nothing to decide"
                )
                if self.available
                else "XLM-RoBERTa unavailable; used edit-distance, phonetic and GRU suggestions only"
            ),
            "output": output,
            "rankings": rankings,
            "model": self.model_name if self.available else None,
        }
        return result

    def correct_sentence(self, sentence: str, **kwargs: Any) -> str:
        """Correct a whitespace-separated sentence (see :meth:`correct_tokens`)."""
        return " ".join(self.correct_tokens(sentence.split(), **kwargs))
