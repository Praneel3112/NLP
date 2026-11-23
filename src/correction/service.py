"""Train and serve the project's sample transliteration correction models."""

import re
import threading
import time
import traceback
from typing import Any

from src.detection import levenshtein as editdistance
import torch

from src.correction.context_ranker import ContextRanker
from src.correction.model_store import load_or_train
from src.correction.pipeline import LEXICON, VARIANTS
from src.correction.rnn_baseline import Seq2SeqRNN
from src.detection.edit_distance import HIGH_CONFIDENCE_COST, EditDistanceDetector, gru_candidate_ok
from src.detection.noise_distance import noise_cost
from src.emotion.detector import EmotionDetector
from src.lang_id.detector import LanguageIdentifier
from src.preprocessing.slang_normalizer import SlangNormalizer

TRAINING_EPOCHS = 25  # GRU training epochs (see model_store.py)
_EDGE_PUNCTUATION = re.compile(r"^\W+|\W+$")
_TOKEN_EDGES = re.compile(r"^(\W*)(.*?)(\W*)$", re.UNICODE)

# A GRU answer is only used when it is a lexicon word that typing slips could explain. (The GRU's
# own confidence is shown but not used as a gate: measured on held-out data it is just as high
# for its mistakes and for unseen words as for its correct answers.)
GRU_MAX_NOISE_COST = 2.5

# Internet and chat words that the NLTK English word list lacks.
EXTRA_ENGLISH = frozenset(
    "bro bruh dude guys wifi whatsapp insta instagram youtube google facebook selfie email internet "
    "online offline app link password chat group status reel reels meme memes ok okay hmm lol omg btw".split()
)
_LONG_REPEAT = re.compile(r"(.)\1{2,}")


class CorrectionService:
    """Manage one training run and serialize inference through its shared models."""

    def __init__(self) -> None:
        self._state_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._state: dict[str, Any] = {
            "status": "loading",
            "stage": "Starting model training",
            "error": None,
            "training_examples": 0,
            "phase2_loss": None,
            "phase3_model": None,
            "emotion_accuracy": None,
            "emotion_macro_f1": None,
            "lexicon_size": len(LEXICON),
            "models_loaded_from_disk": None,
        }
        self._edit_detector = EditDistanceDetector(list(LEXICON), VARIANTS)
        self._lexicon = {word.casefold() for word in LEXICON}
        self._language_id = LanguageIdentifier()
        self._slang = SlangNormalizer()
        self._emotion = EmotionDetector()
        self._emotion_ready = False
        self._rnn: Seq2SeqRNN | None = None
        self._ranker = ContextRanker(list(LEXICON), VARIANTS)

    def warm_up(self) -> None:
        """Train both models once, reporting progress and any startup failure."""
        try:
            torch.set_num_threads(1)
            self._update_state(stage="Training emotion classifier (TF-IDF + Logistic Regression)")
            emotion_metrics = self._emotion.train()
            self._emotion_ready = True
            self._update_state(
                emotion_accuracy=emotion_metrics["comparison"]["hybrid"]["accuracy"],
                emotion_macro_f1=emotion_metrics["comparison"]["hybrid"]["macro_f1"],
            )

            self.load_correction_models()
            self._update_state(status="ready", stage="All models are ready")
        except Exception as error:
            traceback.print_exc()
            self._update_state(
                status="error",
                stage="Model training failed",
                error=str(error),
            )

    def load_correction_models(self) -> None:
        """Load (or train once) the Phase 2 GRU and load the Phase 3 XLM-RoBERTa ranker."""
        self._update_state(stage="Loading saved Phase 2 GRU")
        models = load_or_train(TRAINING_EPOCHS, on_stage=lambda stage: self._update_state(stage=stage))
        self._rnn = models["gru"]
        self._update_state(
            training_examples=models["training_examples"],
            lexicon_size=len(LEXICON),
            models_loaded_from_disk=models["loaded_from_disk"],
            phase2_loss=round(models["gru_loss"], 4),
        )

        self._update_state(stage="Loading XLM-RoBERTa for Phase 3 (first run downloads about 1 GB)")
        loaded = self._ranker.load()
        if not loaded:
            print(f"XLM-RoBERTa could not be loaded: {self._ranker.load_error}", flush=True)
        self._update_state(phase3_model=self._ranker.model_name if loaded else "unavailable")

    def run_phases(self, text: str) -> dict[str, Any]:
        """Run only the correction phases (language filter + Phases 1-3) on ``text``."""
        with self._inference_lock:
            return self._correct_code_mixed(text)

    def get_status(self) -> dict[str, Any]:
        with self._state_lock:
            return dict(self._state)

    def get_emotion_metrics(self) -> dict[str, Any] | None:
        return self._emotion.metrics if self._emotion_ready else None

    def correct(self, text: str) -> dict[str, Any]:
        """Run Phase 0 (slang), emotion detection, and the three correction phases."""
        if not text.strip():
            raise ValueError("Text must contain at least one non-whitespace character.")
        if any(len(word) > 120 for word in text.split()):
            raise ValueError("Each word must be 120 characters or fewer.")

        with self._inference_lock:
            if self._rnn is None:
                status = self.get_status()
                if status["status"] == "error":
                    raise RuntimeError(status["error"] or "Model training failed.")
                raise RuntimeError("The correction models are still training.")

            phase0 = self._slang.normalize(text)
            emotion = self._emotion.detect(text) if self._emotion_ready else None
            normalized = phase0["normalized"]
            if normalized.strip():
                result = self._correct_code_mixed(normalized)
            else:
                # The input was only emoji: there are no words to spell-correct.
                result = {
                    "phase1": "",
                    "phase1_guided": "",
                    "phase1_steps": [],
                    "phase2": "",
                    "phase2_steps": [],
                    "phase3": "",
                    "phase3_trace": {
                        "model_output": None,
                        "decision": "No words to correct; the input contained only emoji.",
                        "output": "",
                    },
                }
            return {"input": text, "phase0": phase0, "emotion": emotion, **result}

    def is_english_token(self, token: str) -> bool:
        """True for English words, which must not be forced onto the Indic lexicon.

        Lexicon words always count as Indic. Letter stretching is ignored, so
        "doingggg" is checked as "doing".
        """
        core = _EDGE_PUNCTUATION.sub("", token).casefold()
        if not core or core in self._lexicon or core in VARIANTS or not core.isalpha():
            return False
        variants = {core, _LONG_REPEAT.sub(r"\1\1", core), _LONG_REPEAT.sub(r"\1", core)}
        if variants & self._lexicon:
            return False
        if variants & EXTRA_ENGLISH:
            return True
        return any(self._language_id.is_english_word(variant) for variant in variants)

    def _correct_code_mixed(self, text: str) -> dict[str, Any]:
        """Send only Indic words through Phases 1-3; English words pass through unchanged."""
        tokens = text.split()
        english = [self.is_english_token(token) for token in tokens]
        languages = [
            {"token": token, "label": "ENGLISH" if is_en else "ROMANIZED_INDIC"}
            for token, is_en in zip(tokens, english)
        ]
        if not any(english):
            # Pure Romanized Indic: exactly the CIE3 behaviour.
            return {**self._correct_phases(text), "languages": languages}

        indic_text = " ".join(token for token, is_en in zip(tokens, english) if not is_en)
        if indic_text:
            indic = self._correct_phases(indic_text, context=(tokens, english))
        else:
            indic = {
                "phase1": "", "phase1_guided": "", "phase1_steps": [],
                "phase2": "", "phase2_steps": [], "phase3": "",
                "phase3_trace": {"model_output": None, "decision": "", "output": ""},
                "timings_ms": {"phase1": 0.0, "phase2": 0.0, "phase3": 0.0},
            }

        def splice(corrected: str) -> str:
            """Put English tokens back between the corrected Indic words."""
            corrected_words = iter(corrected.split())
            merged = [token if is_en else next(corrected_words, token) for token, is_en in zip(tokens, english)]
            return " ".join(merged)

        def splice_steps(steps: list[dict[str, Any]], english_step: Any) -> list[dict[str, Any]]:
            indic_steps = iter(steps)
            return [
                english_step(token) if is_en else next(indic_steps)
                for token, is_en in zip(tokens, english)
            ]

        skipped = "English word (language ID); kept unchanged"
        english_count = sum(english)
        decision = indic["phase3_trace"]["decision"]
        return {
            "phase1": splice(indic["phase1"]),
            "phase1_guided": splice(indic["phase1_guided"]),
            "phase1_steps": splice_steps(
                indic["phase1_steps"],
                lambda w: {"original": w, "corrected": w, "distance": 0, "candidates": [], "decision": skipped},
            ),
            "phase2": splice(indic["phase2"]),
            "phase2_steps": splice_steps(
                indic["phase2_steps"],
                lambda w: {"original": w, "model_output": w, "corrected": w, "decision": skipped},
            ),
            "phase3": splice(indic["phase3"]),
            "phase3_trace": {
                **indic["phase3_trace"],
                "decision": (
                    f"{decision + '; ' if decision else ''}"
                    f"{english_count} English word{'s' if english_count != 1 else ''} kept unchanged"
                ),
                "output": splice(indic["phase3"]),
            },
            "languages": languages,
            "timings_ms": indic["timings_ms"],
        }

    @staticmethod
    def _match_case(template: str, word: str) -> str:
        return word[:1].upper() + word[1:] if template[:1].isupper() else word

    def _phase1_word(self, core: str, index: int) -> dict[str, Any]:
        """Phase 1 decision for one word (without punctuation).

        ``kind`` tells the later phases whether they may touch the word: only "candidate"
        and "unseen" words are open to correction; known words, valid spelling variants,
        names, numbers and symbols are never changed.
        """
        lower = core.lower()
        unchanged = {"shown": core, "guided": core, "distance": 0, "candidates": []}
        if not core or not core.isalpha():
            return {**unchanged, "kind": "nonword", "decision": "Not a word (number or symbol); kept as typed"}
        if index > 0 and len(core) > 1 and core[0].isupper() and core[1:].islower():
            return {**unchanged, "kind": "name", "decision": "Capitalised word treated as a name; kept as typed"}
        variant = self._edit_detector.variant_of(lower)
        if variant:
            return {**unchanged, "kind": "variant", "candidates": [variant],
                    "decision": f"Valid spelling variant of '{variant}'; kept as typed"}
        if lower in self._lexicon:
            return {**unchanged, "kind": "known", "candidates": [lower], "decision": "Known word (in the lexicon)"}

        matches = self._edit_detector.plausible_corrections(lower)
        if not matches:
            return {**unchanged, "kind": "unseen",
                    "decision": "Not in the lexicon and no close match; kept as typed (unseen word)"}
        cost, distance, _, best = matches[0]
        candidates = [entry for *_, entry in matches][:4]
        shown = self._match_case(core, best)
        if cost <= HIGH_CONFIDENCE_COST:
            return {"shown": shown, "guided": shown, "distance": distance, "candidates": candidates,
                    "kind": "known",
                    "decision": f"High-confidence match (dropped or repeated letters, cost {cost:.1f})"}
        return {"shown": shown, "guided": core, "distance": distance, "candidates": candidates,
                "kind": "candidate",
                "decision": f"Possible match (cost {cost:.1f}); Phase 2 must confirm it before it is applied"}

    def _correct_phases(
        self,
        text: str,
        context: tuple[list[str], list[bool]] | None = None,
    ) -> dict[str, Any]:
        """Phases 1-3 on Indic words. Caller holds the inference lock.

        ``context`` is the full sentence (all tokens, and which of them are English) so the
        Phase 3 transformer can read English words around the Indic ones.
        """
        started = time.perf_counter()
        tokens = text.split()
        edges = [_TOKEN_EDGES.match(token).groups() for token in tokens]

        # ---- Phase 1: lexicon lookup, edit distance, phonetic (G2P) matching ------------
        decisions = [self._phase1_word(core, index) for index, (_, core, _) in enumerate(edges)]
        phase1_steps: list[dict[str, Any]] = []
        phase1_tokens: list[str] = []
        guided_tokens: list[str] = []
        for token, (prefix, _, suffix), item in zip(tokens, edges, decisions):
            phase1_tokens.append(f"{prefix}{item['shown']}{suffix}")
            guided_tokens.append(f"{prefix}{item['guided']}{suffix}")
            phase1_steps.append(
                {
                    "original": token,
                    "corrected": f"{prefix}{item['shown']}{suffix}",
                    "distance": item["distance"],
                    "candidates": item["candidates"],
                    "decision": item["decision"],
                }
            )
        phase1 = " ".join(phase1_tokens)
        phase1_guided = " ".join(guided_tokens)
        phase1_done = time.perf_counter()

        # ---- Phase 2: Bi-GRU with attention, trusted only when it is confident ------------
        phase2_tokens: list[str] = []
        phase2_steps: list[dict[str, Any]] = []
        accepted_by_gru: dict[int, str] = {}
        for index, ((prefix, core, suffix), item) in enumerate(zip(edges, decisions)):
            guided_core = item["guided"]
            if item["kind"] in {"known", "variant", "name", "nonword"} or guided_core.lower() in self._lexicon:
                corrected, model_output, confidence = guided_core, guided_core, None
                decision = "Passed through: " + item["decision"].split(";")[0].lower()
            else:
                model_output, confidence = self._rnn.predict_word_with_confidence(guided_core.lower())
                model_word = model_output.casefold()
                if (
                    model_word in self._lexicon
                    and gru_candidate_ok(guided_core, model_word)
                    and noise_cost(guided_core, model_word) <= GRU_MAX_NOISE_COST
                ):
                    corrected = self._match_case(guided_core, model_word)
                    accepted_by_gru[index] = model_word
                    decision = f"Confirmed by the GRU (confidence {confidence:.2f})"
                else:
                    corrected = guided_core
                    decision = (
                        f"Kept as typed: the GRU's answer ('{model_output}') is not a plausible fix, so "
                        "this is treated as an unseen word"
                    )
            phase2_tokens.append(f"{prefix}{corrected}{suffix}")
            phase2_steps.append(
                {
                    "original": guided_tokens[index],
                    "model_output": model_output,
                    "confidence": None if confidence is None else round(confidence, 3),
                    "corrected": f"{prefix}{corrected}{suffix}",
                    "decision": decision,
                }
            )
        phase2_done = time.perf_counter()
        phase2 = " ".join(phase2_tokens)

        # ---- Phase 3: rank each ambiguous word's candidates in whole-sentence context ----------
        # A word may only be changed here if Phase 2 confirmed a correction for it.
        open_to_change = [index in accepted_by_gru for index in range(len(tokens))]
        extra = {index: [word] for index, word in accepted_by_gru.items()}
        if context is None:
            ranker_tokens, correctable, ranker_extra = guided_tokens, open_to_change, extra
        else:
            ranker_tokens, correctable, ranker_extra = [], [], {}
            indic_index = 0
            for token, is_english in zip(*context):
                if is_english:
                    ranker_tokens.append(token)
                    correctable.append(False)
                else:
                    if indic_index in extra:
                        ranker_extra[len(ranker_tokens)] = extra[indic_index]
                    ranker_tokens.append(guided_tokens[indic_index])
                    correctable.append(open_to_change[indic_index])
                    indic_index += 1
        ranked = self._ranker.correct_tokens(ranker_tokens, correctable, ranker_extra)
        if context is None:
            phase3_tokens = ranked
        else:
            is_english_flags = context[1]
            phase3_tokens = [word for word, english in zip(ranked, is_english_flags) if not english]
        phase3 = " ".join(phase3_tokens)
        phase3_done = time.perf_counter()

        return {
            "phase1": phase1,
            "phase1_guided": phase1_guided,
            "phase1_steps": phase1_steps,
            "phase2": phase2,
            "phase2_steps": phase2_steps,
            "phase3": phase3,
            "phase3_trace": dict(self._ranker.last_trace),
            "timings_ms": {
                "phase1": round((phase1_done - started) * 1000, 2),
                "phase2": round((phase2_done - phase1_done) * 1000, 2),
                "phase3": round((phase3_done - phase2_done) * 1000, 2),
            },
        }

    def _update_state(self, **updates: Any) -> None:
        with self._state_lock:
            self._state.update(updates)


MAX_LEXICON_DISTANCE = 2
