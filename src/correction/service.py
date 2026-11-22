"""Train and serve the project's sample transliteration correction models."""

import re
import threading
import traceback
from typing import Any

from src.detection import levenshtein as editdistance
import torch

from src.correction.model_store import load_or_train
from src.correction.pipeline import LEXICON
from src.correction.rnn_baseline import Seq2SeqRNN
from src.correction.transformer_models import TransformerCorrector
from src.detection.edit_distance import EditDistanceDetector
from src.emotion.detector import EmotionDetector
from src.lang_id.detector import LanguageIdentifier
from src.preprocessing.slang_normalizer import SlangNormalizer

TRAINING_EPOCHS = 100
_EDGE_PUNCTUATION = re.compile(r"^\W+|\W+$")
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
            "phase3_loss": None,
            "emotion_accuracy": None,
            "emotion_macro_f1": None,
            "lexicon_size": len(LEXICON),
            "models_loaded_from_disk": None,
        }
        self._edit_detector = EditDistanceDetector(list(LEXICON))
        self._lexicon = {word.casefold() for word in LEXICON}
        self._language_id = LanguageIdentifier()
        self._slang = SlangNormalizer()
        self._emotion = EmotionDetector()
        self._emotion_ready = False
        self._rnn: Seq2SeqRNN | None = None
        self._transformer: TransformerCorrector | None = None

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

            self._update_state(stage="Loading saved Phase 2/3 models")
            models = load_or_train(
                TRAINING_EPOCHS,
                on_stage=lambda stage: self._update_state(stage=stage),
            )
            rnn = models["gru"]
            transformer = models["transformer"]
            phase2_loss = models["gru_loss"]
            phase3_loss = models["transformer_loss"]
            self._update_state(
                training_examples=models["training_examples"],
                lexicon_size=len(LEXICON),
                models_loaded_from_disk=models["loaded_from_disk"],
            )

            self._rnn = rnn
            self._transformer = transformer
            self._update_state(
                status="ready",
                stage="Both correction models are ready",
                phase2_loss=round(phase2_loss, 4),
                phase3_loss=round(phase3_loss, 4),
            )
        except Exception as error:
            traceback.print_exc()
            self._update_state(
                status="error",
                stage="Model training failed",
                error=str(error),
            )

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
            if self._rnn is None or self._transformer is None:
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
        if not core or core in self._lexicon or not core.isalpha():
            return False
        variants = {core, _LONG_REPEAT.sub(r"\1\1", core), _LONG_REPEAT.sub(r"\1", core)}
        if variants & self._lexicon:
            return False
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
            indic = self._correct_phases(indic_text)
        else:
            indic = {
                "phase1": "", "phase1_guided": "", "phase1_steps": [],
                "phase2": "", "phase2_steps": [], "phase3": "",
                "phase3_trace": {"model_output": None, "decision": "", "output": ""},
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
        }

    def _correct_phases(self, text: str) -> dict[str, Any]:
        """Phases 1-3, unchanged from the CIE3 version. Caller holds the inference lock."""
        phase1, errors = self._edit_detector.process_sentence(text)
        errors_by_word = {item["original"]: item for item in errors}
        phase1_steps: list[dict[str, Any]] = []
        guided_words: list[str] = []

        for word in text.split():
            candidate, is_error, distance = (
                self._edit_detector.detect_and_correct_word(word)
            )
            if is_error and distance <= 1:
                guided_words.append(candidate)
                error = errors_by_word.get(word)
                phase1_steps.append(
                    {
                        "original": word,
                        "corrected": candidate,
                        "distance": distance,
                        "candidates": (
                            error["candidates"] if error else [candidate]
                        ),
                        "decision": "High-confidence lexicon match",
                    }
                )
            elif not is_error:
                guided_words.append(word)
                phase1_steps.append(
                    {
                        "original": word,
                        "corrected": word,
                        "distance": 0,
                        "candidates": [word],
                        "decision": "Already in sample lexicon",
                    }
                )
            else:
                guided_words.append(word)
                error = errors_by_word.get(word)
                phase1_steps.append(
                    {
                        "original": word,
                        "corrected": word,
                        "distance": distance,
                        "candidates": error["candidates"] if error else [],
                        "decision": "Kept unchanged; not passed as a correction",
                    }
                )

        phase1_guided = " ".join(guided_words)
        phase2_words: list[str] = []
        phase2_steps: list[dict[str, Any]] = []
        lexicon = {word.casefold() for word in LEXICON}
        for word in phase1_guided.split():
            if word.casefold() in lexicon:
                corrected = word
                model_output = word
                decision = "Passed through; already in sample lexicon"
            else:
                model_output = self._rnn.predict_word(word)
                model_word = model_output.casefold()
                nearest_word, is_error, distance = (
                    self._edit_detector.detect_and_correct_word(
                        word,
                        max_distance=MAX_LEXICON_DISTANCE,
                    )
                )
                model_distance = (
                    editdistance.eval(word.casefold(), model_word)
                    if model_word in lexicon
                    else MAX_LEXICON_DISTANCE + 1
                )

                if model_distance <= MAX_LEXICON_DISTANCE:
                    corrected = model_word
                    decision = "Accepted GRU output within lexicon distance"
                elif is_error and distance <= MAX_LEXICON_DISTANCE:
                    corrected = nearest_word
                    decision = "Used lexicon safeguard for out-of-lexicon GRU output"
                else:
                    corrected = word
                    decision = "Kept input; no close lexicon correction"

            phase2_words.append(corrected)
            phase2_steps.append(
                {
                    "original": word,
                    "model_output": model_output,
                    "corrected": corrected,
                    "decision": decision,
                }
            )

        phase2 = " ".join(phase2_words)
        phase3 = self._transformer.correct_sentence(phase1_guided)

        return {
            "phase1": phase1,
            "phase1_guided": phase1_guided,
            "phase1_steps": phase1_steps,
            "phase2": phase2,
            "phase2_steps": phase2_steps,
            "phase3": phase3,
            "phase3_trace": dict(self._transformer.last_trace),
        }

    def _update_state(self, **updates: Any) -> None:
        with self._state_lock:
            self._state.update(updates)


MAX_LEXICON_DISTANCE = 2
