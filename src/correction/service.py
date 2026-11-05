"""Train and serve the project's sample transliteration correction models."""

import threading
import traceback
from typing import Any

import editdistance
import torch

from src.correction.pipeline import LEXICON, build_training_pairs
from src.correction.rnn_baseline import Seq2SeqRNN
from src.correction.transformer_models import TransformerCorrector
from src.detection.edit_distance import EditDistanceDetector

TRAINING_EPOCHS = 20


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
        }
        self._edit_detector = EditDistanceDetector(list(LEXICON))
        self._rnn: Seq2SeqRNN | None = None
        self._transformer: TransformerCorrector | None = None

    def warm_up(self) -> None:
        """Train both models once, reporting progress and any startup failure."""
        try:
            torch.set_num_threads(1)
            sources, targets = build_training_pairs()
            self._update_state(
                stage=f"Training Phase 2 · character-level GRU ({TRAINING_EPOCHS} epochs)",
                training_examples=len(sources),
            )
            rnn = Seq2SeqRNN(device="cpu", lexicon=list(LEXICON))
            phase2_loss = rnn.train_on_pairs(
                sources,
                targets,
                epochs=TRAINING_EPOCHS,
                learning_rate=1e-3,
                teacher_forcing_ratio=1.0,
            )

            self._update_state(
                stage=f"Training Phase 3 · character Transformer ({TRAINING_EPOCHS} epochs)"
            )
            transformer = TransformerCorrector(device="cpu", lexicon=list(LEXICON))
            phase3_loss = transformer.train_on_pairs(
                sources,
                targets,
                epochs=TRAINING_EPOCHS,
                learning_rate=1e-3,
                batch_size=16,
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

    def correct(self, text: str) -> dict[str, Any]:
        """Run the edit-distance, GRU, and Transformer phases on one sentence."""
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
                "input": text,
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
