"""Trainable emotion classifier: word + character TF-IDF features with Logistic Regression."""

import csv
from pathlib import Path
from typing import Any

import emoji
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from src.emotion.lexicons import EMOTIONS
from src.preprocessing.slang_normalizer import SlangNormalizer

DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "emotion_dataset.csv"
_normalizer = SlangNormalizer()


def prepare_text(text: str) -> str:
    """Expand slang and turn each emoji into a word token, e.g. 😂 -> emo_face_with_tears_of_joy."""
    result = _normalizer.normalize(text)
    emoji_tokens = [
        "emo_" + emoji.demojize(item["emoji"], delimiters=("", "")).replace("-", "_")
        for item in result["emoji"]
    ]
    return " ".join([result["normalized"].casefold(), *emoji_tokens]).strip()


def load_dataset(path: Path = DATASET_PATH) -> tuple[list[str], list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["text"].strip()]
    return [row["text"] for row in rows], [row["label"] for row in rows]


def _build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        # Whole words and word pairs: "gussa", "not happy".
                        ("words", TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"[\w']+", sublinear_tf=True)),
                        # Character pieces cope with spelling variety: bahut / bahot / bhot.
                        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
                    ]
                ),
            ),
            ("model", LogisticRegression(max_iter=3000, C=5.0, class_weight="balanced")),
        ]
    )


class EmotionClassifier:
    """Train on the labelled CSV, report held-out metrics, then refit on all data."""

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._evaluation_model: Pipeline | None = None
        self.held_out: tuple[list[str], list[str]] = ([], [])
        self.metrics: dict[str, Any] | None = None

    @property
    def is_trained(self) -> bool:
        return self._pipeline is not None

    def train(self, path: Path = DATASET_PATH) -> dict[str, Any]:
        texts, labels = load_dataset(path)
        prepared = [prepare_text(text) for text in texts]

        train_idx, test_idx = train_test_split(
            list(range(len(texts))), test_size=0.2, stratify=labels, random_state=42
        )
        train_x = [prepared[i] for i in train_idx]
        train_y = [labels[i] for i in train_idx]
        test_x = [prepared[i] for i in test_idx]
        test_y = [labels[i] for i in test_idx]
        evaluation_model = _build_pipeline().fit(train_x, train_y)
        # Kept so the hybrid detector can be evaluated on the same unseen sentences.
        self.held_out = ([texts[i] for i in test_idx], test_y)
        self._evaluation_model = evaluation_model
        predictions = evaluation_model.predict(test_x)
        cv_scores = cross_val_score(
            _build_pipeline(),
            prepared,
            labels,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring="accuracy",
        )

        self.metrics = {
            "dataset_size": len(texts),
            "train_size": len(train_x),
            "test_size": len(test_x),
            "class_counts": {emotion: labels.count(emotion) for emotion in EMOTIONS},
            "accuracy": round(float(accuracy_score(test_y, predictions)), 4),
            "macro_f1": round(float(f1_score(test_y, predictions, average="macro")), 4),
            "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
            "cv_accuracy_std": round(float(cv_scores.std()), 4),
            "labels": list(EMOTIONS),
            "confusion_matrix": confusion_matrix(test_y, predictions, labels=list(EMOTIONS)).tolist(),
        }
        # Final model uses every labelled example.
        self._pipeline = _build_pipeline().fit(prepared, labels)
        return self.metrics

    def predict_proba(self, text: str, held_out_model: bool = False) -> dict[str, float]:
        """Emotion probabilities; ``held_out_model`` uses the model trained without the test split."""
        pipeline = self._evaluation_model if held_out_model else self._pipeline
        if pipeline is None:
            raise RuntimeError("The emotion classifier has not been trained.")
        probabilities = pipeline.predict_proba([prepare_text(text)])[0]
        by_label = dict(zip(pipeline.classes_, probabilities))
        return {emotion: float(by_label.get(emotion, 0.0)) for emotion in EMOTIONS}
