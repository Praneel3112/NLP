"""Combine the rule engine and the trained classifier into one emotion prediction."""

from typing import Any

from sklearn.metrics import accuracy_score, f1_score

from src.emotion.classifier import EmotionClassifier
from src.emotion.lexicons import EMOTION_EMOJI, EMOTIONS
from src.emotion.rule_scorer import RuleEmotionScorer

RULE_WEIGHT = 0.5


def _combine(rule_result: dict[str, Any], model_scores: dict[str, float]) -> tuple[dict[str, float], float]:
    """final = 0.5 * rules + 0.5 * model; the model decides alone when no rule fired."""
    rule_weight = RULE_WEIGHT if rule_result["has_evidence"] else 0.0
    combined = {
        emotion: rule_weight * rule_result["probabilities"][emotion]
        + (1 - rule_weight) * model_scores[emotion]
        for emotion in EMOTIONS
    }
    return combined, rule_weight


def _argmax(scores: dict[str, float]) -> str:
    return max(scores, key=scores.get)


class EmotionDetector:
    """Hybrid detector built from an explainable rule engine and a trained classifier."""

    def __init__(self, classifier: EmotionClassifier | None = None) -> None:
        self.rules = RuleEmotionScorer()
        self.classifier = classifier or EmotionClassifier()
        self.metrics: dict[str, Any] | None = None

    def train(self) -> dict[str, Any]:
        """Train the classifier and compare rules / model / hybrid on the held-out sentences."""
        metrics = dict(self.classifier.train())
        texts, gold = self.classifier.held_out
        predictions: dict[str, list[str]] = {"rules_only": [], "model_only": [], "hybrid": []}
        for text in texts:
            rule_result = self.rules.score(text)
            model_scores = self.classifier.predict_proba(text, held_out_model=True)
            predictions["rules_only"].append(_argmax(rule_result["probabilities"]))
            predictions["model_only"].append(_argmax(model_scores))
            predictions["hybrid"].append(_argmax(_combine(rule_result, model_scores)[0]))

        metrics["comparison"] = {
            name: {
                "accuracy": round(float(accuracy_score(gold, predicted)), 4),
                "macro_f1": round(float(f1_score(gold, predicted, average="macro")), 4),
            }
            for name, predicted in predictions.items()
        }
        self.metrics = metrics
        return metrics

    def detect(self, text: str) -> dict[str, Any]:
        rule_result = self.rules.score(text)
        model_scores = self.classifier.predict_proba(text)
        combined, rule_weight = _combine(rule_result, model_scores)
        label = _argmax(combined)

        return {
            "label": label,
            "emoji": EMOTION_EMOJI[label],
            "confidence": round(combined[label], 4),
            "method": "hybrid" if rule_weight else "model_only",
            "rule_weight": rule_weight,
            "scores": {k: round(v, 4) for k, v in sorted(combined.items(), key=lambda kv: -kv[1])},
            "rule_scores": {k: round(v, 4) for k, v in rule_result["probabilities"].items()},
            "model_scores": {k: round(v, 4) for k, v in model_scores.items()},
            "intensity": rule_result["intensity"],
            "intensity_score": rule_result["intensity_score"],
            "signals": rule_result["signals"],
            "evidence": rule_result["evidence"],
        }
