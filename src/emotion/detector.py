"""Combine the rule engine and the trained classifier into one emotion prediction."""

from typing import Any

from sklearn.metrics import accuracy_score, f1_score

from src.emotion.classifier import EmotionClassifier
from src.emotion.lexicons import EMOTION_EMOJI, EMOTIONS
from src.emotion.rule_scorer import RuleEmotionScorer
from src.preprocessing.slang_normalizer import SlangNormalizer

RULE_WEIGHT = 0.5

# When no emoji, slang or emotion word is found, the trained model decides alone. A small model is
# unreliable on text unlike its training sentences, so below this confidence the answer is "neutral,
# not sure" instead of a random strong emotion.
UNCERTAIN_BELOW = 0.5


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


def _decide(rule_result: dict[str, Any], model_scores: dict[str, float]) -> dict[str, Any]:
    """Pick the final label, falling back to "neutral, not sure" when only a weak model guess exists."""
    combined, rule_weight = _combine(rule_result, model_scores)
    label = _argmax(combined)
    guess = None
    if rule_weight == 0.0 and label != "neutral" and combined[label] < UNCERTAIN_BELOW:
        guess = {"label": label, "confidence": round(combined[label], 4)}
        label = "neutral"
    return {"label": label, "combined": combined, "rule_weight": rule_weight, "guess": guess}


class EmotionDetector:
    """Hybrid detector built from an explainable rule engine and a trained classifier."""

    def __init__(self, classifier: EmotionClassifier | None = None) -> None:
        self.rules = RuleEmotionScorer()
        self.classifier = classifier or EmotionClassifier()
        self._normalizer = SlangNormalizer()
        self.metrics: dict[str, Any] | None = None

    def _rule_score(self, text: str) -> dict[str, Any]:
        """Score with the rules on the slang-expanded text ("gr8" -> "great"), emoji kept."""
        prepared = self._normalizer.normalize(text)
        emoji_part = "".join(item["emoji"] for item in prepared["emoji"])
        return self.rules.score(f"{prepared['normalized']} {emoji_part}".strip())

    def train(self) -> dict[str, Any]:
        """Train the classifier and compare rules / model / hybrid on the held-out sentences."""
        metrics = dict(self.classifier.train())
        texts, gold = self.classifier.held_out
        predictions: dict[str, list[str]] = {"rules_only": [], "model_only": [], "hybrid": []}
        for text in texts:
            rule_result = self._rule_score(text)
            model_scores = self.classifier.predict_proba(text, held_out_model=True)
            predictions["rules_only"].append(_argmax(rule_result["probabilities"]))
            predictions["model_only"].append(_argmax(model_scores))
            predictions["hybrid"].append(_decide(rule_result, model_scores)["label"])

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
        rule_result = self._rule_score(text)
        model_scores = self.classifier.predict_proba(text)
        decision = _decide(rule_result, model_scores)
        combined, rule_weight, label = decision["combined"], decision["rule_weight"], decision["label"]
        uncertain = decision["guess"] is not None

        return {
            "label": label,
            "emoji": EMOTION_EMOJI[label],
            "confidence": round(combined[label], 4),
            "uncertain": uncertain,
            "model_guess": decision["guess"],
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
