import unittest
from unittest.mock import patch

from src.emotion.detector import EmotionDetector
from src.emotion.rule_scorer import RuleEmotionScorer


class RuleScorerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rules = RuleEmotionScorer()

    def top(self, text: str) -> str:
        probabilities = self.rules.score(text)["probabilities"]
        return max(probabilities, key=probabilities.get)

    def test_emoji_cues(self) -> None:
        self.assertEqual(self.top("😂😂"), "joy")
        self.assertEqual(self.top("😡"), "anger")
        self.assertEqual(self.top("❤️"), "love")

    def test_slang_and_indic_words(self) -> None:
        self.assertEqual(self.top("lol"), "joy")
        self.assertEqual(self.top("bht gussa aa raha h"), "anger")
        self.assertEqual(self.top("tumba bhaya aagtide"), "fear")

    def test_negation_in_english_and_hindi(self) -> None:
        for text in ("not happy", "khush nahi hoon"):
            result = self.rules.score(text)
            self.assertNotEqual(self.top(text), "joy", text)
            self.assertTrue(result["evidence"][0]["negated"], text)

    def test_intensity_increases_with_markers(self) -> None:
        calm = self.rules.score("happy")["intensity_score"]
        loud = self.rules.score("SOOO happyyy!!!")["intensity_score"]
        self.assertGreater(loud, calm)

    def test_evidence_lists_emoji(self) -> None:
        evidence = self.rules.score("bht gussa 😡")["evidence"]
        self.assertIn(("😡", "emoji", "anger"), [(e["token"], e["type"], e["emotion"]) for e in evidence])

    def test_no_cues_means_no_evidence(self) -> None:
        self.assertFalse(self.rules.score("the bus leaves at nine")["has_evidence"])


class EmotionDetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.detector = EmotionDetector()
        cls.metrics = cls.detector.train()

    def test_hybrid_predictions(self) -> None:
        self.assertEqual(self.detector.detect("bht gussa aa raha h 😡😡")["label"], "anger")
        self.assertEqual(self.detector.detect("namast aap kaise ho 😂")["label"], "joy")
        self.assertNotEqual(self.detector.detect("not happy yaar 😔")["label"], "joy")

    def test_model_only_when_no_rule_fires(self) -> None:
        self.assertEqual(self.detector.detect("the bus leaves at nine")["method"], "model_only")

    def test_held_out_accuracy(self) -> None:
        self.assertGreaterEqual(self.metrics["accuracy"], 0.65)
        self.assertGreaterEqual(self.metrics["comparison"]["hybrid"]["accuracy"], 0.80)
        self.assertGreaterEqual(
            self.metrics["comparison"]["hybrid"]["accuracy"],
            self.metrics["comparison"]["model_only"]["accuracy"],
        )


class EmotionRobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.detector = EmotionDetector()
        cls.detector.train()

    def test_a_weak_model_only_guess_becomes_neutral_not_sure(self) -> None:
        weak = {"joy": 0.05, "sadness": 0.05, "anger": 0.4, "fear": 0.1, "surprise": 0.1, "love": 0.1, "neutral": 0.2}
        with patch.object(self.detector.classifier, "predict_proba", return_value=weak):
            result = self.detector.detect("tum kahan ho abhi")
        self.assertEqual(result["label"], "neutral")
        self.assertTrue(result["uncertain"])
        self.assertEqual(result["model_guess"]["label"], "anger")

    def test_a_confident_model_only_guess_is_kept(self) -> None:
        strong = {"joy": 0.05, "sadness": 0.05, "anger": 0.7, "fear": 0.05, "surprise": 0.05, "love": 0.05, "neutral": 0.05}
        with patch.object(self.detector.classifier, "predict_proba", return_value=strong):
            result = self.detector.detect("tum kahan ho abhi")
        self.assertEqual((result["label"], result["uncertain"]), ("anger", False))

    def test_rules_see_slang_expanded_text(self) -> None:
        self.assertEqual(self.detector.detect("hi u r so gr8 yaar")["label"], "joy")

    def test_everyday_phrases_with_spelling_variants(self) -> None:
        self.assertEqual(self.detector.detect("mera dost bahut acha hai")["label"], "joy")
        self.assertEqual(self.detector.detect("mujhe samosa pasand hai")["label"], "joy")
        self.assertEqual(self.detector.detect("pyaari si ladki")["label"], "love")
        self.assertEqual(self.detector.detect("mera dil toot gaya")["label"], "sadness")

    def test_accuracy_on_fresh_sentences(self) -> None:
        import csv
        from pathlib import Path as _Path

        path = _Path(__file__).resolve().parents[1] / "data" / "emotion_validation.csv"
        rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
        right = sum(self.detector.detect(row["text"])["label"] == row["label"] for row in rows)
        self.assertGreaterEqual(right / len(rows), 0.85)


if __name__ == "__main__":
    unittest.main()
