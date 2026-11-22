import unittest

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


if __name__ == "__main__":
    unittest.main()
