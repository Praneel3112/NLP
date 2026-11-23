import unittest

from src.evaluation.metrics import EvaluationSuite

detect = EvaluationSuite.calculate_detection_scores


class DetectionScoreTests(unittest.TestCase):
    reference = ["a b c d"]
    noisy = ["a x c y"]

    def test_changing_nothing_scores_zero(self) -> None:
        self.assertEqual(detect(self.reference, self.noisy, self.noisy), {"Precision": 0.0, "Recall": 0.0, "F1": 0.0})

    def test_fixing_one_of_two_errors(self) -> None:
        scores = detect(self.reference, self.noisy, ["a b c y"])
        self.assertEqual((scores["Precision"], scores["Recall"]), (1.0, 0.5))
        self.assertAlmostEqual(scores["F1"], 2 / 3)

    def test_fixing_everything_scores_one(self) -> None:
        self.assertEqual(detect(self.reference, self.noisy, self.reference)["F1"], 1.0)

    def test_breaking_a_valid_word_is_a_false_positive(self) -> None:
        scores = detect(self.reference, self.noisy, ["q b c y"])  # fixed 'x', broke 'a'
        self.assertEqual(scores["Precision"], 0.5)
        self.assertEqual(scores["Recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
