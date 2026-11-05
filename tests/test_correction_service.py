import unittest
from unittest.mock import Mock

from src.correction.service import CorrectionService


class CorrectionServicePhaseTwoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = CorrectionService()
        self.service._rnn = Mock()
        self.service._transformer = Mock()
        self.service._transformer.correct_sentence.side_effect = (
            lambda sentence: sentence
        )
        self.service._transformer.last_trace = {"decision": "test"}

    def test_preserves_words_already_in_lexicon(self) -> None:
        self.service._rnn.predict_word.return_value = "wrong"

        result = self.service.correct("aap kaise")

        self.assertEqual(result["phase2"], "aap kaise")
        self.service._rnn.predict_word.assert_not_called()
        self.assertTrue(
            all(
                "already in sample lexicon" in step["decision"]
                for step in result["phase2_steps"]
            )
        )

    def test_accepts_nearby_lexicon_word_predicted_by_gru(self) -> None:
        self.service._rnn.predict_word.return_value = "namaste"

        result = self.service.correct("namsate")

        self.assertEqual(result["phase2"], "namaste")
        self.assertEqual(
            result["phase2_steps"][0]["decision"],
            "Accepted GRU output within lexicon distance",
        )

    def test_uses_nearest_lexicon_word_when_gru_output_is_invalid(self) -> None:
        self.service._rnn.predict_word.return_value = "notalexiconword"

        result = self.service.correct("namsate")

        self.assertEqual(result["phase2"], "namaste")
        self.assertIn("safeguard", result["phase2_steps"][0]["decision"])

    def test_preserves_unknown_word_without_a_close_lexicon_match(self) -> None:
        self.service._rnn.predict_word.return_value = "aap"
        unknown = "supercalifragilistic"

        result = self.service.correct(unknown)

        self.assertEqual(result["phase2"], unknown)
        self.assertIn("Kept input", result["phase2_steps"][0]["decision"])


if __name__ == "__main__":
    unittest.main()
