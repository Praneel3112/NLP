import unittest
from unittest.mock import Mock

from src.correction.service import CorrectionService


class CorrectionServiceTests(unittest.TestCase):
    """The correction phases with a stubbed GRU and ranker (no model files needed)."""

    def setUp(self) -> None:
        self.service = CorrectionService()
        self.service._rnn = Mock()
        self.service._rnn.predict_word_with_confidence.return_value = ("wrong", 0.99)
        self.service._ranker = Mock()
        self.service._ranker.correct_tokens.side_effect = (
            lambda tokens, correctable=None, extra=None: list(tokens)
        )
        self.service._ranker.last_trace = {"decision": "test"}

    def gru_says(self, word: str, confidence: float = 0.99) -> None:
        self.service._rnn.predict_word_with_confidence.return_value = (word, confidence)

    # ---------------------------------------------------------------- known words

    def test_known_words_are_never_touched_and_the_gru_is_not_asked(self) -> None:
        result = self.service.correct("aap kaise")

        self.assertEqual((result["phase1"], result["phase2"], result["phase3"]), ("aap kaise",) * 3)
        self.service._rnn.predict_word_with_confidence.assert_not_called()
        self.assertTrue(all("Known word" in step["decision"] for step in result["phase1_steps"]))

    def test_valid_spelling_variants_are_kept_as_typed(self) -> None:
        result = self.service.correct("acha bahot pani")

        self.assertEqual(result["phase3"], "acha bahot pani")
        self.assertIn("variant of 'accha'", result["phase1_steps"][0]["decision"])
        self.service._rnn.predict_word_with_confidence.assert_not_called()

    # ---------------------------------------------------------------- typing slips

    def test_a_single_typing_slip_is_fixed_without_the_gru(self) -> None:
        result = self.service.correct("namsate aap kase")  # swapped letters, dropped vowel

        self.assertEqual(result["phase1"], "namaste aap kaise")
        self.assertIn("High-confidence", result["phase1_steps"][0]["decision"])
        self.service._rnn.predict_word_with_confidence.assert_not_called()

    def test_an_unusual_edit_needs_the_gru_to_confirm_it(self) -> None:
        # "hesau" lacks a consonant of "hesaru": plausible, but not an everyday slip.
        self.gru_says("hesaru")
        confirmed = self.service.correct("hesau")
        self.assertEqual(confirmed["phase1"], "hesaru")  # the rule-based suggestion
        self.assertEqual(confirmed["phase2"], "hesaru")
        self.assertIn("Confirmed by the GRU", confirmed["phase2_steps"][0]["decision"])

        self.gru_says("notalexiconword")
        rejected = self.service.correct("hesau")
        self.assertEqual(rejected["phase2"], "hesau")
        self.assertIn("Kept as typed", rejected["phase2_steps"][0]["decision"])

    # ---------------------------------------------------------------- unseen words

    def test_unseen_words_are_kept_whatever_the_gru_answers(self) -> None:
        self.gru_says("aap")
        unknown = "supercalifragilistic"

        result = self.service.correct(unknown)

        self.assertEqual((result["phase1"], result["phase2"], result["phase3"]), (unknown,) * 3)
        self.assertIn("unseen word", result["phase1_steps"][0]["decision"])

    def test_capitalised_words_mid_sentence_are_treated_as_names(self) -> None:
        result = self.service.correct("aap Kase ho")

        self.assertEqual(result["phase3"], "aap Kase ho")
        self.assertIn("name", result["phase1_steps"][1]["decision"])

    def test_numbers_and_punctuation_survive(self) -> None:
        result = self.service.correct("kaise ho? 98765, namste!")

        self.assertEqual(result["phase1"], "kaise ho? 98765, namaste!")

    # ---------------------------------------------------------------- phase 0 and mixing

    def test_phase_zero_feeds_correction_without_changing_plain_text(self) -> None:
        plain = self.service.correct("namsate aap")
        with_extras = self.service.correct("namsate aap 😂 plz")

        self.assertEqual(plain["input"], "namsate aap")
        self.assertEqual(plain["phase0"]["normalized"], "namsate aap")
        self.assertEqual(plain["phase2"], "namaste aap")
        self.assertEqual(with_extras["phase0"]["normalized"], "namsate aap please")
        self.assertEqual(with_extras["input"], "namsate aap 😂 plz")
        self.assertIn("emotion", with_extras)

    def test_english_words_are_not_forced_onto_the_indic_lexicon(self) -> None:
        result = self.service.correct("how are you doingggg")

        for phase in ("phase1", "phase2", "phase3"):
            self.assertEqual(result[phase], "how are you doingggg")
        self.assertEqual([item["label"] for item in result["languages"]], ["ENGLISH"] * 4)

    def test_code_mixed_sentence_corrects_only_indic_words(self) -> None:
        result = self.service.correct("namsate how are you aap")

        self.assertEqual(result["phase2"], "namaste how are you aap")
        self.assertEqual(result["phase3"], "namaste how are you aap")  # the stub ranker echoes its input
        self.assertIn("English", result["phase1_steps"][1]["decision"])

    def test_lexicon_words_that_look_english_stay_indic(self) -> None:
        self.assertFalse(self.service.is_english_token("ho"))
        self.assertFalse(self.service.is_english_token("naam"))
        self.assertFalse(self.service.is_english_token("acha"))  # a valid variant
        self.assertTrue(self.service.is_english_token("How,"))
        self.assertTrue(self.service.is_english_token("bro"))  # internet word missing from NLTK

    def test_emoji_only_input_skips_correction(self) -> None:
        result = self.service.correct("😂😂")

        self.assertEqual(result["phase3"], "")
        self.assertEqual(result["phase1_steps"], [])
        self.service._rnn.predict_word_with_confidence.assert_not_called()


if __name__ == "__main__":
    unittest.main()
