import unittest

from src.correction.context_ranker import ContextRanker
from src.correction.pipeline import LEXICON


class RankerWithoutTransformerTests(unittest.TestCase):
    """Behaviour that does not depend on XLM-RoBERTa (the model is not loaded)."""

    def setUp(self) -> None:
        self.ranker = ContextRanker(list(LEXICON))

    def test_reports_when_the_transformer_is_unavailable(self) -> None:
        self.assertFalse(self.ranker.available)
        self.ranker.correct_sentence("namast aap")
        self.assertIn("unavailable", self.ranker.last_trace["decision"])

    def test_words_already_in_the_lexicon_are_never_rewritten(self) -> None:
        self.assertEqual(self.ranker.correct_sentence("aap kaise ho"), "aap kaise ho")

    def test_close_misspellings_are_fixed_without_the_transformer(self) -> None:
        self.assertEqual(self.ranker.correct_sentence("namast aap"), "namaste aap")

    def test_candidates_are_all_lexicon_words(self) -> None:
        candidates = self.ranker.candidates_for("naaann")
        self.assertIn("naan", candidates)
        self.assertNotIn("naaann", candidates)
        self.assertTrue(all(word in LEXICON for word in candidates))

    def test_gru_may_suggest_farther_words_than_phase_one(self) -> None:
        # "mmmmina" is 4 edits from "meena": beyond Phase 1's limit, but allowed for the GRU.
        self.assertIn("meena", self.ranker.candidates_for("mmmmina", extra=["meena"]))
        self.assertNotIn("namaste", self.ranker.candidates_for("mmmmina", extra=["namaste"]))

    def test_english_and_flagged_words_are_left_untouched(self) -> None:
        tokens = ["namast", "good", "morning", "aap"]
        result = self.ranker.correct_tokens(tokens, [True, False, False, True])
        self.assertEqual(result, ["namaste", "good", "morning", "aap"])

    def test_punctuation_and_capitals_survive(self) -> None:
        self.assertEqual(self.ranker.correct_tokens(["Namast,"]), ["Namaste,"])

    def test_gru_suggestions_are_only_used_when_they_are_lexicon_words(self) -> None:
        candidates = self.ranker.candidates_for("kaiso", extra=["notaword", "kaise"])
        self.assertNotIn("notaword", candidates)
        self.assertIn("kaise", candidates)

    def test_unknown_words_with_no_close_candidate_are_kept(self) -> None:
        self.assertEqual(self.ranker.correct_sentence("supercalifragilistic"), "supercalifragilistic")


class RankerWithXlmRobertaTests(unittest.TestCase):
    """Uses the real pretrained model; skipped when it cannot be loaded (e.g. offline)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.ranker = ContextRanker(list(LEXICON))
        if not cls.ranker.load():
            raise unittest.SkipTest(f"XLM-RoBERTa unavailable: {cls.ranker.load_error}")

    def test_scores_candidates_in_context_and_records_the_ranking(self) -> None:
        output = self.ranker.correct_sentence("namast aap kaise ho")
        self.assertEqual(output, "namaste aap kaise ho")
        trace = self.ranker.last_trace
        self.assertEqual(trace["model"], "xlm-roberta-base")
        self.assertIn("XLM-RoBERTa", trace["decision"])
        ranking = trace["rankings"][0]
        self.assertEqual(ranking["chosen"], "namaste")
        self.assertGreater(len(ranking["candidates"]), 1)
        self.assertTrue(all(isinstance(item["score"], float) for item in ranking["candidates"]))

    def test_never_leaves_the_candidate_list(self) -> None:
        output = self.ranker.correct_sentence("nanu hesru ravi")
        for original, corrected in zip("nanu hesru ravi".split(), output.split()):
            self.assertIn(corrected, self.ranker.candidates_for(original))


if __name__ == "__main__":
    unittest.main()
