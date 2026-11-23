import unittest

from src.correction.pipeline import LEXICON
from src.preprocessing.slang_normalizer import SLANG_MAP, SlangNormalizer


class SlangNormalizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.normalizer = SlangNormalizer()

    def test_text_without_slang_or_emoji_is_unchanged(self) -> None:
        for text in ("namast aap kaise ho", "nanna hesaru ravi", "en  peyar, meena!", "amar naam suman"):
            result = self.normalizer.normalize(text)
            self.assertEqual(result["normalized"], text)
            self.assertFalse(result["changed"])

    def test_expands_chat_and_indic_slang(self) -> None:
        result = self.normalizer.normalize("bht gussa aa raha h plz")
        self.assertEqual(result["normalized"], "bahut gussa aa raha hai please")
        self.assertEqual(
            [(r["original"], r["replacement"]) for r in result["replacements"]],
            [("bht", "bahut"), ("h", "hai"), ("plz", "please")],
        )

    def test_keeps_punctuation_and_case_around_slang(self) -> None:
        self.assertEqual(self.normalizer.normalize("Plz, come!")["normalized"], "Please, come!")
        self.assertEqual(self.normalizer.normalize("PLZZZ")["normalized"], "PLEASE")

    def test_removes_emoji_from_correction_text_but_reports_them(self) -> None:
        result = self.normalizer.normalize("aap kaise ho😂 ❤️")
        self.assertEqual(result["normalized"], "aap kaise ho")
        self.assertEqual([e["emoji"] for e in result["emoji"]], ["😂", "❤️"])

    def test_never_rewrites_correction_lexicon_words(self) -> None:
        protected = {word.casefold() for word in LEXICON}
        self.assertFalse(protected & set(SLANG_MAP))
        text = " ".join(LEXICON)
        self.assertEqual(self.normalizer.normalize(text)["normalized"], text)


if __name__ == "__main__":
    unittest.main()
