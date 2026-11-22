import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.correction import model_store
from src.correction.pipeline import LEXICON, LEXICON_INFO, build_training_pairs

ORIGINAL_CIE3_LEXICON = (
    "namaste", "aap", "kaise", "ho", "nanna", "hesaru", "ravi",
    "en", "peyar", "meena", "amar", "naam", "suman",
)


class LexiconTests(unittest.TestCase):
    def test_original_words_come_first_so_their_training_pairs_are_unchanged(self) -> None:
        self.assertEqual(LEXICON[:13], ORIGINAL_CIE3_LEXICON)

    def test_lexicon_covers_all_four_languages_without_duplicates(self) -> None:
        self.assertEqual(len(LEXICON), len(set(LEXICON)))
        self.assertGreater(len(LEXICON), 250)
        languages = " ".join(info["language"] for info in LEXICON_INFO.values())
        for language in ("Hindi", "Kannada", "Tamil", "Bengali"):
            self.assertIn(language, languages)

    def test_common_indic_words_are_known(self) -> None:
        for word in ("yaar", "bahut", "gussa", "tumba", "romba", "bhalo"):
            self.assertIn(word, LEXICON)

    def test_four_noisy_pairs_per_word(self) -> None:
        sources, targets = build_training_pairs()
        self.assertEqual(len(sources), 4 * len(LEXICON) + 1)
        self.assertEqual(len(sources), len(targets))


class ModelStoreTests(unittest.TestCase):
    def test_trains_once_then_reloads_identical_models(self) -> None:
        tiny_pairs = (["namste", "ap", "kase"], ["namaste", "aap", "kaise"])
        with tempfile.TemporaryDirectory() as folder, patch.object(
            model_store, "build_training_pairs", return_value=tiny_pairs
        ):
            first = model_store.load_or_train(1, models_dir=Path(folder))
            second = model_store.load_or_train(1, models_dir=Path(folder))

            self.assertFalse(first["loaded_from_disk"])
            self.assertTrue(second["loaded_from_disk"])
            for word in ("namste", "kase"):
                self.assertEqual(first["gru"].predict_word(word), second["gru"].predict_word(word))
                self.assertEqual(
                    first["transformer"]._predict_word(word), second["transformer"]._predict_word(word)
                )

            # Different settings give a different fingerprint, so the models retrain.
            third = model_store.load_or_train(2, models_dir=Path(folder))
            self.assertFalse(third["loaded_from_disk"])

    def test_transformer_never_rewrites_a_correct_lexicon_word(self) -> None:
        tiny_pairs = (["santsha", "santhosam"], ["santosha", "santhosham"])
        with tempfile.TemporaryDirectory() as folder, patch.object(
            model_store, "build_training_pairs", return_value=tiny_pairs
        ):
            transformer = model_store.load_or_train(1, models_dir=Path(folder))["transformer"]
        with patch.object(transformer, "_predict_word", return_value="santhosham"):
            self.assertEqual(transformer.correct_sentence("naanu santosha"), "naanu santosha")


if __name__ == "__main__":
    unittest.main()
