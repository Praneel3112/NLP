import unittest

from src.correction.pipeline import LEXICON
from src.detection.edit_distance import EditDistanceDetector
from src.preprocessing.noise_generator import NoiseGenerator


class PhoneticPhaseOneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = EditDistanceDetector(list(LEXICON))

    def test_small_typos_use_edit_distance(self) -> None:
        word, is_error, distance = self.detector.detect_and_correct_word("namast")
        self.assertEqual((word, is_error, distance), ("namaste", True, 1))

    def test_heavily_distorted_words_are_rescued_by_g2p(self) -> None:
        # Both are more than two edits away, so plain edit distance gives up on them.
        self.assertEqual(self.detector.detect_and_correct_word("mmmmina")[0], "meena")
        self.assertEqual(self.detector.detect_and_correct_word("naaaaam")[0], "naam")

    def test_unrelated_words_are_left_alone(self) -> None:
        word, is_error, _ = self.detector.detect_and_correct_word("supercalifragilistic")
        self.assertEqual(word, "supercalifragilistic")
        self.assertTrue(is_error)

    def test_candidates_are_lexicon_words_best_first(self) -> None:
        candidates = self.detector.get_candidates("naaann")
        self.assertIn("naan", candidates)
        self.assertTrue(all(candidate in LEXICON for candidate in candidates))
        self.assertEqual(self.detector.get_candidates("kaise"), ["kaise"])


class NoiseEngineTests(unittest.TestCase):
    def test_typos_cover_all_four_operations(self) -> None:
        results = {NoiseGenerator(seed=seed).apply_typos("namaste", 1.0) for seed in range(60)}
        lengths = {len(word) for word in results}
        self.assertIn(6, lengths)  # deletion
        self.assertIn(8, lengths)  # insertion
        self.assertIn(7, lengths)  # substitution / transposition
        self.assertTrue(any(sorted(word) == sorted("namaste") and word != "namaste" for word in results))

    def test_short_words_and_first_letters_are_kept(self) -> None:
        generator = NoiseGenerator(seed=1)
        self.assertEqual(generator.apply_typos("ho en", 1.0), "ho en")
        self.assertTrue(all(NoiseGenerator(seed=s).apply_typos("kaise", 1.0)[0] == "k" for s in range(30)))

    def test_noise_is_deterministic_per_seed(self) -> None:
        self.assertEqual(
            NoiseGenerator(seed=7).generate_noisy_text("namaste aap kaise ho"),
            NoiseGenerator(seed=7).generate_noisy_text("namaste aap kaise ho"),
        )


if __name__ == "__main__":
    unittest.main()
