import unittest

from src.correction.pipeline import LEXICON, VARIANTS
from src.detection.edit_distance import EditDistanceDetector
from src.detection.noise_distance import noise_cost
from src.preprocessing.noise_generator import NoiseGenerator


class PhaseOneDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = EditDistanceDetector(list(LEXICON), VARIANTS)

    def test_small_typos_use_edit_distance(self) -> None:
        word, is_error, distance = self.detector.detect_and_correct_word("namast")
        self.assertEqual((word, is_error, distance), ("namaste", True, 1))

    def test_heavily_distorted_words_are_rescued_by_g2p(self) -> None:
        # Both are more than two edits away, so plain edit distance gives up on them.
        self.assertEqual(self.detector.detect_and_correct_word("naaaaam")[0], "naam")
        self.assertEqual(self.detector.detect_and_correct_word("kaaaaam")[0], "kaam")

    def test_unrelated_words_are_left_alone(self) -> None:
        word, is_error, _ = self.detector.detect_and_correct_word("supercalifragilistic")
        self.assertEqual(word, "supercalifragilistic")
        self.assertTrue(is_error)

    def test_candidates_are_lexicon_words_best_first(self) -> None:
        candidates = self.detector.get_candidates("naaann")
        self.assertIn("naan", candidates)
        self.assertTrue(all(candidate in LEXICON for candidate in candidates))
        self.assertEqual(self.detector.get_candidates("kaise"), ["kaise"])

    def test_valid_spelling_variants_are_known_not_errors(self) -> None:
        for variant, canonical in (("acha", "accha"), ("bahot", "bahut"), ("pani", "paani")):
            word, is_error, distance = self.detector.detect_and_correct_word(variant)
            self.assertEqual((word, is_error, distance), (variant, False, 0))
            self.assertEqual(self.detector.variant_of(variant), canonical)
            self.assertTrue(self.detector.is_known(variant))

    def test_long_vowels_written_doubled_or_single_are_alternative_spellings(self) -> None:
        for typed, known in (("pyaari", "pyari"), ("saree", "sari"), ("aapka", "apka"), ("kitaab", "kitab")):
            self.assertEqual(self.detector.variant_of(typed), known, typed)
            self.assertEqual(self.detector.detect_and_correct_word(typed), (typed, False, 0))

    def test_chat_stretching_of_letters_is_still_corrected(self) -> None:
        # Three or more repeated letters are stretching, not an alternative spelling.
        self.assertIsNone(self.detector.variant_of("naaaam"))
        self.assertEqual(self.detector.detect_and_correct_word("naaaam")[0], "naam")

    def test_valid_words_the_lexicon_never_saw_are_not_forced_onto_lookalikes(self) -> None:
        # Names, nouns and festivals: each is a few edits from some known word, yet correct as typed.
        for word in ("rahul", "tamasha", "bhajan", "pongal", "dasara", "pariksha", "kabaddi", "durga"):
            self.assertEqual(self.detector.detect_and_correct_word(word)[0], word, word)

    def test_two_letter_words_are_valid_but_never_correction_targets(self) -> None:
        self.assertIn("ho", LEXICON)  # a known word...
        self.assertEqual(self.detector.detect_and_correct_word("ho"), ("ho", False, 0))
        self.assertFalse(any(len(entry) < 3 for entry in self.detector._targets))  # ...but no target
        self.assertTrue(all(len(c) >= 3 for c in self.detector.get_candidates("hsa")))

    def test_corrections_beyond_two_edits_only_happen_for_stretched_words(self) -> None:
        import re

        for word in ("kitaabein", "namastee", "bhaaaaiyaa", "kaaaaam", "pariksha"):
            corrected, _, distance = self.detector.detect_and_correct_word(word)
            if corrected != word and not re.search(r"(.)\1{2,}", word):
                self.assertLessEqual(distance, 2, word)

    def test_phonetic_distance_compares_sound_codes(self) -> None:
        self.assertEqual(self.detector.phonetic_distance("pyar", "pyaar"), 0)
        self.assertGreater(self.detector.phonetic_distance("pyar", "dost"), 0)


class NoiseAwareDistanceTests(unittest.TestCase):
    def test_everyday_typing_slips_are_cheap(self) -> None:
        self.assertLessEqual(noise_cost("kase", "kaise"), 0.5)  # dropped vowel
        self.assertLessEqual(noise_cost("naaam", "naam"), 0.5)  # repeated letter
        self.assertLessEqual(noise_cost("namsate", "namaste"), 0.8)  # swapped letters

    def test_a_different_word_is_expensive(self) -> None:
        self.assertGreater(noise_cost("thali", "khali"), 1.4)  # t and k are not neighbouring keys
        self.assertGreater(noise_cost("bhajan", "bhaya"), 1.4)
        self.assertGreater(noise_cost("tamasha", "hamesha"), 1.4)

    def test_distance_is_zero_for_identical_words(self) -> None:
        self.assertEqual(noise_cost("namaste", "namaste"), 0.0)


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

    def test_vowels_are_only_dropped_from_words_longer_than_two_letters(self) -> None:
        self.assertEqual(NoiseGenerator(seed=3).drop_vowels("aap ho en ek", 1.0), "ap ho en ek")
        self.assertEqual(NoiseGenerator(seed=3).drop_vowels("namaste", 1.0), "nmst")

    def test_the_last_letter_is_repeated_as_in_the_paper(self) -> None:
        stretched = {NoiseGenerator(seed=s).repeat_final_letter("kaise", 1.0) for s in range(20)}
        self.assertTrue(all(word.startswith("kaise") and set(word[4:]) == {"e"} for word in stretched))
        self.assertEqual(NoiseGenerator(seed=1).repeat_final_letter("kaise", 0.0), "kaise")

    def test_phonetic_substitutions_apply_with_the_given_probability(self) -> None:
        self.assertEqual(NoiseGenerator(seed=1).apply_phonetic_substitutions("bhai", 1.0), "bai")
        self.assertEqual(NoiseGenerator(seed=1).apply_phonetic_substitutions("bhai", 0.0), "bhai")

    def test_noise_is_deterministic_per_seed(self) -> None:
        self.assertEqual(
            NoiseGenerator(seed=7).generate_noisy_text("namaste aap kaise ho"),
            NoiseGenerator(seed=7).generate_noisy_text("namaste aap kaise ho"),
        )


if __name__ == "__main__":
    unittest.main()
