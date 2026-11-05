"""Phonetic similarity candidate ranking for Romanized words."""

import re
from difflib import SequenceMatcher


class PhoneticDetector:
    """Rank lexicon words by character n-gram and phonetic similarity."""

    _PHONETIC_PAIRS = (
        ("ph", "f"),
        ("bh", "b"),
        ("kh", "k"),
        ("gh", "g"),
        ("dh", "d"),
        ("th", "t"),
        ("sh", "s"),
        ("ch", "c"),
    )
    _SOUNDEX_GROUPS = {
        **dict.fromkeys("bfpv", "1"),
        **dict.fromkeys("cgjkqsxz", "2"),
        **dict.fromkeys("dt", "3"),
        **dict.fromkeys("l", "4"),
        **dict.fromkeys("mn", "5"),
        **dict.fromkeys("r", "6"),
    }
    _PAIR_PATTERN = re.compile(
        "|".join(pair for pair, _ in _PHONETIC_PAIRS)
    )
    _PAIR_REPLACEMENTS = dict(_PHONETIC_PAIRS)
    _NGRAM_SIZE = 2

    def __init__(self, lexicon: list[str]) -> None:
        """Initialize with a lexicon; candidates are normalized to lowercase."""
        self.lexicon = list(dict.fromkeys(word.lower() for word in lexicon if word.strip()))

    def get_phonetic_encoding(self, word: str) -> str:
        """Return a simplified Soundex-like encoding of ``word``.

        Common Romanized Indic digraphs are first reduced to a single sound.
        The first letter is retained, and later letters are converted to
        consonant groups with duplicate adjacent groups removed.
        """
        normalized = self._PAIR_PATTERN.sub(
            lambda match: self._PAIR_REPLACEMENTS[match.group()],
            word.lower(),
        )
        letters = [char for char in normalized if char.isalpha()]
        if not letters:
            return ""

        first_letter = letters[0]
        encoded_digits: list[str] = []
        previous_group = self._SOUNDEX_GROUPS.get(first_letter, "")
        for char in letters[1:]:
            group = self._SOUNDEX_GROUPS.get(char, "")
            if group and group != previous_group:
                encoded_digits.append(group)
            previous_group = group

        return (first_letter + "".join(encoded_digits) + "000")[:4]

    def find_closest_candidates(
        self, word: str, top_k: int = 3
    ) -> list[tuple[str, float]]:
        """Return the top lexicon candidates with similarity scores.

        Scores range from 0 to 1. They combine character bigram Jaccard
        similarity (70%) and phonetic-encoding similarity (30%), sorted from
        highest to lowest score with lexicon order breaking ties.
        """
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if top_k == 0:
            return []

        word_ngrams = self._character_ngrams(word.lower())
        word_encoding = self.get_phonetic_encoding(word)
        scored_candidates: list[tuple[str, float]] = []

        for candidate in self.lexicon:
            candidate_ngrams = self._character_ngrams(candidate)
            union = word_ngrams | candidate_ngrams
            jaccard_similarity = (
                len(word_ngrams & candidate_ngrams) / len(union) if union else 1.0
            )

            candidate_encoding = self.get_phonetic_encoding(candidate)
            phonetic_similarity = SequenceMatcher(
                None, word_encoding, candidate_encoding
            ).ratio()
            score = 0.7 * jaccard_similarity + 0.3 * phonetic_similarity
            scored_candidates.append((candidate, score))

        scored_candidates.sort(key=lambda item: -item[1])
        return scored_candidates[:top_k]

    @classmethod
    def _character_ngrams(cls, word: str) -> set[str]:
        padded_word = f"^{word}$"
        size = cls._NGRAM_SIZE
        return {
            padded_word[index : index + size]
            for index in range(len(padded_word) - size + 1)
        }