"""Generate informal transliteration noise for Romanized Indic text."""

import random
import re


class NoiseGenerator:
    """Simulate informal digital typing noise in Romanized Indic text."""

    _VOWELS = frozenset("aeiou")
    _PHONETIC_MAP = {
        "ph": "f",
        "bh": "b",
        "kh": "k",
        "gh": "g",
        "dh": "d",
        "ee": "i",
        "oo": "u",
    }

    def __init__(self, seed: int = 42) -> None:
        """Initialize a reproducible random generator with the given seed."""
        self._random = random.Random(seed)

    def drop_vowels(self, text: str, p: float = 0.25) -> str:
        """Drop non-initial vowels independently with probability ``p``."""
        self._validate_probability(p)
        result: list[str] = []
        at_word_start = True

        for char in text:
            if char.isalnum():
                if (
                    not at_word_start
                    and char.lower() in self._VOWELS
                    and self._random.random() < p
                ):
                    continue
                at_word_start = False
            else:
                at_word_start = True
            result.append(char)

        return "".join(result)

    def repeat_characters(self, text: str, p: float = 0.15) -> str:
        """Repeat alphabetic characters one to three times with probability ``p``."""
        self._validate_probability(p)
        result: list[str] = []
        for char in text:
            result.append(char)
            if char.isalpha() and self._random.random() < p:
                result.append(char * self._random.randint(1, 3))
        return "".join(result)

    def apply_phonetic_substitutions(self, text: str) -> str:
        """Replace common phonetic pairs while preserving initial capitalization."""

        def substitute(match: re.Match[str]) -> str:
            pair = match.group().lower()
            replacement = self._PHONETIC_MAP[pair]
            return replacement.capitalize() if match.group()[0].isupper() else replacement

        pattern = re.compile(
            "|".join(self._PHONETIC_MAP),
            re.IGNORECASE,
        )
        return pattern.sub(substitute, text)

    def generate_noisy_text(self, text: str) -> str:
        """Apply phonetic substitutions, vowel dropping, and character repetition."""
        text = self.apply_phonetic_substitutions(text)
        text = self.drop_vowels(text)
        return self.repeat_characters(text)

    @staticmethod
    def _validate_probability(p: float) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be between 0.0 and 1.0")
