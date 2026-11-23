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

    # Neighbouring keys on a QWERTY keyboard, used for realistic substitutions/insertions.
    _KEY_NEIGHBOURS = {
        "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh",
        "u": "yij", "i": "uok", "o": "ipl", "p": "ol", "a": "qsz", "s": "awdx",
        "d": "sfec", "f": "dgrv", "g": "fhtb", "h": "gjyn", "j": "hkum",
        "k": "jli", "l": "ko", "z": "ax", "x": "zsc", "c": "xdv", "v": "cfb",
        "b": "vgn", "n": "bhm", "m": "nj",
    }
    TYPO_OPERATIONS = ("insertion", "substitution", "deletion", "transposition")

    def apply_typos(self, text: str, p: float = 0.25) -> str:
        """Apply one typing mistake to each word of 3+ letters with probability ``p``.

        The mistake is one of: inserting a neighbouring key, substituting a
        neighbouring key, deleting a letter, or swapping two adjacent letters.
        """
        self._validate_probability(p)

        def mistype(match: re.Match[str]) -> str:
            word = match.group()
            if len(word) < 3 or self._random.random() >= p:
                return word
            operation = self._random.choice(self.TYPO_OPERATIONS)
            # Never touch the first letter; a swap needs two letters after it to stay clear of it.
            position = self._random.randrange(2 if operation == "transposition" else 1, len(word))
            neighbours = self._KEY_NEIGHBOURS.get(word[position - 1].lower(), "e")
            if operation == "insertion":
                return word[:position] + self._random.choice(neighbours) + word[position:]
            if operation == "substitution":
                original = word[position].lower()
                options = self._KEY_NEIGHBOURS.get(original, "e")
                return word[:position] + self._random.choice(options) + word[position + 1 :]
            if operation == "deletion":
                return word[:position] + word[position + 1 :]
            # transposition: swap the letter at ``position`` with the one before it
            return word[: position - 1] + word[position] + word[position - 1] + word[position + 1 :]

        return re.sub(r"[A-Za-z]+", mistype, text)

    def generate_noisy_text(self, text: str, typo_p: float = 0.25) -> str:
        """Apply phonetic substitutions, vowel dropping, repetition and typos."""
        text = self.apply_phonetic_substitutions(text)
        text = self.drop_vowels(text)
        text = self.repeat_characters(text)
        return self.apply_typos(text, typo_p)

    @staticmethod
    def _validate_probability(p: float) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be between 0.0 and 1.0")
