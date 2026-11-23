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
        """Drop non-initial vowels with probability ``p`` in words of more than two characters."""
        self._validate_probability(p)

        def drop(match: re.Match[str]) -> str:
            word = match.group()
            if len(word) <= 2:
                return word
            kept = [word[0]]
            for char in word[1:]:
                if char.lower() in self._VOWELS and self._random.random() < p:
                    continue
                kept.append(char)
            return "".join(kept)

        return re.sub(r"[^\W_]+", drop, text)

    def repeat_final_letter(self, text: str, p: float = 0.15) -> str:
        """Duplicate a word's last letter one to three times with probability ``p`` (paper, eq. 1)."""
        self._validate_probability(p)

        def repeat(match: re.Match[str]) -> str:
            word = match.group()
            if self._random.random() < p:
                return word + word[-1] * self._random.randint(1, 3)
            return word

        return re.sub(r"[A-Za-z]+", repeat, text)

    def repeat_characters(self, text: str, p: float = 0.05) -> str:
        """Repeat letters inside words one to three times with probability ``p`` per letter.

        An extension of the paper's final-letter repetition: chat text also stretches
        vowels in the middle of words ("yaaar", "naaam").
        """
        self._validate_probability(p)
        result: list[str] = []
        for char in text:
            result.append(char)
            if char.isalpha() and self._random.random() < p:
                result.append(char * self._random.randint(1, 3))
        return "".join(result)

    def apply_phonetic_substitutions(self, text: str, p: float = 0.30) -> str:
        """Replace each common phonetic pair with probability ``p``, keeping initial capitals."""
        self._validate_probability(p)

        def substitute(match: re.Match[str]) -> str:
            if self._random.random() >= p:
                return match.group()
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
        """Apply phonetic substitutions, vowel dropping, letter repetition and typing mistakes."""
        text = self.apply_phonetic_substitutions(text)
        text = self.drop_vowels(text)
        text = self.repeat_final_letter(text)
        text = self.repeat_characters(text)
        return self.apply_typos(text, typo_p)

    @staticmethod
    def _validate_probability(p: float) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be between 0.0 and 1.0")
