"""Phase 1: edit-distance and phonetic (G2P) detection and correction of spelling errors.

The detector is deliberately cautious. A word is only "corrected" to a lexicon word when
the match is close in *relative* terms, so a valid word the lexicon has never seen (a
name, a noun) is left alone instead of being forced onto a similar-looking known word.
"""

import re

from src.detection import levenshtein as editdistance
from src.detection.noise_distance import noise_cost
from src.detection.phonetic_sim import PhoneticDetector

MAX_DISTANCE = 2  # tau in the report: at most two edits
MAX_RELATIVE_DISTANCE = 0.34  # edits divided by the length of the longer word
HIGH_CONFIDENCE_COST = 0.8  # one typing slip (dropped/repeated letter, neighbour key, swap): applied outright
RESCUE_COST = 1.0  # cost assigned to a phonetic rescue so that it needs confirmation
NOISE_COST_LIMIT = 1.4  # the edits must be explainable as typing slips (see noise_distance.py)
MIN_TYPED_LENGTH = 2  # one-letter tokens are never corrected
MIN_TARGET_LENGTH = 3  # 1-2 letter lexicon words are valid words but never correction targets

# Heavily distorted words ("mmmmina", "naaaaam") that edit distance cannot reach are matched
# by sound instead, but only when they show the tell-tale letter stretching.
RESCUE_MIN_SCORE = 0.5
RESCUE_MAX_RELATIVE_DISTANCE = 0.6
_STRETCHED = re.compile(r"(.)\1{2,}")


def long_vowel_key(word: str) -> str:
    """Spelling-insensitive key: exactly doubled long vowels collapse (aa -> a, ee -> i, oo -> u).

    Romanization writes the same sound either way ("paani"/"pani", "sari"/"saree", "pyaari"/"pyari"),
    so words with equal keys are alternative spellings. Three or more repeated letters ("naaam") are
    chat-style stretching, a different matter, and are left alone by this key.
    """
    word = re.sub(r"(?<!e)ee(?!e)", "i", word.lower())
    word = re.sub(r"(?<!o)oo(?!o)", "u", word)
    word = re.sub(r"(?<!i)ii(?!i)", "i", word)
    word = re.sub(r"(?<!u)uu(?!u)", "u", word)
    return re.sub(r"(?<!a)aa(?!a)", "a", word)


def relative_distance(first: str, second: str) -> float:
    """Edit distance divided by the length of the longer word."""
    longest = max(len(first), len(second)) or 1
    return editdistance.eval(first, second) / longest


def gru_distance_limit(word: str) -> int:
    """How many edits away a lexicon word suggested by the GRU may be from what was typed.

    Phase 1 stops at 2 edits. The GRU exists to handle noisier words, so it may go
    further for longer words (half the word's length), but its answer must still be a
    lexicon word, which keeps it from inventing spellings.
    """
    return max(2, (len(word) + 1) // 2)


def gru_candidate_ok(typed: str, candidate: str) -> bool:
    """Whether a GRU suggestion is close enough to the typed word to be considered."""
    typed, candidate = typed.lower(), candidate.lower()
    return (
        len(candidate) >= MIN_TARGET_LENGTH
        and editdistance.eval(typed, candidate) <= gru_distance_limit(typed)
        and relative_distance(typed, candidate) <= 0.5
    )


class EditDistanceDetector:
    """Find likely word corrections using a reference lexicon."""

    _WORD_PATTERN = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)

    def __init__(self, lexicon: list[str], variants: dict[str, str] | None = None) -> None:
        """Initialize with reference words (and accepted spelling variants), lowercased."""
        normalized_lexicon = [word.lower() for word in lexicon if word.strip()]
        if not normalized_lexicon:
            raise ValueError("lexicon must contain at least one non-empty word")
        self.lexicon = list(dict.fromkeys(normalized_lexicon))
        self._lexicon_set = set(self.lexicon)
        self.variants = {key.lower(): value.lower() for key, value in (variants or {}).items()}
        self._targets = [word for word in self.lexicon if len(word) >= MIN_TARGET_LENGTH] or self.lexicon
        # Alternative Romanizations of a lexicon word (long vowels written doubled or single).
        keys: dict[str, list[str]] = {}
        for entry in self.lexicon:
            keys.setdefault(long_vowel_key(entry), []).append(entry)
        self._spelling_keys = {key: words[0] for key, words in keys.items() if len(words) == 1}
        self._phonetic = PhoneticDetector(self.lexicon)
        self._codes: dict[str, list[str]] = {}
        for entry in self._targets:
            self._codes.setdefault(self._phonetic.get_phonetic_encoding(entry), []).append(entry)

    # ------------------------------------------------------------------ knowledge

    def is_known(self, word: str) -> bool:
        """True for lexicon words and for accepted alternative spellings of them."""
        clean = word.lower().strip()
        return clean in self._lexicon_set or self.variant_of(clean) is not None

    def variant_of(self, word: str) -> str | None:
        """The lexicon word that ``word`` is an accepted alternative spelling of, if any.

        Either listed in data/variants.csv, or the same word with long vowels written the other
        way (see :func:`long_vowel_key`).
        """
        clean = word.lower().strip()
        if clean in self.variants:
            return self.variants[clean]
        if clean in self._lexicon_set or not clean.isalpha():
            return None
        target = self._spelling_keys.get(long_vowel_key(clean))
        return target if target and target != clean else None

    # ------------------------------------------------------------------ matching helpers

    def phonetic_distance(self, first: str, second: str) -> int:
        """Edit distance between the phonetic (G2P) codes of two words (eq. 4 in the paper)."""
        return editdistance.eval(
            self._phonetic.get_phonetic_encoding(first), self._phonetic.get_phonetic_encoding(second)
        )

    def _best_by_phonetics(self, word: str, candidates: list[str]) -> tuple[str, float]:
        """Rank ``candidates`` by sound (code distance), then by bigram overlap with ``word``."""
        ranked = PhoneticDetector(candidates).find_closest_candidates(word, len(candidates))
        return min(ranked, key=lambda item: (self.phonetic_distance(word, item[0]), -item[1]))

    def _near(self, clean_word: str, max_distance: int) -> list[tuple[float, int, int, str]]:
        """Plausible corrections as ``(noise cost, edits, sound distance, word)``, best first.

        A word qualifies when it is within ``max_distance`` edits, within the relative-distance
        limit, and the difference looks like a typing slip rather than a different word.
        """
        near = []
        for entry in self._targets:
            distance = editdistance.eval(clean_word, entry)
            if distance > max_distance or distance / max(len(clean_word), len(entry)) > MAX_RELATIVE_DISTANCE:
                continue
            cost = noise_cost(clean_word, entry)
            if cost <= NOISE_COST_LIMIT:
                near.append((cost, distance, self.phonetic_distance(clean_word, entry), entry))
        near.sort(key=lambda item: item[:3])
        return near

    def _rescue(self, clean_word: str) -> str | None:
        """Match a letter-stretched word to a lexicon word with the same sound (distance 0)."""
        if not _STRETCHED.search(clean_word):
            return None
        same_sound = self._codes.get(self._phonetic.get_phonetic_encoding(clean_word), [])
        if not same_sound:
            return None
        best, score = self._best_by_phonetics(clean_word, same_sound)
        if score >= RESCUE_MIN_SCORE and relative_distance(clean_word, best) <= RESCUE_MAX_RELATIVE_DISTANCE:
            return best
        return None

    # ------------------------------------------------------------------ public API

    def plausible_corrections(
        self, word: str, max_distance: int = MAX_DISTANCE
    ) -> list[tuple[float, int, int, str]]:
        """All plausible corrections as ``(noise cost, edits, sound distance, word)``, best first.

        Includes the phonetic rescue for letter-stretched words. Known words, variants,
        one-letter tokens and non-alphabetic tokens have none.
        """
        clean_word = word.lower().strip()
        if self.is_known(clean_word) or not clean_word.isalpha() or len(clean_word) < MIN_TYPED_LENGTH:
            return []
        found = self._near(clean_word, max_distance)
        rescued = self._rescue(clean_word)
        if rescued is not None and all(rescued != entry for *_, entry in found):
            found.append(
                (RESCUE_COST, editdistance.eval(clean_word, rescued), self.phonetic_distance(clean_word, rescued), rescued)
            )
        return found

    def detect_and_correct_word(
        self, word: str, max_distance: int = MAX_DISTANCE
    ) -> tuple[str, bool, int]:
        """Return the closest lexicon word, error status, and edit distance.

        Known words and accepted variants come back unchanged with ``is_error`` False. A
        word with no acceptable match is returned unchanged with ``is_error`` True (it may
        simply be a valid word the lexicon does not contain).
        """
        if max_distance < 0:
            raise ValueError("max_distance must be non-negative")

        clean_word = word.lower().strip()
        if self.is_known(clean_word) or not clean_word.isalpha():
            return clean_word, False, 0

        matches = self.plausible_corrections(clean_word, max_distance)
        if matches:
            _, best_distance, _, best = matches[0]
            return best, True, best_distance
        return clean_word, True, min(editdistance.eval(clean_word, entry) for entry in self._targets)

    def get_candidates(self, word: str, max_distance: int = MAX_DISTANCE, limit: int = 4) -> list[str]:
        """Lexicon words that could be the intended spelling of ``word``, best first."""
        clean_word = word.lower().strip()
        if self.is_known(clean_word):
            return [clean_word]
        return [entry for *_, entry in self.plausible_corrections(clean_word, max_distance)][:limit]

    def process_sentence(self, sentence: str) -> tuple[str, list[dict]]:
        """Correct nearby misspellings and log each detected error.

        Non-word separators and punctuation are retained. Each error log
        contains the original token, selected candidate, candidate list,
        reported correction, and edit distance.
        """
        error_logs: list[dict] = []

        def replace_word(match: re.Match[str]) -> str:
            original = match.group()
            corrected, is_error, distance = self.detect_and_correct_word(original)
            if is_error:
                candidates = self.get_candidates(original)
                error_logs.append(
                    {
                        "original": original,
                        "corrected": corrected,
                        "candidate": candidates[0] if candidates else None,
                        "candidates": candidates,
                        "edit_distance": distance,
                    }
                )
            return corrected

        corrected_sentence = self._WORD_PATTERN.sub(replace_word, sentence)
        return corrected_sentence, error_logs
