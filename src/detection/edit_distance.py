"""Edit-distance-based detection and correction for transliteration errors."""

import re

from src.detection import levenshtein as editdistance
from src.detection.phonetic_sim import PhoneticDetector

# A word farther than ``max_distance`` from every lexicon word is still matched when
# it sounds like one (same grapheme-to-phoneme code) and is similar enough overall.
PHONETIC_RESCUE_THRESHOLD = 0.5


def gru_distance_limit(word: str) -> int:
    """How many edits away a lexicon word suggested by the GRU may be from what was typed.

    Phase 1 stops at 2 edits. The GRU exists to handle noisier words, so it may go
    further for longer words (half the word's length), but its answer must still be a
    lexicon word, which keeps it from inventing spellings.
    """
    return max(2, (len(word) + 1) // 2)


class EditDistanceDetector:
    """Find likely word corrections using a reference lexicon."""

    _WORD_PATTERN = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)

    def __init__(self, lexicon: list[str]) -> None:
        """Initialize with reference words, normalized to lowercase."""
        normalized_lexicon = [word.lower() for word in lexicon if word.strip()]
        if not normalized_lexicon:
            raise ValueError("lexicon must contain at least one non-empty word")
        self.lexicon = list(dict.fromkeys(normalized_lexicon))
        self._lexicon_set = set(self.lexicon)
        self._phonetic = PhoneticDetector(self.lexicon)
        self._codes: dict[str, list[str]] = {}
        for entry in self.lexicon:
            self._codes.setdefault(self._phonetic.get_phonetic_encoding(entry), []).append(entry)

    def _best_by_phonetics(self, word: str, candidates: list[str]) -> tuple[str, float]:
        """Rank ``candidates`` with the G2P detector (bigram overlap + phonetic code)."""
        best, score = PhoneticDetector(candidates).find_closest_candidates(word, 1)[0]
        return best, score

    def get_candidates(self, word: str, max_distance: int = 2, limit: int = 4) -> list[str]:
        """Lexicon words that could be the intended spelling of ``word``, best first.

        Combines character edit distance (<= ``max_distance``) with the phonetic
        (G2P) rescue used by :meth:`detect_and_correct_word`.
        """
        clean_word = word.lower().strip()
        if clean_word in self._lexicon_set:
            return [clean_word]
        scored = [
            (editdistance.eval(clean_word, entry), entry) for entry in self.lexicon
        ]
        near = [entry for distance, entry in scored if distance <= max_distance]
        ranked: list[str] = []
        if near:
            for candidate, _ in PhoneticDetector(near).find_closest_candidates(clean_word, limit):
                ranked.append(candidate)
            ranked.sort(key=lambda entry: editdistance.eval(clean_word, entry))
        rescued = self._rescue(clean_word)
        if rescued is not None and rescued not in ranked:
            ranked.append(rescued)
        return ranked[:limit]

    def _rescue(self, clean_word: str) -> str | None:
        same_sound = self._codes.get(self._phonetic.get_phonetic_encoding(clean_word), [])
        if not same_sound:
            return None
        best, score = self._best_by_phonetics(clean_word, same_sound)
        return best if score >= PHONETIC_RESCUE_THRESHOLD else None

    def detect_and_correct_word(
        self, word: str, max_distance: int = 2
    ) -> tuple[str, bool, int]:
        """Return the closest lexicon word, error status, and edit distance.

        If the closest word is farther than ``max_distance``, leave the input
        unchanged and still report its closest distance.
        """
        if max_distance < 0:
            raise ValueError("max_distance must be non-negative")

        clean_word = word.lower().strip()
        if clean_word in self._lexicon_set:
            return clean_word, False, 0

        distances = [
            (editdistance.eval(clean_word, valid_word), valid_word)
            for valid_word in self.lexicon
        ]
        min_dist = min(distance for distance, _ in distances)
        closest = [valid_word for distance, valid_word in distances if distance == min_dist]

        if min_dist <= max_distance:
            # Several words equally close: let the phonetic (G2P) ranking choose.
            best_match = closest[0] if len(closest) == 1 else self._best_by_phonetics(clean_word, closest)[0]
            return best_match, True, min_dist

        rescued = self._rescue(clean_word)
        if rescued is not None:
            return rescued, True, editdistance.eval(clean_word, rescued)
        return clean_word, True, min_dist

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
                candidates = [
                    candidate
                    for candidate in self.lexicon
                    if editdistance.eval(original.lower(), candidate) == distance
                ]
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
