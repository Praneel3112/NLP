"""Edit-distance-based detection and correction for transliteration errors."""

import re

from src.detection import levenshtein as editdistance


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

        best_match = clean_word
        min_dist = len(clean_word) + max(
            (len(valid_word) for valid_word in self.lexicon), default=0
        )
        for valid_word in self.lexicon:
            distance = editdistance.eval(clean_word, valid_word)
            if distance < min_dist:
                min_dist = distance
                best_match = valid_word

        if min_dist <= max_distance:
            return best_match, True, min_dist
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
