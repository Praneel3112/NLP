"""Dictionary-based language labels for code-mixed Romanized text."""

import re


_FALLBACK_ENGLISH_WORDS = frozenset(
    """
    a about after again all also am an and any are as at back be because been
    before being between both but by can come could day do does down each even
    every few find first for from get give go good had has have he her here him
    his how i if in into is it its just know like little long look made make man
    many may me more most much must my new no not now of off on one only or other
    our out over people said same see she should so some take than that the their
    them then there these they thing think this those time to two up us use very
    want was way we well were what when which who will with would year you your
    hello hi love please thanks thank yes no okay ok
    """.split()
)


class LanguageIdentifier:
    """Label tokens as English or Romanized Indic using a word dictionary."""

    _TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

    def __init__(self) -> None:
        """Load NLTK's English word corpus or use a small offline fallback list."""
        try:
            from nltk.corpus import words

            self._english_words = {word.lower() for word in words.words()}
        except ImportError:
            self._english_words = set(_FALLBACK_ENGLISH_WORDS)
        except LookupError:
            self._english_words = set(_FALLBACK_ENGLISH_WORDS)

    def is_english_word(self, word: str) -> bool:
        """Return whether ``word`` appears in the available English dictionary."""
        return word.casefold() in self._english_words

    def segregate_code_mixed_tokens(self, sentence: str) -> list[dict[str, str]]:
        """Return word tokens with ENGLISH or ROMANIZED_INDIC labels.

        Tokens absent from the English dictionary are conservatively labeled
        ROMANIZED_INDIC; this also serves as the fallback for unknown words.
        """
        return [
            {
                "token": token,
                "label": (
                    "ENGLISH"
                    if self.is_english_word(token)
                    else "ROMANIZED_INDIC"
                ),
            }
            for token in self._TOKEN_PATTERN.findall(sentence)
        ]