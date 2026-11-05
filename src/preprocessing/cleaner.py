"""Utilities for normalizing and tokenizing Romanized Indic text."""

import re
import unicodedata


class TextCleaner:
    """Normalize raw Romanized Indic text for downstream processing."""

    def clean_text(self, text: str) -> str:
        """Remove punctuation and other noise, then collapse whitespace.

        Characters that are not alphanumeric or whitespace are replaced with
        spaces so punctuation between words does not join those words.
        """
        normalized = self.normalize_unicode(text)
        cleaned = "".join(
            char if char.isalnum() or char.isspace() else " "
            for char in normalized
        )
        return re.sub(r"\s+", " ", cleaned).strip()

    def normalize_unicode(self, text: str) -> str:
        """Decompose accented Latin characters and remove their accent marks."""
        decomposed = unicodedata.normalize("NFKD", text)
        return "".join(
            char for char in decomposed if unicodedata.category(char) != "Mn"
        )

    def tokenize(self, text: str) -> list[str]:
        """Clean text and return its non-empty whitespace-separated tokens."""
        cleaned = self.clean_text(text)
        return cleaned.split() if cleaned else []