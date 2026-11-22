"""Shared lexicon and training data for the transliteration correction pipeline."""

import csv
from pathlib import Path

from src.preprocessing.noise_generator import NoiseGenerator

LEXICON_PATH = Path(__file__).resolve().parents[2] / "data" / "lexicon.csv"


def _load_lexicon(path: Path = LEXICON_PATH) -> tuple[tuple[str, ...], dict[str, dict[str, str]]]:
    """Read ``word,language,meaning`` rows, keeping file order and dropping duplicates."""
    words: list[str] = []
    info: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            word = row["word"].strip().casefold()
            if word and word not in info:
                words.append(word)
                info[word] = {"language": row["language"].strip(), "meaning": row["meaning"].strip()}
    return tuple(words), info


# The first 13 words are the original CIE3 sample lexicon; the rest are common
# Hindi, Kannada, Tamil and Bengali words added for the sem-end version.
LEXICON, LEXICON_INFO = _load_lexicon()


def build_training_pairs() -> tuple[list[str], list[str]]:
    """Create the deterministic noisy-to-clean pairs used to train Phases 2 and 3."""
    sources: list[str] = []
    targets: list[str] = []

    for word_index, word in enumerate(LEXICON):
        for variant in range(4):
            generator = NoiseGenerator(seed=1000 + word_index * 4 + variant)
            sources.append(generator.generate_noisy_text(word))
            targets.append(word)

    sources.append("ap")
    targets.append("aap")
    return sources, targets
