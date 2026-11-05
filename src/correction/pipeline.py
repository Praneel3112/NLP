"""Shared sample data for the transliteration correction pipeline."""

from src.preprocessing.noise_generator import NoiseGenerator


LEXICON = (
    "namaste",
    "aap",
    "kaise",
    "ho",
    "nanna",
    "hesaru",
    "ravi",
    "en",
    "peyar",
    "meena",
    "amar",
    "naam",
    "suman",
)


def build_training_pairs() -> tuple[list[str], list[str]]:
    """Create the deterministic noisy-to-clean pairs used by the demo pipeline."""
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
