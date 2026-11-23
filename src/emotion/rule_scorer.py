"""Explainable rule-based emotion scoring from emoji, slang and emotion words."""

import re
from typing import Any

import emoji

from src.emotion.lexicons import (
    EMOJI_EMOTIONS,
    EMOTIONS,
    INTENSIFIERS,
    NEGATORS,
    SLANG_EMOTIONS,
    WORD_EMOTIONS,
)

# What a cue turns into when it is negated: "not happy" reads as mildly sad,
# "not angry" / "not scared" read as calm (neutral).
NEGATION_FLIP = {
    "joy": "sadness",
    "love": "sadness",
    "sadness": "joy",
    "anger": "neutral",
    "fear": "neutral",
    "surprise": "neutral",
    "neutral": "neutral",
}
NEGATION_FACTOR = 0.6
ELONGATION_BOOST = 1.3
CAPS_BOOST = 1.3
_REPEAT = re.compile(r"(.)\1{2,}")
_CORE = re.compile(r"[^\w']+", re.UNICODE)
_VARIATION = re.compile("[︎️]")


def _variants(word: str) -> list[str]:
    """The word itself plus letter-stretching removed: "khushiii" -> "khushii", "khushi"."""
    out = [word]
    for squashed in (_REPEAT.sub(r"\1\1", word), _REPEAT.sub(r"\1", word)):
        if squashed not in out:
            out.append(squashed)
    return out


def tokenize(text: str) -> list[dict[str, str]]:
    """Split text into word and emoji tokens, keeping each emoji as its own token."""
    spaced = emoji.replace_emoji(text, replace=lambda chars, _data: f" {chars} ")
    tokens: list[dict[str, str]] = []
    for raw in spaced.split():
        if emoji.is_emoji(raw) or emoji.purely_emoji(raw):
            tokens.append({"kind": "emoji", "raw": raw, "key": raw})
            continue
        core = _CORE.sub("", raw)
        if core:
            tokens.append({"kind": "word", "raw": core, "key": core.casefold()})
    return tokens


def _lookup_emoji(raw: str) -> tuple[str, float] | None:
    return EMOJI_EMOTIONS.get(raw) or EMOJI_EMOTIONS.get(_VARIATION.sub("", raw)) or (
        EMOJI_EMOTIONS.get(_VARIATION.sub("", raw) + "️")
    )


def _lookup_word(key: str) -> tuple[str, str, float, bool] | None:
    """Return (source, emotion, weight, was_elongated) for a word token."""
    for index, variant in enumerate(_variants(key)):
        for source, table in (("slang", SLANG_EMOTIONS), ("word", WORD_EMOTIONS)):
            if variant in table:
                emotion, weight = table[variant]
                return source, emotion, weight, index > 0
    return None


class RuleEmotionScorer:
    """Score each emotion from dictionary cues, negation and intensity markers."""

    def score(self, text: str) -> dict[str, Any]:
        tokens = tokenize(text)
        scores = {emotion: 0.0 for emotion in EMOTIONS}
        evidence: list[dict[str, Any]] = []
        signals: list[str] = []

        for index, token in enumerate(tokens):
            if token["kind"] == "emoji":
                match = _lookup_emoji(token["raw"])
                if match is None:
                    continue
                emotion, weight = match
                scores[emotion] += weight
                evidence.append(
                    {
                        "token": token["raw"],
                        "type": "emoji",
                        "emotion": emotion,
                        "weight": round(weight, 2),
                        "negated": False,
                        "boosts": [],
                    }
                )
                continue

            found = _lookup_word(token["key"])
            if found is None:
                continue
            source, emotion, weight, elongated = found
            boosts: list[str] = []

            nearby_before = [t["key"] for t in tokens[max(0, index - 2):index] if t["kind"] == "word"]
            nearby_after = [t["key"] for t in tokens[index + 1:index + 3] if t["kind"] == "word"]

            intensifier = next(
                (
                    (word, variant)
                    for word in nearby_before
                    for variant in _variants(word)
                    if variant in INTENSIFIERS
                ),
                None,
            )
            if intensifier is not None:
                weight *= INTENSIFIERS[intensifier[1]]
                boosts.append(f"intensifier '{intensifier[0]}'")
            if elongated:
                weight *= ELONGATION_BOOST
                boosts.append("stretched letters")
            if len(token["raw"]) > 1 and token["raw"].isupper():
                weight *= CAPS_BOOST
                boosts.append("CAPS")

            # English negates before ("not happy"); Hindi/Kannada/Tamil often after ("khush nahi").
            negated = any(word in NEGATORS for word in nearby_before + nearby_after[:1])
            final_emotion = emotion
            if negated:
                final_emotion = NEGATION_FLIP[emotion]
                weight *= NEGATION_FACTOR

            scores[final_emotion] += weight
            evidence.append(
                {
                    "token": token["raw"],
                    "type": source,
                    "emotion": final_emotion,
                    "original_emotion": emotion,
                    "weight": round(weight, 2),
                    "negated": negated,
                    "boosts": boosts,
                }
            )

        exclamations = text.count("!")
        if exclamations >= 2:
            signals.append(f"{exclamations} exclamation marks")
        letters = [c for c in text if c.isalpha()]
        if len(letters) >= 4 and sum(c.isupper() for c in letters) / len(letters) > 0.7:
            signals.append("mostly CAPS")
        if any(e["boosts"] for e in evidence):
            signals.extend(sorted({b for e in evidence for b in e["boosts"]}))

        emotional_total = sum(v for k, v in scores.items() if k != "neutral")
        strength = emotional_total * (1 + 0.1 * min(exclamations, 5))
        if "mostly CAPS" in signals:
            strength *= 1.2
        intensity = "none" if strength == 0 else "low" if strength < 2 else "medium" if strength < 4 else "high"

        total = sum(scores.values())
        probabilities = (
            {emotion: value / total for emotion, value in scores.items()}
            if total > 0
            else {emotion: (1.0 if emotion == "neutral" else 0.0) for emotion in EMOTIONS}
        )
        return {
            "has_evidence": bool(evidence),
            "raw_scores": {k: round(v, 3) for k, v in scores.items()},
            "probabilities": probabilities,
            "evidence": evidence,
            "intensity": intensity,
            "intensity_score": round(strength, 2),
            "signals": signals,
        }
