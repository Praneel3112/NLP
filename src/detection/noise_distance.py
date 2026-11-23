"""Noise-aware edit distance: how plausible is it that ``typed`` is a mistyping of ``target``?

Plain edit distance treats every edit alike, so a real word the lexicon has never seen can
look like a typo of a different known word (``thali`` vs ``khali`` is one edit). Real typing
mistakes are not random, though: vowels get dropped, letters get repeated, fingers hit a
neighbouring key, two letters get swapped. This distance charges little for those and a lot
for anything else, which lets the detector tell a typo from a different word.
"""

from src.preprocessing.noise_generator import NoiseGenerator

_VOWELS = frozenset("aeiou")

# QWERTY neighbours (made symmetric): the keys a finger can slip onto.
_NEIGHBOURS: dict[str, set[str]] = {}
for _key, _others in NoiseGenerator._KEY_NEIGHBOURS.items():
    for _other in _others:
        _NEIGHBOURS.setdefault(_key, set()).add(_other)
        _NEIGHBOURS.setdefault(_other, set()).add(_key)

COST_REPEATED_LETTER = 0.3  # "naaam": the typed word has an extra copy of a letter
COST_DROPPED_VOWEL = 0.4  # "kse": the typed word lacks a vowel
COST_SWAPPED_PAIR = 0.8  # "namsate": two neighbouring letters exchanged
COST_NEIGHBOUR_KEY = 0.7  # a letter replaced by an adjacent key
COST_NEIGHBOUR_INSERT = 0.8  # an extra letter typed next to an adjacent key
COST_DROPPED_CONSONANT = 1.0
COST_OTHER = 1.5  # anything a typing slip would not explain


def _adjacent(first: str, second: str) -> bool:
    return second in _NEIGHBOURS.get(first, ())


def _extra_letter_cost(typed: str, position: int) -> float:
    """Cost of the typed word having a letter its target does not have."""
    char = typed[position]
    before = typed[position - 1] if position > 0 else ""
    after = typed[position + 1] if position + 1 < len(typed) else ""
    if char in (before, after):
        return COST_REPEATED_LETTER
    if (before and _adjacent(char, before)) or (after and _adjacent(char, after)):
        return COST_NEIGHBOUR_INSERT
    return COST_OTHER


def _missing_letter_cost(char: str) -> float:
    """Cost of the typed word lacking a letter its target has."""
    return COST_DROPPED_VOWEL if char in _VOWELS else COST_DROPPED_CONSONANT


def _substitution_cost(typed_char: str, target_char: str) -> float:
    if typed_char == target_char:
        return 0.0
    # A vowel for another vowel is only a typing slip when the keys are neighbours (u/i, i/o).
    # Otherwise it is a different ending or a different word ("dekhne"/"dekhna", "nadi"/"nodi").
    if _adjacent(typed_char, target_char):
        return COST_NEIGHBOUR_KEY
    return COST_OTHER


def noise_cost(typed: str, target: str) -> float:
    """Total plausibility cost of reading ``typed`` as a mistyped ``target`` (lower = likelier)."""
    typed, target = typed.lower(), target.lower()
    rows, cols = len(typed), len(target)
    table = [[0.0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        table[i][0] = table[i - 1][0] + _extra_letter_cost(typed, i - 1)
    for j in range(1, cols + 1):
        table[0][j] = table[0][j - 1] + _missing_letter_cost(target[j - 1])

    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            best = min(
                table[i - 1][j] + _extra_letter_cost(typed, i - 1),
                table[i][j - 1] + _missing_letter_cost(target[j - 1]),
                table[i - 1][j - 1] + _substitution_cost(typed[i - 1], target[j - 1]),
            )
            if i > 1 and j > 1 and typed[i - 1] == target[j - 2] and typed[i - 2] == target[j - 1]:
                best = min(best, table[i - 2][j - 2] + COST_SWAPPED_PAIR)
            table[i][j] = best
    return round(table[rows][cols], 3)
