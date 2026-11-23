"""Levenshtein distance with a fallback for Python versions lacking ``editdistance`` wheels.

``editdistance.eval`` and ``rapidfuzz.distance.Levenshtein.distance`` both compute the
plain (unit-cost insert/delete/substitute) Levenshtein distance, so results are identical.
"""

try:
    from editdistance import eval as _distance
except ImportError:  # e.g. Python 3.15, where editdistance has no prebuilt wheel
    from rapidfuzz.distance.Levenshtein import distance as _distance


def eval(first: str, second: str) -> int:  # noqa: A001 - mirrors editdistance.eval
    """Return the Levenshtein distance between two strings."""
    return int(_distance(first, second))
