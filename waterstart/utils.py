from _typeshed import SupportsLessThan
from collections.abc import Sequence


def is_sorted(seq: Sequence[SupportsLessThan]) -> bool:
    return all(a < b for a, b in zip(seq[:-1], seq[1:]))


def is_contiguous(seq: Sequence[int]) -> bool:
    return all(b - a == 1 for a, b in zip(seq[:-1], seq[1:]))
