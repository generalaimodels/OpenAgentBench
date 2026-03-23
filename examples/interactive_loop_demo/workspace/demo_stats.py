"""Small demo module exercised by the compatibility example tests."""

from __future__ import annotations


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)


def minimum(values: list[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    return min(values)
