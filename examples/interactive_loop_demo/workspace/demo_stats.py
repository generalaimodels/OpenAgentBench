"""Small demo module used by the interactive loop terminal tool."""

from __future__ import annotations


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("window must be positive")
    if not values:
        return []
    if window > len(values):
        return [sum(values) / len(values)]
    averages: list[float] = []
    running_total = sum(values[:window])
    averages.append(running_total / window)
    for index in range(window, len(values)):
        running_total += values[index]
        running_total -= values[index - window]
        averages.append(running_total / window)
    return averages


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    maximum = max(scores)
    if maximum == 0:
        return [0.0 for _ in scores]
    return [score / maximum for score in scores]
