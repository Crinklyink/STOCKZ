"""Adaptive composite-weight evolution based on recent prediction accuracy."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List

from stock_predictor.utils import clamp


@dataclass(slots=True)
class AdaptiveWeightResult:
    weights: Dict[str, float]
    rolling_accuracy: Dict[str, float]
    wrong_streaks: Dict[str, int]


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalize weights to sum to 1."""

    total = sum(max(value, 0.0) for value in weights.values()) or 1.0
    return {key: max(value, 0.0) / total for key, value in weights.items()}


def compute_signal_accuracies(rows: Iterable[dict]) -> tuple[Dict[str, float], Dict[str, int]]:
    """Compute 4-week rolling accuracy and recent wrong-streaks per signal."""

    weekly_accuracy: Dict[str, List[float]] = defaultdict(list)
    ordered_weeks: Dict[str, List[tuple[str, float]]] = defaultdict(list)
    grouped: Dict[tuple[str, str], List[int]] = defaultdict(list)

    for row in rows:
        signal_name = row.get("signal_name")
        week = row.get("week")
        signal_score = float(row.get("signal_score", 0.0))
        hit_target = int(row.get("hit_target", 0))
        if not signal_name or not week:
            continue
        if signal_score < 55.0:
            continue
        grouped[(signal_name, week)].append(hit_target)

    for (signal_name, week), hits in grouped.items():
        accuracy = sum(hits) / max(len(hits), 1)
        weekly_accuracy[signal_name].append(accuracy)
        ordered_weeks[signal_name].append((week, accuracy))

    rolling = {
        signal_name: sum(values[-4:]) / max(len(values[-4:]), 1)
        for signal_name, values in weekly_accuracy.items()
    }
    wrong_streaks: Dict[str, int] = {}
    for signal_name, values in ordered_weeks.items():
        streak = 0
        for _, accuracy in sorted(values, key=lambda item: item[0], reverse=True):
            if accuracy < 0.5:
                streak += 1
            else:
                break
        wrong_streaks[signal_name] = streak
    return rolling, wrong_streaks


def evolve_weights(
    base_weights: Dict[str, float],
    recent_rows: Iterable[dict],
    current_weights: Dict[str, float] | None = None,
) -> AdaptiveWeightResult:
    """Adjust weights up or down based on rolling signal accuracy."""

    normalized_base = normalize_weights(base_weights)
    normalized_current = normalize_weights(current_weights or normalized_base)
    rolling_accuracy, wrong_streaks = compute_signal_accuracies(recent_rows)
    evolved: Dict[str, float] = {}
    for signal_name, base_weight in normalized_base.items():
        accuracy = rolling_accuracy.get(signal_name, 0.5)
        wrong_streak = wrong_streaks.get(signal_name, 0)
        multiplier = 0.75 + (0.5 * accuracy)
        if accuracy >= 0.8:
            multiplier += 0.05
        if wrong_streak >= 3:
            multiplier *= 0.80
        evolved[signal_name] = normalized_current.get(signal_name, base_weight) * multiplier
        evolved[signal_name] = clamp(evolved[signal_name], 0.02, 0.5)
    evolved = normalize_weights(evolved)
    return AdaptiveWeightResult(
        weights=evolved,
        rolling_accuracy=rolling_accuracy,
        wrong_streaks=wrong_streaks,
    )

