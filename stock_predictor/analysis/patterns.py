"""Pattern recognition for swing-trade setups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from stock_predictor.analysis.indicators import add_indicators
from stock_predictor.utils import clamp


@dataclass(slots=True)
class PatternResult:
    name: str
    score: float
    notes: str


def detect_patterns(frame: pd.DataFrame) -> Dict[str, PatternResult]:
    """Return a suite of pattern detections."""

    enriched = add_indicators(frame)
    if enriched.empty or len(enriched) < 60:
        return {
            "best_pattern": PatternResult("none", 0.0, "Not enough data"),
        }
    close = enriched["close"]
    high = enriched["high"]
    low = enriched["low"]
    volume = enriched["volume"]
    patterns = {
        "bull_flag": bull_flag(close, volume),
        "cup_handle": cup_and_handle(close, volume),
        "ascending_triangle": ascending_triangle(close, high, low, volume),
        "golden_cross": golden_cross(enriched),
        "vwap_reclaim": vwap_reclaim(enriched),
        "inside_bar_breakout": inside_bar_breakout(enriched),
    }
    best = max(patterns.values(), key=lambda item: item.score)
    patterns["best_pattern"] = best
    return patterns


def bull_flag(close: pd.Series, volume: pd.Series) -> PatternResult:
    impulse = close.iloc[-20:-10].pct_change().add(1).prod() - 1
    pullback = (close.iloc[-10:] / close.iloc[-10] - 1).min()
    volume_confirm = volume.iloc[-5:].mean() / max(volume.iloc[-20:-5].mean(), 1)
    score = clamp((impulse * 20) + (1 + pullback) * 4 + volume_confirm * 2, 0.0, 10.0)
    return PatternResult("bull_flag", score, "Strong impulse followed by shallow pullback")


def cup_and_handle(close: pd.Series, volume: pd.Series) -> PatternResult:
    window = close.iloc[-60:]
    left = window.iloc[:20].max()
    trough = window.iloc[20:40].min()
    right = window.iloc[40:55].max()
    handle = window.iloc[55:].min()
    depth = (left - trough) / max(left, 1)
    symmetry = 1 - abs(left - right) / max(left, 1)
    handle_quality = 1 - abs(right - handle) / max(right, 1)
    volume_factor = volume.iloc[-10:].mean() / max(volume.iloc[-30:-10].mean(), 1)
    score = clamp((symmetry * 4 + handle_quality * 3 + depth * 6 + volume_factor), 0.0, 10.0)
    return PatternResult("cup_handle", score, "Rounded base with controlled handle")


def ascending_triangle(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
) -> PatternResult:
    recent_highs = high.iloc[-30:]
    recent_lows = low.iloc[-30:]
    flat_top = 1 - (recent_highs.max() - recent_highs.quantile(0.8)) / max(recent_highs.max(), 1)
    rising_lows = np.polyfit(range(len(recent_lows)), recent_lows.to_numpy(), 1)[0]
    breakout = close.iloc[-1] / max(recent_highs.max(), 1)
    volume_factor = volume.iloc[-5:].mean() / max(volume.iloc[-30:-5].mean(), 1)
    score = clamp(flat_top * 4 + rising_lows * 10 + (breakout - 0.98) * 20 + volume_factor, 0.0, 10.0)
    return PatternResult("ascending_triangle", score, "Higher lows into stable resistance")


def golden_cross(frame: pd.DataFrame) -> PatternResult:
    latest = frame.iloc[-1]
    recent = frame.iloc[-10:]
    cross = recent["golden_cross"].max()
    slope = (latest["sma_50"] - frame["sma_50"].iloc[-10]) / max(frame["sma_50"].iloc[-10], 1)
    score = clamp(cross * 7 + slope * 50, 0.0, 10.0)
    return PatternResult("golden_cross", score, "50-day moving average crossed above the 200-day")


def vwap_reclaim(frame: pd.DataFrame) -> PatternResult:
    recent = frame.iloc[-10:]
    below_then_above = (recent["close"].shift(3) < recent["vwap"].shift(3)).fillna(False) & (
        recent["close"] > recent["vwap"]
    )
    reclaim = below_then_above.any()
    distance = recent["vwap_distance"].iloc[-1]
    score = clamp((4 if reclaim else 0) + (distance + 0.03) * 100, 0.0, 10.0)
    return PatternResult("vwap_reclaim", score, "Price reclaimed VWAP with positive distance")


def inside_bar_breakout(frame: pd.DataFrame) -> PatternResult:
    recent = frame.iloc[-5:]
    triggered = bool(recent["inside_bar"].iloc[-2] and recent["close"].iloc[-1] > recent["high"].iloc[-2])
    volume_factor = recent["volume"].iloc[-1] / max(recent["volume"].iloc[:-1].mean(), 1)
    score = clamp((6 if triggered else 0) + volume_factor * 2, 0.0, 10.0)
    return PatternResult("inside_bar_breakout", score, "Inside bar resolved upward on volume")
