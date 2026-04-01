"""Multi-timeframe trend and momentum confirmation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from stock_predictor.analysis.indicators import add_indicators
from stock_predictor.utils import clamp


@dataclass(slots=True)
class MultiTimeframeResult:
    qualifies: bool
    contradicts: bool
    agreement_count: int
    penalty_factor: float
    directions: Dict[str, str]
    scores: Dict[str, float]
    summary: str


def resample_ohlcv(frame: pd.DataFrame, rule: str = "4h") -> pd.DataFrame:
    """Resample hourly OHLCV into a higher timeframe."""

    if frame.empty:
        return frame
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    resampled = frame.resample(rule).agg(agg).dropna()
    return resampled


def timeframe_signal(frame: pd.DataFrame, mode: str) -> float:
    """Return a normalized bullishness score for a timeframe."""

    data = add_indicators(frame).dropna()
    if data.empty:
        return 0.5
    latest = data.iloc[-1]
    if mode == "daily":
        score = (
            0.35 * float(latest["close"] > latest["sma_20"])
            + 0.25 * float(latest["close"] > latest["sma_50"])
            + 0.20 * clamp((latest["rsi"] - 45.0) / 20.0, 0.0, 1.0)
            + 0.20 * float(latest["macd_hist"] > 0)
        )
    elif mode == "4h":
        score = (
            0.35 * float(latest["close"] > latest["sma_20"])
            + 0.25 * float(latest["vwap_distance"] > 0)
            + 0.20 * float(latest["macd_hist"] > 0)
            + 0.20 * clamp((latest["rsi"] - 48.0) / 18.0, 0.0, 1.0)
        )
    else:
        score = (
            0.40 * float(latest["close"] > latest["vwap"])
            + 0.25 * float(latest["return_5"] > 0)
            + 0.20 * float(latest["macd_hist"] > 0)
            + 0.15 * clamp((latest["mfi"] - 45.0) / 25.0, 0.0, 1.0)
        )
    return clamp(score, 0.0, 1.0)


def classify_direction(score: float) -> str:
    """Convert a normalized score into a direction label."""

    if score >= 0.55:
        return "bullish"
    if score <= 0.45:
        return "bearish"
    return "neutral"


def evaluate_multi_timeframe_alignment(
    daily_frame: pd.DataFrame,
    hourly_frame: pd.DataFrame,
    partial_penalty: float = 0.80,
) -> MultiTimeframeResult:
    """Evaluate whether daily, 4-hour, and 1-hour timeframes align."""

    four_hour = resample_ohlcv(hourly_frame, "4h")
    scores = {
        "daily": timeframe_signal(daily_frame, "daily"),
        "4h": timeframe_signal(four_hour, "4h"),
        "1h": timeframe_signal(hourly_frame, "1h"),
    }
    directions = {name: classify_direction(score) for name, score in scores.items()}
    contradiction = "bullish" in directions.values() and "bearish" in directions.values()
    agreement_count = sum(direction == "bullish" for direction in directions.values())
    qualifies = not contradiction and agreement_count >= 2
    penalty_factor = 1.0 if agreement_count == 3 else partial_penalty if qualifies else 0.0
    summary = "All timeframes aligned bullishly."
    if contradiction:
        summary = "Timeframes contradict each other."
    elif qualifies and agreement_count == 2:
        summary = "Two of three timeframes align; confidence reduced."
    elif not qualifies:
        summary = "Not enough bullish confirmation across timeframes."
    return MultiTimeframeResult(
        qualifies=qualifies,
        contradicts=contradiction,
        agreement_count=agreement_count,
        penalty_factor=penalty_factor,
        directions=directions,
        scores=scores,
        summary=summary,
    )

