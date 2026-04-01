"""Ticker-level data quality validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pandas as pd


@dataclass(slots=True)
class DataQualityResult:
    score: float
    is_valid: bool
    issues: List[str]


def validate_ticker_data(
    *,
    info: Dict[str, Any],
    daily: pd.DataFrame,
    hourly: pd.DataFrame,
    minimum_score: float,
) -> DataQualityResult:
    """Validate required data fields and freshness before scoring."""

    score = 1.0
    issues: List[str] = []
    now = datetime.now(timezone.utc)
    if daily.empty:
        score -= 0.6
        issues.append("yfinance: missing daily price history")
    else:
        latest_daily = pd.to_datetime(daily.index[-1], utc=True)
        if latest_daily < now - timedelta(days=10):
            score -= 0.2
            issues.append("yfinance: stale daily history")
    if hourly.empty:
        score -= 0.4
        issues.append("yfinance: missing hourly price history")
    else:
        latest_hourly = pd.to_datetime(hourly.index[-1], utc=True)
        if latest_hourly < now - timedelta(days=5):
            score -= 0.15
            issues.append("yfinance: stale hourly history")
    if not info:
        score -= 0.25
        issues.append("yfinance: missing company info")
    if not info.get("currentPrice") and not info.get("lastPrice"):
        score -= 0.15
        issues.append("yfinance: missing price field")
    if not (
        info.get("averageVolume")
        or info.get("averageDailyVolume3Month")
        or info.get("tenDayAverageVolume")
    ):
        score -= 0.10
        issues.append("yfinance: missing average volume field")
    score = max(score, 0.0)
    return DataQualityResult(score=score, is_valid=score >= minimum_score, issues=issues)

