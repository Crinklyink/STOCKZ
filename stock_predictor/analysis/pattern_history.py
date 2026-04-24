"""Historical win-rate backtests for detected chart patterns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import pandas as pd

from stock_predictor.analysis.patterns import PatternResult, detect_patterns


@dataclass(slots=True)
class PatternWinRateResult:
    pattern_name: str
    win_rate: float
    sample_size: int
    qualified: bool

    @property
    def label(self) -> str:
        percentage = round(self.win_rate * 100, 1)
        return f"{self.pattern_name.replace('_', ' ').title()} - {percentage}% win rate (last {self.sample_size} occurrences)"


class PatternWinRateAnalyzer:
    """Backtest pattern occurrence win rates on similar stocks."""

    def __init__(self, lookback_occurrences: int = 50, future_days: int = 5, step: int = 10) -> None:
        self.lookback_occurrences = lookback_occurrences
        self.future_days = future_days
        self.step = step
        self._window_cache: Dict[tuple[str, int], tuple[str, Dict[str, PatternResult], float]] = {}
        self._analysis_cache: Dict[tuple[str, str, str, float], PatternWinRateResult] = {}

    def analyze(
        self,
        pattern_name: str,
        sector: str,
        sector_universe: Dict[str, List[str]],
        price_frames: Dict[str, pd.DataFrame],
        current_regime: str,
        pattern_threshold: float,
    ) -> PatternWinRateResult:
        cache_key = (pattern_name, sector, current_regime, float(pattern_threshold))
        cached = self._analysis_cache.get(cache_key)
        if cached is not None:
            return cached
        relevant = sector_universe.get(sector, [])
        if not relevant:
            relevant = list(price_frames.keys())
        outcomes = self._collect_outcomes(
            pattern_name,
            relevant,
            price_frames,
            current_regime=current_regime,
            pattern_threshold=pattern_threshold,
            limit=self.lookback_occurrences,
        )
        wins = sum(outcomes)
        sample_size = len(outcomes)
        win_rate = wins / sample_size if sample_size else 0.0
        result = PatternWinRateResult(
            pattern_name=pattern_name,
            win_rate=win_rate,
            sample_size=sample_size,
            qualified=(sample_size >= 10 and win_rate >= 0.60),
        )
        self._analysis_cache[cache_key] = result
        return result

    def _collect_outcomes(
        self,
        pattern_name: str,
        tickers: Iterable[str],
        price_frames: Dict[str, pd.DataFrame],
        *,
        current_regime: str,
        pattern_threshold: float,
        limit: int,
    ) -> List[bool]:
        outcomes: List[bool] = []
        for ticker in tickers:
            frame = price_frames.get(ticker)
            if frame is None or frame.empty or len(frame) < 90:
                continue
            start_idx = max(70, len(frame) - 260)
            for end_idx in range(len(frame) - self.future_days - 1, start_idx, -self.step):
                cache_key = (ticker, end_idx)
                cached = self._window_cache.get(cache_key)
                if cached is None:
                    window = frame.iloc[end_idx - 70 : end_idx]
                    regime = infer_regime(window)
                    patterns = detect_patterns(window)
                    future_return = frame["close"].iloc[end_idx + self.future_days] / frame["close"].iloc[end_idx] - 1
                    cached = (regime, patterns, future_return)
                    self._window_cache[cache_key] = cached
                else:
                    regime, patterns, future_return = cached
                if current_regime != regime:
                    continue
                result = patterns.get(pattern_name)
                if result is None or result.score < pattern_threshold:
                    continue
                outcomes.append(bool(future_return >= 0.04))
                if len(outcomes) >= limit:
                    return outcomes
        return outcomes


def infer_regime(frame: pd.DataFrame) -> str:
    """Infer a rough market regime from the local price trend."""

    if frame.empty or len(frame) < 40:
        return "neutral"
    close = frame["close"]
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma20
    if close.iloc[-1] > sma20 >= sma50:
        return "risk_on"
    if close.iloc[-1] < sma20 <= sma50:
        return "risk_off"
    return "neutral"
