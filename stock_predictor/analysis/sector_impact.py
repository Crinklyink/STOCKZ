"""Sector-specific macro, weather, and commodity tailwind scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

try:  # pragma: no cover - optional dependency
    from fredapi import Fred
except Exception:  # pragma: no cover
    Fred = None

from stock_predictor.config import AppConfig
from stock_predictor.data.fetcher import MarketDataFetcher
from stock_predictor.utils import clamp, coerce_float


@dataclass(slots=True)
class SectorImpactResult:
    points: float
    normalized_score: float
    summary: str


class SectorImpactEngine:
    """Calculate sector-specific macro tailwind or headwind points."""

    def __init__(self, config: AppConfig, fetcher: MarketDataFetcher) -> None:
        self.config = config
        self.fetcher = fetcher
        self.fred = Fred(api_key=config.fred_api_key) if (Fred and config.fred_api_key) else None
        self.shared_benchmark_return: float | None = None

    def score_sector(self, sector: str, fresh: bool = False) -> SectorImpactResult:
        points = 0.0
        reasons = []
        if sector == "Energy":
            oil = trailing_return(self.fetcher.fetch_macro_history("CL=F", fresh=fresh), 5)
            gas = trailing_return(self.fetcher.fetch_macro_history("NG=F", fresh=fresh), 5)
            points = clamp((oil + gas) * 100, -10.0, 10.0)
            reasons.append("Energy tailwind from crude and nat gas futures")
        elif sector in {"Consumer Discretionary", "Consumer Staples"}:
            confidence = self._consumer_confidence_signal()
            points = clamp(confidence * 10.0, -10.0, 10.0)
            reasons.append("Consumer confidence trend")
        elif sector == "Materials":
            copper = trailing_return(self.fetcher.fetch_macro_history("HG=F", fresh=fresh), 5)
            points = clamp(copper * 100, -10.0, 10.0)
            reasons.append("Commodity input trend")
        else:
            benchmark = (
                self.shared_benchmark_return
                if self.shared_benchmark_return is not None
                else trailing_return(self.fetcher.fetch_macro_history(self.config.benchmark_ticker, fresh=fresh), 5)
            )
            points = clamp(benchmark * 50.0, -5.0, 5.0)
            reasons.append("Broad market sector tailwind")
        normalized = clamp((points + 10.0) / 20.0, 0.0, 1.0)
        return SectorImpactResult(points=points, normalized_score=normalized, summary=", ".join(reasons))

    def _consumer_confidence_signal(self) -> float:
        if self.fred is None:
            return 0.0
        try:
            series = self.fred.get_series_latest_release("UMCSENT")
        except Exception:
            return 0.0
        if len(series) < 2:
            return 0.0
        latest = coerce_float(series.iloc[-1])
        prior = coerce_float(series.iloc[-2], default=latest)
        return clamp((latest - prior) / max(abs(prior), 1.0), -1.0, 1.0)


def trailing_return(frame, periods: int = 5) -> float:
    if frame is None or frame.empty or len(frame) <= periods:
        return 0.0
    close = frame["close"]
    return float(close.iloc[-1] / close.iloc[-periods - 1] - 1)
