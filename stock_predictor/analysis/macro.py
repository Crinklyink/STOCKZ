"""Macro regime and sector rotation analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from stock_predictor.config import AppConfig
from stock_predictor.data.fetcher import MarketDataFetcher
from stock_predictor.utils import clamp, coerce_float


@dataclass(slots=True)
class MacroSnapshot:
    risk_regime: str
    vix: float
    dxy_5d_return: float
    yield_spread_proxy: float
    breadth_percentile: float
    top_sectors: List[str]
    bottom_sectors: List[str]
    sector_returns: Dict[str, float]
    fed_event_window: bool
    score_multiplier: float


class MacroRegimeAnalyzer:
    """Evaluate regime, volatility, and sector leadership."""

    def __init__(self, config: AppConfig, fetcher: MarketDataFetcher) -> None:
        self.config = config
        self.fetcher = fetcher

    def build_snapshot(
        self,
        fresh: bool = False,
        *,
        price_frames: Dict[str, pd.DataFrame] | None = None,
    ) -> MacroSnapshot:
        vix_history = self.fetcher.fetch_macro_history(self.config.vix_ticker, fresh=fresh)
        dxy_history = self.fetcher.fetch_macro_history(self.config.dxy_ticker, fresh=fresh)
        ten_year = self.fetcher.fetch_macro_history(self.config.ten_year_ticker, fresh=fresh)
        two_year = self.fetcher.fetch_macro_history(self.config.two_year_ticker, fresh=fresh)
        sector_history = self.fetcher.fetch_sector_history(fresh=fresh)
        sector_returns = {
            sector: trailing_return(frame, periods=5) for sector, frame in sector_history.items() if not frame.empty
        }
        top_sectors = [
            item[0]
            for item in sorted(sector_returns.items(), key=lambda pair: pair[1], reverse=True)[
                : self.config.top_sector_count
            ]
        ]
        bottom_sectors = [
            item[0]
            for item in sorted(sector_returns.items(), key=lambda pair: pair[1])[
                : self.config.top_sector_count
            ]
        ]
        vix = coerce_float(vix_history["close"].iloc[-1] if not vix_history.empty else 0.0)
        dxy_5d = trailing_return(dxy_history, periods=5)
        ten_close = coerce_float(ten_year["close"].iloc[-1] if not ten_year.empty else 0.0)
        two_close = coerce_float(two_year["close"].iloc[-1] if not two_year.empty else 0.0)
        breadth_percentile = clamp(
            self.fetcher.calculate_market_breadth(price_frames or {}),
            0.0,
            1.0,
        )
        spread = ten_close - two_close
        risk_regime = "risk_on"
        multiplier = 1.0
        if vix > self.config.thresholds.vix_risk_off or spread < 0:
            risk_regime = "risk_off"
            multiplier = 0.85
        elif vix > self.config.thresholds.vix_risk_off - 4:
            risk_regime = "neutral"
            multiplier = 0.93
        fed_event_window = False
        return MacroSnapshot(
            risk_regime=risk_regime,
            vix=vix,
            dxy_5d_return=dxy_5d,
            yield_spread_proxy=spread,
            breadth_percentile=breadth_percentile,
            top_sectors=top_sectors,
            bottom_sectors=bottom_sectors,
            sector_returns=sector_returns,
            fed_event_window=fed_event_window,
            score_multiplier=multiplier,
        )

    def sector_score(self, sector: str, snapshot: MacroSnapshot) -> float:
        if not snapshot.sector_returns:
            return 0.5
        values = list(snapshot.sector_returns.values())
        minimum = min(values)
        maximum = max(values)
        current = snapshot.sector_returns.get(sector, minimum)
        if maximum == minimum:
            return 0.5
        raw = (current - minimum) / (maximum - minimum)
        if sector in snapshot.top_sectors:
            raw += 0.15
        return clamp(raw, 0.0, 1.0)


def trailing_return(frame: pd.DataFrame, periods: int = 5) -> float:
    if frame.empty or len(frame) <= periods:
        return 0.0
    close = frame["close"]
    return float(close.iloc[-1] / close.iloc[-periods - 1] - 1)
