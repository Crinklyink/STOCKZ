"""Fast cached-daily prefilter for two-stage scans."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

import pandas as pd

from stock_predictor.analysis.rs_rating import build_close_matrix, compute_rs_scores
from stock_predictor.utils import clamp


@dataclass(slots=True)
class PrefilterEntry:
    ticker: str
    sector: str
    rs_score: float
    momentum_5d_score: float
    volume_ratio_score: float
    stage1_score: float


@dataclass(slots=True)
class PrefilterResult:
    total_tickers: int
    warm_tickers: int
    cache_hit_rate: float
    survivors: List[str]
    top_rows: List[PrefilterEntry]

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_tickers": self.total_tickers,
            "warm_tickers": self.warm_tickers,
            "cache_hit_rate": self.cache_hit_rate,
            "survivors": self.survivors,
            "top_rows": [asdict(row) for row in self.top_rows],
        }


def build_fast_prefilter(
    *,
    universe: Dict[str, List[str]],
    daily_frames: Dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    survivor_limit: int,
) -> PrefilterResult:
    assert all(isinstance(frame, pd.DataFrame) for frame in daily_frames.values()), (
        "Stage 1 must only use pre-loaded frames"
    )
    sector_by_ticker = {
        ticker: sector
        for sector, tickers in universe.items()
        for ticker in tickers
    }
    usable_frames = {
        ticker: frame
        for ticker, frame in daily_frames.items()
        if not frame.empty and {"close", "volume"}.issubset(frame.columns)
    }
    total = len(sector_by_ticker)
    warm = len(usable_frames)
    if not usable_frames:
        return PrefilterResult(total, 0, 0.0, [], [])

    rs_scores = compute_rs_scores(usable_frames, benchmark)
    close_matrix = build_close_matrix(usable_frames)
    volume_matrix = pd.DataFrame(
        {ticker: frame["volume"] for ticker, frame in usable_frames.items()},
    ).sort_index().ffill()
    momentum_5d = (close_matrix / close_matrix.shift(5) - 1.0).iloc[-1].fillna(0.0)
    volume_ratio = (volume_matrix / volume_matrix.rolling(20).mean()).iloc[-1].fillna(0.0)

    rs_rank = pd.Series({ticker: metrics.get("rs_rating", 50.0) for ticker, metrics in rs_scores.items()}).reindex(
        close_matrix.columns,
        fill_value=50.0,
    )
    momentum_rank = momentum_5d.rank(pct=True).mul(100.0).fillna(50.0)
    volume_rank = volume_ratio.rank(pct=True).mul(100.0).fillna(50.0)
    stage1_score = (
        0.45 * rs_rank
        + 0.35 * momentum_rank
        + 0.20 * volume_rank
    ).sort_values(ascending=False)

    top_rows = [
        PrefilterEntry(
            ticker=ticker,
            sector=sector_by_ticker.get(ticker, "Unknown"),
            rs_score=round(float(rs_rank.get(ticker, 50.0)), 2),
            momentum_5d_score=round(float(momentum_rank.get(ticker, 50.0)), 2),
            volume_ratio_score=round(float(volume_rank.get(ticker, 50.0)), 2),
            stage1_score=round(float(score), 2),
        )
        for ticker, score in stage1_score.items()
    ]
    survivors = [row.ticker for row in top_rows[:survivor_limit]]
    cache_hit_rate = clamp(warm / max(total, 1), 0.0, 1.0)
    return PrefilterResult(
        total_tickers=total,
        warm_tickers=warm,
        cache_hit_rate=cache_hit_rate,
        survivors=survivors,
        top_rows=top_rows,
    )
