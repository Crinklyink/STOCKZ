"""Momentum watchlist building and persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from stock_predictor.utils import clamp


@dataclass(slots=True)
class MomentumEntry:
    ticker: str
    sector: str
    five_day_return: float
    excess_return_vs_spy: float
    persistent: bool


@dataclass(slots=True)
class MomentumWatchlistResult:
    generated_at: str
    week: str
    top_50: List[MomentumEntry]
    persistent_tickers: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "generated_at": self.generated_at,
            "week": self.week,
            "top_50": [asdict(entry) for entry in self.top_50],
            "persistent_tickers": self.persistent_tickers,
        }


def trailing_return(frame: pd.DataFrame, periods: int = 5) -> float:
    if frame.empty or len(frame) <= periods:
        return 0.0
    close = frame["close"]
    return float(close.iloc[-1] / close.iloc[-periods - 1] - 1)


def load_watchlist(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_momentum_watchlist(
    *,
    benchmark: pd.DataFrame,
    bundles: Dict[str, object],
    path: Path,
) -> MomentumWatchlistResult:
    benchmark_return = trailing_return(benchmark, periods=5)
    prior = load_watchlist(path)
    prior_top = {
        str(entry.get("ticker", "")).upper()
        for entry in prior.get("top_50", [])
        if entry.get("ticker")
    }
    rows: List[MomentumEntry] = []
    for ticker, bundle in bundles.items():
        daily = getattr(bundle, "daily", pd.DataFrame())
        sector = str(getattr(bundle, "sector", "Unknown"))
        five_day = trailing_return(daily, periods=5)
        excess = five_day - benchmark_return
        rows.append(
            MomentumEntry(
                ticker=ticker,
                sector=sector,
                five_day_return=five_day,
                excess_return_vs_spy=excess,
                persistent=ticker in prior_top,
            )
        )
    top_50 = sorted(rows, key=lambda item: item.excess_return_vs_spy, reverse=True)[:50]
    persistent = sorted(entry.ticker for entry in top_50 if entry.persistent)
    result = MomentumWatchlistResult(
        generated_at=datetime.now(timezone.utc).isoformat(),
        week=datetime.now(timezone.utc).strftime("%Y-W%W"),
        top_50=top_50,
        persistent_tickers=persistent,
    )
    payload = result.to_dict()
    history = prior.get("history") if isinstance(prior.get("history"), list) else []
    trimmed_history = [row for row in history if isinstance(row, dict)][-11:]
    history_entry = result.to_dict()
    payload["history"] = [*trimmed_history, history_entry]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return result


def rank_priority_tickers(
    tickers: Iterable[str],
    watchlist: MomentumWatchlistResult | None,
) -> List[str]:
    if watchlist is None:
        return list(tickers)
    priority = {entry.ticker: index for index, entry in enumerate(watchlist.top_50)}
    return sorted(
        tickers,
        key=lambda ticker: (ticker not in priority, priority.get(ticker, 9999), ticker),
    )


def momentum_bonus_points(
    ticker: str,
    watchlist: MomentumWatchlistResult | None,
    bonus_points: float,
) -> float:
    if watchlist is None:
        return 0.0
    return bonus_points if ticker in set(watchlist.persistent_tickers) else 0.0


def float_rotation_ratio(price: float, average_volume: float, shares_outstanding: float) -> float:
    if price <= 0 or average_volume <= 0 or shares_outstanding <= 0:
        return 0.0
    dollar_volume = price * average_volume
    float_value = price * shares_outstanding
    return clamp(dollar_volume / max(float_value, 1.0), 0.0, 10.0)
