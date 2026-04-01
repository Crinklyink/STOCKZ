"""IBD-style relative strength ranking."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from stock_predictor.utils import clamp


def compute_rs_scores(price_frames: Dict[str, pd.DataFrame], benchmark: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute weighted relative strength versus the benchmark."""

    close_frame = build_close_matrix(price_frames)
    if close_frame.empty or benchmark.empty:
        return {}
    benchmark_close = benchmark["close"].reindex(close_frame.index).ffill()
    benchmark_periods = {
        "1m": float((benchmark_close / benchmark_close.shift(21) - 1.0).iloc[-1]),
        "3m": float((benchmark_close / benchmark_close.shift(63) - 1.0).iloc[-1]),
        "6m": float((benchmark_close / benchmark_close.shift(126) - 1.0).iloc[-1]),
        "12m": float((benchmark_close / benchmark_close.shift(252) - 1.0).iloc[-1]),
    }
    weights = {"1m": 0.4, "3m": 0.3, "6m": 0.2, "12m": 0.1}
    stock_returns = {
        "1m": (close_frame / close_frame.shift(21) - 1.0).iloc[-1].fillna(0.0),
        "3m": (close_frame / close_frame.shift(63) - 1.0).iloc[-1].fillna(0.0),
        "6m": (close_frame / close_frame.shift(126) - 1.0).iloc[-1].fillna(0.0),
        "12m": (close_frame / close_frame.shift(252) - 1.0).iloc[-1].fillna(0.0),
    }
    weighted_excess = sum(
        weights[label] * (stock_returns[label] - benchmark_periods[label])
        for label in stock_returns
    )
    raw_scores = {
        ticker: {
            "weighted_excess_return": float(weighted_excess.get(ticker, 0.0)),
            "1m": float(stock_returns["1m"].get(ticker, 0.0)),
            "3m": float(stock_returns["3m"].get(ticker, 0.0)),
            "6m": float(stock_returns["6m"].get(ticker, 0.0)),
            "12m": float(stock_returns["12m"].get(ticker, 0.0)),
        }
        for ticker in close_frame.columns
    }
    ordered = sorted(raw_scores.items(), key=lambda item: item[1]["weighted_excess_return"])
    if not ordered:
        return {}
    total = max(len(ordered) - 1, 1)
    scores = {}
    for rank, (ticker, metrics) in enumerate(ordered):
        percentile = rank / total
        rs_rating = clamp(percentile * 100, 0.0, 100.0)
        scores[ticker] = {
            **metrics,
            "rs_rating": rs_rating,
            "rs_signal": 1.0 if rs_rating >= 85 else rs_rating / 85.0,
        }
    return scores


def build_close_matrix(price_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    series = {
        ticker: frame["close"]
        for ticker, frame in price_frames.items()
        if not frame.empty and "close" in frame
    }
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).sort_index().ffill()


def period_return(frame: pd.DataFrame, periods: int) -> float:
    if frame.empty or len(frame) <= periods:
        return 0.0
    close = frame["close"]
    return float(close.iloc[-1] / close.iloc[-periods - 1] - 1)
