"""Portfolio construction, diversification, and sizing."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from stock_predictor.config import AppConfig
from stock_predictor.scoring.composite import CandidateScore


def select_portfolio(
    config: AppConfig,
    candidates: Iterable[CandidateScore],
    price_frames: Dict[str, pd.DataFrame],
    *,
    threshold_used: float,
    top_n: int,
    vix: float = 0.0,
    benchmark: pd.DataFrame | None = None,
) -> List[CandidateScore]:
    """Select diversified top picks with sector and correlation constraints."""

    ordered = sorted(candidates, key=lambda item: item.final_score, reverse=True)
    if vix > config.thresholds.kill_switch_vix:
        return []

    allowed_sectors = set()
    if benchmark is not None and not benchmark.empty and len(benchmark) >= 50:
        close = benchmark["close"].dropna()
        spy_5d = float(close.iloc[-1] / close.iloc[-6] - 1.0) if len(close) >= 6 else 0.0
        spy_above_50ma = bool(close.iloc[-1] > close.rolling(50).mean().iloc[-1])
        if spy_5d < -0.02 and not spy_above_50ma:
            allowed_sectors = {"Healthcare", "Consumer Staples", "Consumer Defensive", "Utilities"}

    normalized_allowed_sectors = {_normalized_sector(sector) for sector in allowed_sectors}
    eligible: List[CandidateScore] = []
    for candidate in ordered:
        if candidate.final_score < threshold_used:
            continue
        if candidate.confluence_count < 3:
            continue
        if candidate.risk_reward < 1.0:
            continue
        if normalized_allowed_sectors and _normalized_sector(candidate.sector) not in normalized_allowed_sectors:
            continue
        if vix > 25.0:
            if candidate.final_score < 70.0:
                continue
            if candidate.ml_score < 70.0:
                continue
            if "TIER 1" not in candidate.tier_label:
                continue
        eligible.append(candidate)

    strict_mode = bool(allowed_sectors) or vix > 25.0
    ordered = eligible if (strict_mode or eligible) else ordered
    selected: List[CandidateScore] = []
    used_sectors = set()
    correlation_matrix = build_correlation_matrix(price_frames)
    minimum_returned = min(max(config.thresholds.minimum_returned_picks, 1), max(top_n, 1))
    effective_top_n = min(top_n, 5) if vix > 25.0 else top_n

    for candidate in ordered:
        if candidate.final_score < threshold_used:
            continue
        if candidate.sector in used_sectors:
            continue
        if has_high_correlation(candidate.ticker, selected, correlation_matrix, config.thresholds.max_correlation):
            continue
        selected.append(candidate)
        used_sectors.add(candidate.sector)
        if len(selected) == effective_top_n:
            return selected

    if config.allow_sector_override_for_diversification and len(selected) < effective_top_n:
        for candidate in ordered:
            if candidate in selected:
                continue
            if candidate.final_score < threshold_used:
                continue
            if has_high_correlation(candidate.ticker, selected, correlation_matrix, config.thresholds.max_correlation):
                continue
            selected.append(candidate)
            if len(selected) == effective_top_n:
                break
    if strict_mode:
        return selected
    if len(selected) < minimum_returned:
        for candidate in ordered:
            if candidate in selected:
                continue
            selected.append(candidate)
            if len(selected) == minimum_returned:
                break
    if len(selected) < effective_top_n:
        for candidate in ordered:
            if candidate in selected:
                continue
            if has_high_correlation(candidate.ticker, selected, correlation_matrix, config.thresholds.max_correlation):
                continue
            selected.append(candidate)
            if len(selected) == effective_top_n:
                break
    if len(selected) < effective_top_n:
        for candidate in ordered:
            if candidate in selected:
                continue
            selected.append(candidate)
            if len(selected) == effective_top_n:
                break
    return selected


def _normalized_sector(sector: str) -> str:
    normalized = (sector or "").strip().lower()
    aliases = {
        "consumer staples": "consumer defensive",
        "consumer defensive": "consumer defensive",
        "healthcare": "healthcare",
        "utilities": "utilities",
    }
    return aliases.get(normalized, normalized)


def build_correlation_matrix(price_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    returns = {}
    for ticker, frame in price_frames.items():
        if frame.empty or len(frame) < 40:
            continue
        returns[ticker] = frame["close"].pct_change().dropna().tail(60)
    if not returns:
        return pd.DataFrame()
    return pd.DataFrame(returns).corr()


def has_high_correlation(
    ticker: str,
    selected: List[CandidateScore],
    correlation_matrix: pd.DataFrame,
    threshold: float,
) -> bool:
    if correlation_matrix.empty:
        return False
    for existing in selected:
        if ticker not in correlation_matrix.index or existing.ticker not in correlation_matrix.columns:
            continue
        correlation = correlation_matrix.loc[ticker, existing.ticker]
        if np.isfinite(correlation) and correlation > threshold:
            return True
    return False
