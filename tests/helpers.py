from __future__ import annotations

import numpy as np
import pandas as pd

from stock_predictor.scoring.composite import CandidateScore


def make_ohlcv(
    periods: int = 260,
    freq: str = "D",
    start_price: float = 100.0,
    drift: float = 0.002,
    volume_base: float = 1_000_000,
    last_boost: float = 0.0,
    volatility: float = 0.0015,
) -> pd.DataFrame:
    if periods <= 0:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    end = pd.Timestamp.now(tz="UTC").floor("h")
    index = pd.date_range(end=end, periods=periods, freq=freq, tz="UTC")
    wave = volatility * np.sin(np.linspace(0, 8 * np.pi, periods))
    returns = drift + wave
    close = start_price * np.cumprod(1 + returns)
    if last_boost:
        close[-5:] = close[-5:] * (1 + np.linspace(0.01, last_boost, 5))
    open_ = close * (1 - 0.002)
    high = close * 1.01
    low = close * 0.99
    volume = volume_base * (1 + 0.05 * np.sin(np.linspace(0, 4 * np.pi, periods)))
    if last_boost:
        volume[-5:] = volume[-5:] * 1.8
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def make_info(**overrides):
    base = {
        "currentPrice": 100.0,
        "averageVolume": 1_500_000,
        "marketCap": 25_000_000_000,
        "shortName": "Test Corp",
        "shortPercentOfFloat": 0.18,
        "shortRatio": 3.0,
        "borrowRate": 0.03,
    }
    base.update(overrides)
    return base


def make_candidate(ticker: str = "NVDA", sector: str = "Technology") -> CandidateScore:
    return CandidateScore(
        ticker=ticker,
        sector=sector,
        company_name="Test Corp",
        current_price=100.0,
        market_cap=25_000_000_000,
        final_score=82.0,
        score_low=76.0,
        score_high=88.0,
        score_uncertainty=6.0,
        confidence_label="high",
        ml_score=84.0,
        technical_score=80.0,
        volume_momentum_score=78.0,
        options_score=78.0,
        sentiment_score=72.0,
        rs_score=88.0,
        institutional_score=75.0,
        news_score=70.0,
        probability_4pct_5d=68.0,
        pattern_name="bull_flag",
        pattern_score=8.5,
        pattern_win_rate=73.0,
        pattern_win_rate_label="Bull Flag - 73.0% win rate (last 50 occurrences)",
        confluence_count=5,
        risk_reward=2.5,
        stop_loss=95.0,
        targets={"tp1": 102.0, "tp2": 104.0, "tp3": 107.0},
        position_size_pct=6.5,
        kelly_size_pct=4.2,
        data_quality_score=92.0,
        smart_money_score=81.0,
        squeeze_score=74.0,
        gpt_news_score=76.0,
        gpt_news_reason="Recent headlines point to a credible near-term catalyst.",
        anomaly_score=61.0,
        sector_tailwind_points=3.0,
        congress_score=50.0,
        supply_chain_score=60.0,
        tier_label="🥇 TIER 1",
        sector_temperature_tag="🔥 HOT SECTOR",
        meets_threshold=True,
        threshold_used=60.0,
        defaulted_signals={"sentiment": False, "flow": False},
        ai_explanation="Institutions are buying into a strong technical setup.",
        notes=["Bullish pattern", "Institutions buying"],
        diagnostics={
            "price_chart": [
                {"date": "2026-03-01", "open": 95.0, "high": 96.0, "low": 94.0, "close": 95.5, "volume": 1_000_000},
                {"date": "2026-03-02", "open": 95.5, "high": 97.0, "low": 95.0, "close": 96.5, "volume": 1_100_000},
            ]
        },
    )
