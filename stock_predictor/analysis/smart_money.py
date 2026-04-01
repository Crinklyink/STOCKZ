"""Retail-versus-institutional divergence analysis."""

from __future__ import annotations

from dataclasses import dataclass

from stock_predictor.utils import clamp


@dataclass(slots=True)
class SmartMoneyResult:
    score: float
    adjustment_points: float
    reject: bool
    label: str
    summary: str


def evaluate_smart_money_divergence(
    sentiment_score: float,
    options_score: float,
    institutional_score: float,
    bearish_flow_ratio: float = 0.0,
) -> SmartMoneyResult:
    """Compare retail sentiment with institutional positioning."""

    smart_score = clamp(0.55 * institutional_score + 0.45 * options_score - 0.2 * bearish_flow_ratio, 0.0, 1.0)
    retail_score = clamp(sentiment_score, 0.0, 1.0)
    if retail_score >= 0.65 and smart_score <= 0.45:
        return SmartMoneyResult(
            score=smart_score,
            adjustment_points=-20.0,
            reject=True,
            label="reject",
            summary="Retail is bullish while institutional flow is weak or bearish.",
        )
    if smart_score >= 0.65 and retail_score <= 0.45:
        return SmartMoneyResult(
            score=smart_score,
            adjustment_points=8.0,
            reject=False,
            label="strongest",
            summary="Institutions are buying while retail sentiment remains bearish.",
        )
    if smart_score >= 0.60 and retail_score >= 0.55:
        return SmartMoneyResult(
            score=smart_score,
            adjustment_points=4.0,
            reject=False,
            label="aligned",
            summary="Retail and institutional flows are both bullish.",
        )
    return SmartMoneyResult(
        score=smart_score,
        adjustment_points=0.0,
        reject=False,
        label="neutral",
        summary="No decisive smart-money divergence signal.",
    )

