"""Model ensembling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from stock_predictor.utils import clamp


@dataclass(slots=True)
class EnsembleOutput:
    probability: float
    lstm_probability: float
    xgb_probability: float
    tft_probability: float | None
    model_status: str
    model_spread: float
    score_uncertainty: float
    confidence_label: str
    model_family: str = "legacy"
    blend_weights: Dict[str, float] | None = None


def blend_probabilities(
    lstm_probability: float,
    xgb_probability: float,
    tft_probability: float | None = None,
) -> EnsembleOutput:
    """Blend the model outputs into a single probability."""

    active = [
        ("lstm", lstm_probability, 0.40),
        ("xgb", xgb_probability, 0.35),
    ]
    if tft_probability is not None:
        active.append(("tft", tft_probability, 0.25))
    total_weight = sum(weight for _, _, weight in active) or 1.0
    probability = sum(probability * weight for _, probability, weight in active) / total_weight
    probabilities = [probability for _, probability, _ in active]
    spread = max(probabilities) - min(probabilities)
    uncertainty = clamp(spread * 30.0, 2.0, 12.0)
    confidence_label = "high" if spread <= 0.10 else "medium" if spread <= 0.20 else "low"
    return EnsembleOutput(
        probability=clamp(probability, 0.0, 1.0),
        lstm_probability=lstm_probability,
        xgb_probability=xgb_probability,
        tft_probability=tft_probability,
        model_status="blended",
        model_spread=spread,
        score_uncertainty=uncertainty,
        confidence_label=confidence_label,
        model_family="legacy",
        blend_weights={name: weight / total_weight for name, _, weight in active},
    )


def build_tree_ensemble_output(
    *,
    probability: float,
    primary_probability: float,
    secondary_probability: float | None = None,
    model_status: str = "trained",
    confidence_label: str = "medium",
    model_family: str = "XGB+LGBM",
    blend_weights: Dict[str, float] | None = None,
) -> EnsembleOutput:
    probabilities = [primary_probability]
    if secondary_probability is not None:
        probabilities.append(secondary_probability)
    spread = max(probabilities) - min(probabilities) if len(probabilities) > 1 else 0.0
    uncertainty = clamp(spread * 30.0, 2.0, 12.0) if len(probabilities) > 1 else 4.0
    return EnsembleOutput(
        probability=clamp(probability, 0.0, 1.0),
        lstm_probability=clamp(primary_probability, 0.0, 1.0),
        xgb_probability=clamp(
            secondary_probability if secondary_probability is not None else primary_probability,
            0.0,
            1.0,
        ),
        tft_probability=None,
        model_status=model_status,
        model_spread=spread,
        score_uncertainty=uncertainty,
        confidence_label=confidence_label,
        model_family=model_family,
        blend_weights=blend_weights,
    )
