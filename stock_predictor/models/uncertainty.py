"""Uncertainty quantification for bootstrap tree ensembles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from stock_predictor.config import AppConfig, get_config
from stock_predictor.utils import clamp


@dataclass(slots=True)
class UncertaintyEstimate:
    mean_probability: float
    stddev: float
    confidence_label: str
    allow_pick: bool
    suggested_position_size_pct: float
    disagreement: float

    def to_dict(self) -> dict[str, float | str | bool]:
        return asdict(self)


class UncertaintyModel:
    """Turn model disagreement into a confidence signal."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()

    def estimate(self, probabilities: Iterable[float]) -> UncertaintyEstimate:
        values = np.asarray([float(value) for value in probabilities], dtype=float)
        if values.size == 0:
            values = np.asarray([0.5], dtype=float)
        mean_probability = float(np.mean(values))
        # Use percentile range for robust uncertainty estimation: less sensitive to outliers
        # in small bootstrap ensembles (typically 5–20 models).  The 90th-10th percentile
        # range of a normal distribution equals ~2.56 * stddev, so dividing by 2.56 yields
        # a value on the same scale as std, keeping the existing config thresholds valid.
        if values.size >= 4:
            p10 = float(np.percentile(values, 10))
            p90 = float(np.percentile(values, 90))
            stddev = (p90 - p10) / 2.56
        else:
            stddev = float(np.std(values))
        if stddev > self.config.adaptive_uncertainty_block:
            confidence = "low"
            allow_pick = False
            size = 0.01
        elif stddev > self.config.adaptive_uncertainty_high:
            confidence = "low"
            allow_pick = True
            size = 0.01
        elif stddev > self.config.adaptive_uncertainty_warn:
            confidence = "medium"
            allow_pick = True
            size = 0.02
        else:
            confidence = "high"
            allow_pick = True
            size = 0.03
        disagreement = float(clamp(stddev * 5.0, 0.0, 1.0))
        return UncertaintyEstimate(
            mean_probability=clamp(mean_probability, 0.0, 1.0),
            stddev=stddev,
            confidence_label=confidence,
            allow_pick=allow_pick,
            suggested_position_size_pct=size,
            disagreement=disagreement,
        )
