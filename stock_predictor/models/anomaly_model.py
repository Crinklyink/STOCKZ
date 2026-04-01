"""Isolation-forest anomaly detection for unusual accumulation or breakdowns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from stock_predictor.analysis.indicators import add_indicators
from stock_predictor.utils import clamp


FEATURE_COLUMNS = ["return_1", "return_5", "return_20", "volume_delta", "atr", "vwap_distance", "mfi"]


@dataclass(slots=True)
class AnomalyResult:
    score: float
    is_anomaly: bool
    direction_aligned: bool
    adjustment_multiplier: float
    summary: str


class AnomalyDetector:
    """Train on normal behavior and flag unusual aligned breakouts."""

    def __init__(self) -> None:
        self.model: IsolationForest | None = None

    def fit(self, daily_frames: Dict[str, pd.DataFrame]) -> None:
        rows: List[np.ndarray] = []
        for frame in daily_frames.values():
            data = add_indicators(frame).dropna()
            if len(data) < 80:
                continue
            sample = data[FEATURE_COLUMNS].tail(120)
            rows.extend(sample.to_numpy(dtype=float))
        if len(rows) < 200:
            return
        self.model = IsolationForest(random_state=42, contamination=0.08)
        self.model.fit(np.asarray(rows))

    def score(self, frame: pd.DataFrame) -> AnomalyResult:
        data = add_indicators(frame).dropna()
        if data.empty or self.model is None:
            return AnomalyResult(0.0, False, False, 1.0, "Anomaly model unavailable")
        latest = data.iloc[-1]
        features = latest[FEATURE_COLUMNS].to_numpy(dtype=float).reshape(1, -1)
        raw_score = float(self.model.score_samples(features)[0])
        is_anomaly = bool(self.model.predict(features)[0] == -1)
        direction_aligned = bool(latest["close"] > latest["sma_20"] and latest["return_20"] > 0)
        normalized = clamp((0.2 - raw_score) / 0.4, 0.0, 1.0)
        multiplier = 1.15 if is_anomaly and direction_aligned else 0.90 if is_anomaly else 1.0
        summary = "Normal behavior."
        if is_anomaly and direction_aligned:
            summary = "Unusual bullish accumulation aligned with trend."
        elif is_anomaly:
            summary = "Abnormal move against prevailing trend."
        return AnomalyResult(
            score=normalized,
            is_anomaly=is_anomaly,
            direction_aligned=direction_aligned,
            adjustment_multiplier=multiplier,
            summary=summary,
        )

