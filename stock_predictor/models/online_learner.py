"""Online learning wrapper for weekly adaptive feedback."""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable

from stock_predictor.config import AppConfig, get_config
from stock_predictor.utils import clamp

try:  # pragma: no cover - optional dependency
    from river import drift
except Exception:  # pragma: no cover
    drift = None

try:  # pragma: no cover - optional dependency
    from river import forest
except Exception:  # pragma: no cover
    forest = None

try:  # pragma: no cover - compatibility for older river versions
    from river import ensemble as river_ensemble
except Exception:  # pragma: no cover
    river_ensemble = None


@dataclass(slots=True)
class OnlineUpdateResult:
    regime: str
    updated: bool
    drift_detected: bool
    samples_seen: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _build_river_model():
    detector = drift.ADWIN() if drift is not None else None
    warning = drift.ADWIN(delta=0.01) if drift is not None else None
    if river_ensemble is not None and hasattr(river_ensemble, "AdaptiveRandomForestClassifier"):
        return river_ensemble.AdaptiveRandomForestClassifier(
            n_models=10,
            max_features="sqrt",
            drift_detector=detector,
            warning_detector=warning,
            seed=42,
        )
    if forest is not None and hasattr(forest, "ARFClassifier"):
        return forest.ARFClassifier(
            n_models=10,
            max_features="sqrt",
            drift_detector=detector,
            warning_detector=warning,
            seed=42,
        )
    return None


class OnlineLearningWrapper:
    """Persist regime-specific online learners and update them from completed picks."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self.path = self.config.online_learner_path
        self.models: Dict[str, object] = {}
        self.sample_counts: Dict[str, int] = {}
        self.drift_flags: Dict[str, bool] = {}
        self._load()

    def predict_proba(self, regime: str, features: Dict[str, float]) -> float | None:
        model = self.models.get(regime)
        if model is None:
            return None
        try:
            probability = model.predict_proba_one(features)
            if probability is None:
                return None
            if isinstance(probability, dict):
                return clamp(float(probability.get(True, probability.get(1, 0.5))), 0.0, 1.0)
            return clamp(float(probability), 0.0, 1.0)
        except Exception:
            return None

    def learn_one(self, regime: str, features: Dict[str, float], target: bool) -> OnlineUpdateResult:
        model = self.models.get(regime)
        if model is None:
            model = _build_river_model()
            if model is None:
                return OnlineUpdateResult(regime=regime, updated=False, drift_detected=False, samples_seen=0)
            self.models[regime] = model
        drift_detected = False
        if drift is not None:
            detector = getattr(model, "drift_detector", None)
            if detector is not None:
                detector.update(int(target))
                drift_detected = bool(getattr(detector, "drift_detected", False))
        if drift_detected:
            model = _build_river_model()
            if model is not None:
                self.models[regime] = model
                self.sample_counts[regime] = 0
        active = self.models.get(regime)
        if active is None:
            return OnlineUpdateResult(regime=regime, updated=False, drift_detected=drift_detected, samples_seen=0)
        active.learn_one(features, bool(target))
        self.sample_counts[regime] = int(self.sample_counts.get(regime, 0)) + 1
        self.drift_flags[regime] = drift_detected
        self.save()
        return OnlineUpdateResult(
            regime=regime,
            updated=True,
            drift_detected=drift_detected,
            samples_seen=self.sample_counts[regime],
        )

    def apply_rows(
        self,
        rows: Iterable[dict[str, object]],
        *,
        regime_key: str = "regime",
        target_key: str = "resolved_target_hit",
        feature_key: str = "online_features",
    ) -> list[OnlineUpdateResult]:
        results: list[OnlineUpdateResult] = []
        for row in rows:
            regime = str(row.get(regime_key) or "neutral")
            features = row.get(feature_key)
            if not isinstance(features, dict) or not features:
                continue
            results.append(self.learn_one(regime, {str(k): float(v) for k, v in features.items()}, bool(row.get(target_key))))
        return results

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as handle:
            pickle.dump(
                {
                    "models": self.models,
                    "sample_counts": self.sample_counts,
                    "drift_flags": self.drift_flags,
                },
                handle,
            )

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("rb") as handle:
                payload = pickle.load(handle)
            self.models = dict(payload.get("models", {}))
            self.sample_counts = {str(k): int(v) for k, v in payload.get("sample_counts", {}).items()}
            self.drift_flags = {str(k): bool(v) for k, v in payload.get("drift_flags", {}).items()}
        except Exception:
            self.models = {}
            self.sample_counts = {}
            self.drift_flags = {}
