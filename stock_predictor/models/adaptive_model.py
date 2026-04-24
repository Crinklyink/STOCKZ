"""Adaptive regime-specific model layer with online updates and uncertainty."""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from stock_predictor.config import AppConfig, get_config
from stock_predictor.models.feature_health import FeatureHealthTracker
from stock_predictor.models.online_learner import OnlineLearningWrapper
from stock_predictor.models.regime_classifier import REGIME_LABELS, RegimeClassifier, RegimeSnapshot, position_size_from_regime
from stock_predictor.models.uncertainty import UncertaintyModel
from stock_predictor.models.xgboost_model import FEATURE_COLUMNS, LABEL_DEFINITION, TrainingReport, XGBOutput, XGBoostPredictor
from stock_predictor.utils import clamp

LOGGER = logging.getLogger(__name__)


ADAPTIVE_EXTRA_FEATURES = [
    "momentum_regime_score",
    "rotation_speed",
    "earnings_quality",
    "liquidity_stress",
    "intermarket_divergence",
]


class AdaptivePredictor:
    """Adaptive regime-aware wrapper around the existing feature pipeline."""

    model_family = "AdaptiveRegimeEnsemble"

    def __init__(self, config: AppConfig | None = None, horizon_days: int = 5) -> None:
        self.config = config or get_config()
        self.horizon_days = horizon_days
        self.feature_engine = XGBoostPredictor(self.config, horizon_days=horizon_days)
        self.regime_classifier = RegimeClassifier(self.config)
        self.online_learner = OnlineLearningWrapper(self.config)
        self.feature_health = FeatureHealthTracker(self.config)
        self.uncertainty_model = UncertaintyModel(self.config)
        self.active_regime = "neutral"
        self.active_breadth_percentile = 0.5
        self.active_market_snapshot: RegimeSnapshot | None = None
        self.regime_ensembles: Dict[str, list[object]] = {}
        self.regime_training_counts: Dict[str, int] = {}
        self.regime_feature_columns: Dict[str, List[str]] = {}
        self.regime_win_rates: Dict[str, float] = {}
        self.feature_columns = FEATURE_COLUMNS.copy() + ADAPTIVE_EXTRA_FEATURES
        self.training_samples = 0
        self.validation_samples = 0
        self.selected_profile = "adaptive_regime_ensemble"
        self.last_report: TrainingReport | None = None
        self._latest_regime_history = pd.DataFrame()
        self._load()

    @property
    def is_trained(self) -> bool:
        return bool(self.regime_ensembles)

    def fit(
        self,
        daily_frames: Dict[str, pd.DataFrame],
        *,
        cutoff: pd.Timestamp | None = None,
        save_model: bool = True,
        sector_map: Dict[str, str] | None = None,
        sector_histories: Dict[str, pd.DataFrame] | None = None,
        vix_history: pd.DataFrame | None = None,
        earnings_dates_map: Dict[str, List[Dict[str, object]]] | None = None,
        benchmark_history: pd.DataFrame | None = None,
        breadth_history: pd.Series | pd.DataFrame | None = None,
        put_call_history: pd.DataFrame | None = None,
        hyg_history: pd.DataFrame | None = None,
        lqd_history: pd.DataFrame | None = None,
        tlt_history: pd.DataFrame | None = None,
    ) -> TrainingReport:
        dataset = self.feature_engine.build_training_frame(
            daily_frames,
            cutoff=cutoff,
            sector_map=sector_map,
            sector_histories=sector_histories,
            vix_history=vix_history,
            earnings_dates_map=earnings_dates_map,
            benchmark_history=benchmark_history,
            breadth_history=breadth_history,
        )
        dataset = self._augment_dataset(
            dataset,
            daily_frames=daily_frames,
            benchmark_history=benchmark_history,
            breadth_history=breadth_history,
            sector_histories=sector_histories,
            vix_history=vix_history,
            put_call_history=put_call_history,
            hyg_history=hyg_history,
            lqd_history=lqd_history,
            tlt_history=tlt_history,
        )
        self.training_samples = len(dataset)
        if dataset.empty or len(dataset) < 250 or dataset["target"].nunique() < 2:
            report = TrainingReport(
                trained=False,
                training_samples=len(dataset),
                validation_samples=0,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                auc=0.0,
                positive_ratio=0.0,
                negative_ratio=0.0,
                scale_pos_weight=1.0,
                feature_importance={},
                trained_at=datetime.now(timezone.utc).isoformat(),
                model_family=self.model_family,
                selected_profile=self.selected_profile,
                label_definition=f"{LABEL_DEFINITION} Regime-specific adaptive ensemble.",
            )
            self.last_report = report
            return report

        dataset = dataset.sort_values("date").reset_index(drop=True)
        cv_result = self._walk_forward_cv(dataset)
        self.validation_samples = int(cv_result.get("validation_samples", 0))
        self.regime_win_rates = {
            regime: float(metrics["win_rate"])
            for regime, metrics in cv_result.get("regime_summary", {}).items()
        }
        self._train_all_regime_ensembles(dataset)
        feature_importance = self._aggregate_feature_importance()
        positive_ratio = float(dataset["target"].mean())
        negative_ratio = 1.0 - positive_ratio
        predictions = np.asarray(cv_result.get("predictions", []), dtype=float)
        targets = np.asarray(cv_result.get("targets", []), dtype=int)
        binary = (predictions >= 0.5).astype(int) if predictions.size else np.zeros(0, dtype=int)
        report = TrainingReport(
            trained=True,
            training_samples=len(dataset),
            validation_samples=self.validation_samples,
            accuracy=float(accuracy_score(targets, binary)) if targets.size else 0.0,
            precision=float(precision_score(targets, binary, zero_division=0)) if targets.size else 0.0,
            recall=float(recall_score(targets, binary, zero_division=0)) if targets.size else 0.0,
            auc=float(cv_result.get("auc", 0.0)),
            positive_ratio=positive_ratio,
            negative_ratio=negative_ratio,
            scale_pos_weight=float(negative_ratio / max(positive_ratio, 1e-6)),
            feature_importance=feature_importance,
            trained_at=datetime.now(timezone.utc).isoformat(),
            xgb_auc=float(cv_result.get("auc", 0.0)),
            lightgbm_auc=0.0,
            ensemble_auc=float(cv_result.get("auc", 0.0)),
            fold_aucs=list(cv_result.get("rows", [])),
            ensemble_weights={"xgb": 1.0, "lgbm": 0.0},
            label_definition=f"{LABEL_DEFINITION} Regime-specific adaptive ensemble.",
            model_family=self.model_family,
            selected_profile=self.selected_profile,
        )
        self.last_report = report
        if save_model:
            self._save(report, cv_result)
        return report

    def predict_proba(
        self,
        frame: pd.DataFrame,
        *,
        sector_rs_rank: float = 50.0,
        weeks_since_earnings: float | None = None,
        earnings_dates: List[Dict[str, object]] | None = None,
        earnings_proximity_score: float | None = None,
        breadth_percentile: float = 0.5,
        regime: str | None = None,
    ) -> XGBOutput:
        active_regime = regime or self.active_regime or "neutral"
        features = self.feature_engine._prepare_inference_frame(
            frame,
            sector_rank_series=pd.Series(sector_rs_rank, index=frame.index),
            earnings_dates=earnings_dates,
            weeks_since_earnings_override=weeks_since_earnings,
            earnings_proximity_override=earnings_proximity_score,
            breadth_series=pd.Series(breadth_percentile, index=frame.index),
        )
        if features.empty:
            probability = self.feature_engine._heuristic_probability(frame)
            return XGBOutput(probability=probability, status="heuristic")
        latest = features.iloc[-1].to_dict()
        adaptive_features = self._adaptive_features_for_inference(active_regime, breadth_percentile)
        latest.update(adaptive_features)
        feature_vector = {
            column: float(latest.get(column, 0.0))
            for column in self.feature_columns
        }
        regime_models = self.regime_ensembles.get(active_regime) or self.regime_ensembles.get("neutral") or []
        if not regime_models:
            probability = self.feature_engine._heuristic_probability(frame)
            return XGBOutput(probability=probability, status="heuristic", regime=active_regime)

        probabilities = []
        row_frame = pd.DataFrame([feature_vector], columns=self.feature_columns)
        for model in regime_models:
            probabilities.append(float(self.feature_engine._predict_model_probabilities(model, row_frame, None)[0]))
        uncertainty = self.uncertainty_model.estimate(probabilities)
        online_probability = None
        if self.config.feature_flags.online_learning:
            online_probability = self.online_learner.predict_proba(active_regime, self._online_features(feature_vector))
        probability = uncertainty.mean_probability
        weights = {"xgb": 1.0, "lgbm": 0.0}
        if online_probability is not None:
            tree_weight = 1.0 - self.config.adaptive_online_blend
            online_weight = self.config.adaptive_online_blend
            probability = tree_weight * probability + online_weight * online_probability
            weights = {"xgb": tree_weight, "lgbm": online_weight}
        if not uncertainty.allow_pick:
            probability = 0.5
        regime_win_rate = float(self.regime_win_rates.get(active_regime, 0.5))
        position_size = position_size_from_regime(
            regime=active_regime,
            model_confidence=probability,
            uncertainty=uncertainty.stddev,
            regime_historical_win_rate=regime_win_rate,
            vix=float(self.active_market_snapshot.vix) if self.active_market_snapshot else 20.0,
        )
        return XGBOutput(
            probability=clamp(probability, 0.0, 1.0),
            status=active_regime,
            xgb_probability=clamp(uncertainty.mean_probability, 0.0, 1.0),
            lightgbm_probability=clamp(online_probability, 0.0, 1.0) if online_probability is not None else None,
            blend_weights=weights,
            model_spread=uncertainty.stddev,
            score_uncertainty=clamp(uncertainty.stddev * 100.0, 2.0, 15.0),
            confidence_label=uncertainty.confidence_label,
            regime=active_regime,
            position_size_pct=position_size,
        )

    def walk_forward_backtest(
        self,
        daily_frames: Dict[str, pd.DataFrame],
        *,
        benchmark_history: pd.DataFrame | None = None,
        breadth_history: pd.Series | pd.DataFrame | None = None,
        sector_histories: Dict[str, pd.DataFrame] | None = None,
        vix_history: pd.DataFrame | None = None,
        put_call_history: pd.DataFrame | None = None,
        hyg_history: pd.DataFrame | None = None,
        lqd_history: pd.DataFrame | None = None,
        tlt_history: pd.DataFrame | None = None,
        weeks: int = 52,
    ) -> Dict[str, object]:
        dataset = self.feature_engine.build_training_frame(
            daily_frames,
            benchmark_history=benchmark_history,
            breadth_history=breadth_history,
        )
        dataset = self._augment_dataset(
            dataset,
            daily_frames=daily_frames,
            benchmark_history=benchmark_history,
            breadth_history=breadth_history,
            sector_histories=sector_histories,
            vix_history=vix_history,
            put_call_history=put_call_history,
            hyg_history=hyg_history,
            lqd_history=lqd_history,
            tlt_history=tlt_history,
        )
        if dataset.empty:
            return {"rows": [], "summary": {}, "markdown": "No adaptive backtest rows were available."}
        dataset["date"] = pd.to_datetime(dataset["date"], utc=True)
        weekly_points = sorted(dataset["date"].drop_duplicates())[-weeks:]
        rows: list[dict[str, object]] = []
        regime_rows: dict[str, list[float]] = {label: [] for label in REGIME_LABELS}
        weekly_returns: list[float] = []
        weekly_hits: list[float] = []
        for week_start in weekly_points:
            train = dataset.loc[dataset["date"] < week_start]
            test = dataset.loc[dataset["date"] == week_start]
            train = self.feature_engine._purge_train_rows(train, test_start=week_start)
            if len(train) < 250 or test.empty or train["target"].nunique() < 2:
                continue
            regime_models = self._fit_regime_models(
                train,
                model_count=max(3, int(self.config.adaptive_backtest_models)),
            )
            probabilities = self._predict_regime_probabilities(test, regime_models)
            if probabilities.size == 0:
                continue
            scored = test.copy()
            scored["probability"] = probabilities
            top = scored.sort_values("probability", ascending=False).head(10)
            avg_return = float(top["future_return"].mean())
            win_rate = float((top["target"] == 1).mean())
            weekly_returns.append(avg_return)
            weekly_hits.append(win_rate)
            top_regime = str(top["regime"].mode().iloc[0]) if not top.empty else "neutral"
            regime_rows.setdefault(top_regime, []).append(avg_return)
            rows.append(
                {
                    "week": pd.Timestamp(week_start).strftime("%Y-%m-%d"),
                    "regime": top_regime,
                    "picks": int(len(top)),
                    "win_rate": round(win_rate * 100.0, 2),
                    "avg_return": round(avg_return * 100.0, 2),
                }
            )
        sharpe = 0.0
        if len(weekly_returns) > 1 and np.std(weekly_returns) > 0:
            sharpe = float(np.mean(weekly_returns) / np.std(weekly_returns) * np.sqrt(52))
        summary = {
            "weeks": len(rows),
            "win_rate": round(np.mean(weekly_hits) * 100.0 if weekly_hits else 0.0, 2),
            "average_return": round(np.mean(weekly_returns) * 100.0 if weekly_returns else 0.0, 2),
            "sharpe": round(sharpe, 2),
        }
        regime_summary = {}
        for regime, returns in regime_rows.items():
            regime_weeks = [row for row in rows if row["regime"] == regime]
            if not regime_weeks:
                continue
            regime_summary[regime] = {
                "weeks": len(regime_weeks),
                "win_rate": float(np.mean([row["win_rate"] for row in regime_weeks])),
                "average_return": float(np.mean([row["avg_return"] for row in regime_weeks])),
            }
        markdown_lines = [
            "# Adaptive Regime Backtest",
            "",
            "## Regime Results",
            "",
            "| Regime | Weeks | Win Rate | Avg Return |",
            "| --- | ---: | ---: | ---: |",
        ]
        for regime in REGIME_LABELS:
            metrics = regime_summary.get(regime)
            if not metrics:
                continue
            markdown_lines.append(
                f"| {regime.upper()} | {metrics['weeks']} | {metrics['win_rate']:.2f}% | {metrics['average_return']:.2f}% |"
            )
        markdown_lines.extend(
            [
                "",
                f"OVERALL: {summary['weeks']} weeks | {summary['win_rate']:.2f}% win rate | {summary['average_return']:.2f}% avg | Sharpe {summary['sharpe']:.2f}",
            ]
        )
        return {
            "rows": rows,
            "summary": summary,
            "regime_summary": regime_summary,
            "markdown": "\n".join(markdown_lines),
        }

    def apply_feedback(self, completed_rows: Iterable[dict[str, object]]) -> Dict[str, object]:
        updates = self.online_learner.apply_rows(completed_rows)
        health = self.feature_health.refresh() if self.config.feature_flags.feature_health_decay else self.feature_health.records
        drifted = [update.regime for update in updates if update.drift_detected]
        summary = {
            "updated_rows": len(updates),
            "drifted_regimes": drifted,
            "feature_health": {name: record.to_dict() for name, record in health.items()},
        }
        if self.last_report is not None:
            self._save(self.last_report, {"rows": list(self.last_report.fold_aucs), "regime_summary": {}})
        return summary

    def _augment_dataset(
        self,
        dataset: pd.DataFrame,
        *,
        daily_frames: Dict[str, pd.DataFrame],
        benchmark_history: pd.DataFrame | None,
        breadth_history: pd.Series | pd.DataFrame | None,
        sector_histories: Dict[str, pd.DataFrame] | None,
        vix_history: pd.DataFrame | None,
        put_call_history: pd.DataFrame | None,
        hyg_history: pd.DataFrame | None,
        lqd_history: pd.DataFrame | None,
        tlt_history: pd.DataFrame | None,
    ) -> pd.DataFrame:
        if dataset.empty or benchmark_history is None or benchmark_history.empty:
            return dataset
        regime_history = self.regime_classifier.build_regime_history(
            price_frames=daily_frames,
            benchmark_history=benchmark_history,
            breadth_history=breadth_history,
            sector_histories=sector_histories,
            vix_history=vix_history,
            put_call_history=put_call_history,
            hyg_history=hyg_history,
            lqd_history=lqd_history,
            tlt_history=tlt_history,
        )
        self._latest_regime_history = regime_history
        if regime_history.empty:
            for column in ADAPTIVE_EXTRA_FEATURES:
                dataset[column] = 0.5 if column != "liquidity_stress" else 0.02
            dataset["regime"] = "neutral"
            return dataset
        regime_history = regime_history.sort_index()
        augmented = dataset.copy()
        augmented["date"] = pd.to_datetime(augmented["date"], utc=True)
        for column in ["regime", *ADAPTIVE_EXTRA_FEATURES]:
            series = regime_history[column]
            augmented[column] = (
                series.reindex(augmented["date"], method="ffill")
                .reset_index(drop=True)
                .values
            )
        return augmented.dropna(subset=self.feature_columns + ["target"]).reset_index(drop=True)

    def _fit_regime_models(self, dataset: pd.DataFrame, *, model_count: int | None = None) -> Dict[str, list[object]]:
        trained: Dict[str, list[object]] = {}
        for regime in REGIME_LABELS:
            subset = dataset.loc[dataset["regime"] == regime].copy()
            if len(subset) < self.config.adaptive_regime_min_samples or subset["target"].nunique() < 2:
                continue
            trained[regime] = self._bootstrap_ensemble(subset, model_count=model_count)
        if not trained and not dataset.empty:
            trained["neutral"] = self._bootstrap_ensemble(dataset, model_count=model_count)
        return trained

    def _train_all_regime_ensembles(self, dataset: pd.DataFrame) -> None:
        self.regime_ensembles = self._fit_regime_models(
            dataset,
            model_count=max(5, int(self.config.adaptive_uncertainty_models)),
        )
        self.regime_training_counts = {
            regime: int(len(dataset.loc[dataset["regime"] == regime]))
            for regime in REGIME_LABELS
        }
        self.regime_feature_columns = {regime: self.feature_columns.copy() for regime in self.regime_ensembles}

    def _bootstrap_ensemble(self, subset: pd.DataFrame, *, model_count: int | None = None) -> list[object]:
        features = subset[self.feature_columns]
        target = subset["target"]
        sample_dates = subset["date"] if "date" in subset else None
        pos_count = int(target.sum())
        neg_count = int(len(target) - pos_count)
        scale_pos_weight = float(neg_count / max(pos_count, 1))
        models = []
        effective_model_count = max(int(model_count or self.config.adaptive_uncertainty_models), 3)
        for seed in range(effective_model_count):
            sample = subset.sample(frac=1.0, replace=True, random_state=42 + seed)
            model = self.feature_engine._build_model(
                scale_pos_weight=scale_pos_weight,
                overrides={"random_state": 42 + seed, "n_jobs": 1},
            )
            train_x, train_y, train_dates = self.feature_engine._balance_training_data(
                sample[self.feature_columns],
                sample["target"],
                sample_dates=sample["date"] if "date" in sample else None,
            )
            model.fit(
                train_x,
                train_y,
                **self.feature_engine._fit_kwargs(model, train_y, scale_pos_weight, sample_dates=train_dates),
            )
            models.append(model)
        return models

    def _predict_regime_probabilities(self, dataset: pd.DataFrame, regime_models: Dict[str, list[object]]) -> np.ndarray:
        probabilities = []
        for row in dataset.itertuples(index=False):
            regime = str(getattr(row, "regime", "neutral"))
            models = regime_models.get(regime) or regime_models.get("neutral") or []
            if not models:
                probabilities.append(0.5)
                continue
            feature_row = pd.DataFrame(
                [{column: float(getattr(row, column)) for column in self.feature_columns}],
                columns=self.feature_columns,
            )
            probs = [
                float(self.feature_engine._predict_model_probabilities(model, feature_row, None)[0])
                for model in models
            ]
            probabilities.append(float(np.mean(probs)))
        return np.asarray(probabilities, dtype=float)

    def _walk_forward_cv(self, dataset: pd.DataFrame) -> Dict[str, object]:
        frame = dataset.copy()
        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        frame["month"] = frame["date"].dt.tz_localize(None).dt.to_period("M").astype(str)
        months = sorted(frame["month"].unique())
        if len(months) <= 12:
            return {"rows": [], "auc": 0.5, "predictions": [], "targets": [], "regime_summary": {}, "validation_samples": 0}
        rows: list[dict[str, object]] = []
        all_probabilities: list[float] = []
        all_targets: list[int] = []
        regime_summary: Dict[str, dict[str, list[float]]] = {}
        for fold_index, month in enumerate(months[12:], start=1):
            test = frame.loc[frame["month"] == month].copy()
            if test.empty or test["target"].nunique() < 2:
                continue
            train = frame.loc[frame["month"] < month].copy()
            train = self.feature_engine._purge_train_rows(train, test_start=pd.to_datetime(test["date"].min(), utc=True))
            if len(train) < 250 or train["target"].nunique() < 2:
                continue
            models = self._fit_regime_models(
                train,
                model_count=max(3, int(self.config.adaptive_cv_models)),
            )
            probabilities = self._predict_regime_probabilities(test, models)
            auc = float(roc_auc_score(test["target"], probabilities)) if test["target"].nunique() > 1 else 0.5
            trade_metrics = self.feature_engine._evaluate_trade_basket(test, probabilities)
            fold_regime = str(test["regime"].mode().iloc[0]) if not test.empty else "neutral"
            regime_entry = regime_summary.setdefault(fold_regime, {"win_rates": [], "returns": []})
            regime_entry["win_rates"].append(float(trade_metrics["trade_win_rate"]))
            regime_entry["returns"].append(float(trade_metrics["trade_average_return"]))
            rows.append(
                {
                    "fold": fold_index,
                    "month": month,
                    "regime": fold_regime,
                    "auc": round(auc, 4),
                    "trade_win_rate": round(float(trade_metrics["trade_win_rate"]), 4),
                    "trade_average_return": round(float(trade_metrics["trade_average_return"]), 4),
                }
            )
            all_probabilities.extend(probabilities.tolist())
            all_targets.extend(test["target"].astype(int).tolist())
        reduced_summary = {
            regime: {
                "weeks": len(values["win_rates"]),
                "win_rate": float(np.mean(values["win_rates"])) if values["win_rates"] else 0.0,
                "average_return": float(np.mean(values["returns"])) if values["returns"] else 0.0,
            }
            for regime, values in regime_summary.items()
        }
        auc = float(roc_auc_score(all_targets, all_probabilities)) if len(set(all_targets)) > 1 else 0.5
        return {
            "rows": rows,
            "auc": auc,
            "predictions": all_probabilities,
            "targets": all_targets,
            "regime_summary": reduced_summary,
            "validation_samples": len(all_targets),
        }

    def _aggregate_feature_importance(self) -> Dict[str, float]:
        aggregate: Dict[str, float] = {column: 0.0 for column in self.feature_columns}
        total_models = 0
        for models in self.regime_ensembles.values():
            for model in models:
                if not hasattr(model, "feature_importances_"):
                    continue
                total_models += 1
                raw = np.asarray(model.feature_importances_, dtype=float)
                total = float(raw.sum()) or 1.0
                for feature, value in zip(self.feature_columns, raw / total, strict=False):
                    aggregate[feature] += float(value)
        if total_models <= 0:
            return {}
        normalized = {feature: value / total_models for feature, value in aggregate.items() if value > 0}
        return dict(sorted(normalized.items(), key=lambda item: item[1], reverse=True))

    def _adaptive_features_for_inference(self, regime: str, breadth_percentile: float) -> Dict[str, float]:
        if self.active_market_snapshot is None:
            return {
                "momentum_regime_score": 0.5,
                "rotation_speed": 0.04,
                "earnings_quality": 0.5,
                "liquidity_stress": 0.02,
                "intermarket_divergence": 0.02,
            }
        snapshot = self.active_market_snapshot
        return {
            "momentum_regime_score": float(snapshot.momentum_regime_score),
            "rotation_speed": float(snapshot.rotation_speed),
            "earnings_quality": float(snapshot.earnings_quality),
            "liquidity_stress": float(snapshot.liquidity_stress),
            "intermarket_divergence": float(snapshot.intermarket_divergence),
        }

    def _online_features(self, feature_vector: Dict[str, float]) -> Dict[str, float]:
        allowed = {
            "return_5",
            "return_20",
            "volume_ratio",
            "macd_hist",
            "rsi",
            "adx",
            "price_vs_52w_high",
            "sector_rs_rank",
            "trend_consistency",
            "price_acceleration",
            "momentum_regime_score",
            "rotation_speed",
            "earnings_quality",
            "liquidity_stress",
            "intermarket_divergence",
            "breadth_percentile",
        }
        return {
            key: float(value)
            for key, value in feature_vector.items()
            if key in allowed and pd.notna(value)
        }

    def feedback_rows_from_payloads(self, rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
        payload_rows: list[dict[str, object]] = []
        for row in rows:
            payload = row.get("payload")
            if not isinstance(payload, dict):
                continue
            diagnostics = payload.get("diagnostics", {}) if isinstance(payload.get("diagnostics"), dict) else {}
            subscores = diagnostics.get("subscores", {}) if isinstance(diagnostics, dict) else {}
            technical = subscores.get("technical_signals", {}) if isinstance(subscores, dict) else {}
            online_features = {
                "ml_score": float(payload.get("ml_score", 50.0)),
                "technical_score": float(payload.get("technical_score", 50.0)),
                "volume_momentum_score": float(payload.get("volume_momentum_score", 50.0)),
                "options_score": float(payload.get("options_score", 50.0)),
                "sentiment_score": float(payload.get("sentiment_score", 50.0)),
                "rs_score": float(payload.get("rs_score", 50.0)),
                "pattern_score": float(payload.get("pattern_score", 0.0)),
            }
            for feature in ("rsi", "macd", "volume", "price_vs_50ma", "bollinger", "adx", "momentum", "vwap"):
                if feature in technical:
                    online_features[f"tech_{feature}"] = float(technical.get(feature, 50.0))
            payload_rows.append(
                {
                    "regime": str(row.get("regime") or payload.get("regime_label") or self.active_regime or "neutral"),
                    "resolved_target_hit": bool(row.get("resolved_target_hit")),
                    "online_features": online_features,
                }
            )
        return payload_rows

    def _save(self, report: TrainingReport, cv_result: Dict[str, object]) -> None:
        self.config.model_dir.mkdir(parents=True, exist_ok=True)
        for regime, path in self.config.adaptive_regime_paths.items():
            models = self.regime_ensembles.get(regime)
            if not models:
                continue
            with path.open("wb") as handle:
                pickle.dump(
                    {
                        "models": models,
                        "feature_columns": self.feature_columns,
                        "training_count": self.regime_training_counts.get(regime, 0),
                    },
                    handle,
                )
        metadata = report.to_dict()
        metadata.update(
            {
                "model_family": self.model_family,
                "regime_training_counts": self.regime_training_counts,
                "regime_win_rates": self.regime_win_rates,
                "feature_columns": self.feature_columns,
                "feature_health_path": str(self.config.feature_health_path),
                "online_learner_path": str(self.config.online_learner_path),
                "regime_summary": cv_result.get("regime_summary", {}),
                "trained_regimes": sorted(self.regime_ensembles.keys()),
            }
        )
        self.config.adaptive_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        # Keep the existing UI surfaces reading a current metadata file.
        self.config.xgb_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.feature_health.save()
        self.online_learner.save()

    def _load(self) -> None:
        if not self.config.adaptive_metadata_path.exists():
            return
        try:
            metadata = json.loads(self.config.adaptive_metadata_path.read_text(encoding="utf-8"))
            self.feature_columns = list(metadata.get("feature_columns", self.feature_columns))
            self.regime_training_counts = {
                str(key): int(value)
                for key, value in metadata.get("regime_training_counts", {}).items()
            }
            self.regime_win_rates = {
                str(key): float(value)
                for key, value in metadata.get("regime_win_rates", {}).items()
            }
            self.last_report = TrainingReport(
                trained=bool(metadata.get("trained", True)),
                training_samples=int(metadata.get("training_samples", 0)),
                validation_samples=int(metadata.get("validation_samples", 0)),
                accuracy=float(metadata.get("accuracy", 0.0)),
                precision=float(metadata.get("precision", 0.0)),
                recall=float(metadata.get("recall", 0.0)),
                auc=float(metadata.get("auc", 0.0)),
                positive_ratio=float(metadata.get("positive_ratio", 0.0)),
                negative_ratio=float(metadata.get("negative_ratio", 0.0)),
                scale_pos_weight=float(metadata.get("scale_pos_weight", 1.0)),
                feature_importance={str(k): float(v) for k, v in metadata.get("feature_importance", {}).items()},
                trained_at=str(metadata.get("trained_at", "")),
                xgb_auc=float(metadata.get("xgb_auc", metadata.get("auc", 0.0))),
                lightgbm_auc=float(metadata.get("lightgbm_auc", 0.0)),
                ensemble_auc=float(metadata.get("ensemble_auc", metadata.get("auc", 0.0))),
                fold_aucs=list(metadata.get("fold_aucs", [])),
                ensemble_weights={"xgb": 1.0, "lgbm": 0.0},
                label_definition=str(metadata.get("label_definition", LABEL_DEFINITION)),
                model_family=self.model_family,
                selected_profile=self.selected_profile,
            )
            self.training_samples = self.last_report.training_samples
            self.validation_samples = self.last_report.validation_samples
            for regime, path in self.config.adaptive_regime_paths.items():
                if not path.exists():
                    continue
                with path.open("rb") as handle:
                    payload = pickle.load(handle)
                self.regime_ensembles[regime] = list(payload.get("models", []))
                self.regime_feature_columns[regime] = list(payload.get("feature_columns", self.feature_columns))
        except Exception:
            LOGGER.debug("Failed to load adaptive model state", exc_info=True)
            self.regime_ensembles = {}
            self.regime_feature_columns = {}
