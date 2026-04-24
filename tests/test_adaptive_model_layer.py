from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from stock_predictor.config import get_config
from stock_predictor.data.cache import SQLiteCache
from stock_predictor.data.fetcher import MarketDataFetcher
from stock_predictor.models.adaptive_model import AdaptivePredictor
from stock_predictor.models.feature_health import FeatureHealthTracker
from stock_predictor.models.regime_classifier import RegimeClassifier
from stock_predictor.models.uncertainty import UncertaintyModel
from stock_predictor.models.xgboost_model import FEATURE_COLUMNS
from tests.helpers import make_ohlcv


class _ConstantModel:
    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.feature_importances_ = [1.0 / len(FEATURE_COLUMNS)] * len(FEATURE_COLUMNS)

    def predict_proba(self, features: pd.DataFrame):
        return [[1.0 - self.probability, self.probability] for _ in range(len(features))]


class AdaptiveModelLayerTests(unittest.TestCase):
    def test_regime_classifier_distinguishes_bull_and_crisis(self) -> None:
        classifier = RegimeClassifier(get_config())

        self.assertEqual(
            classifier.classify(
                vix=15.0,
                vix_trend=-0.05,
                spy_20d_momentum=0.04,
                breadth=0.72,
                sector_dispersion=0.03,
                put_call_trend=-0.02,
                credit_spread_proxy=0.01,
            ),
            "bull_quiet",
        )
        self.assertEqual(
            classifier.classify(
                vix=38.0,
                vix_trend=0.20,
                spy_20d_momentum=-0.08,
                breadth=0.18,
                sector_dispersion=0.12,
                put_call_trend=0.15,
                credit_spread_proxy=-0.03,
            ),
            "crisis",
        )

    def test_uncertainty_model_blocks_high_disagreement(self) -> None:
        estimate = UncertaintyModel(get_config()).estimate([0.10, 0.85, 0.25, 0.90])

        self.assertFalse(estimate.allow_pick)
        self.assertEqual(estimate.confidence_label, "low")

    def test_feature_health_tracker_marks_weak_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_config()
            db_path = Path(tmpdir) / "paper.sqlite3"
            config.paper_trade_db = db_path
            config.feature_health_path = Path(tmpdir) / "feature_health.json"
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE paper_predictions (
                        run_id TEXT,
                        ticker TEXT,
                        created_at TEXT,
                        entry_price REAL,
                        target_price REAL,
                        final_score REAL,
                        payload_json TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE paper_evaluations (
                        run_id TEXT,
                        ticker TEXT,
                        latest_price REAL,
                        realized_return REAL,
                        hit_target INTEGER,
                        window_high_price REAL,
                        resolved_target_hit INTEGER,
                        resolution_method TEXT
                    )
                    """
                )
                payload = {
                    "ml_score": 70.0,
                    "technical_score": 65.0,
                    "volume_momentum_score": 72.0,
                    "options_score": 20.0,
                    "sentiment_score": 25.0,
                    "rs_score": 63.0,
                    "pattern_score": 0.0,
                    "diagnostics": {
                        "subscores": {
                            "technical_signals": {
                                "rsi": 72.0,
                                "volume": 75.0,
                            }
                        }
                    },
                }
                for offset, hit in enumerate([0, 0, 0, 0], start=1):
                    created_at = f"2026-02-{offset * 7:02d}T00:00:00+00:00"
                    conn.execute(
                        "INSERT INTO paper_predictions VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (f"run{offset}", "AAA", created_at, 10.0, 10.4, 70.0, json.dumps(payload)),
                    )
                    conn.execute(
                        "INSERT INTO paper_evaluations VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (f"run{offset}", "AAA", 10.1, 0.01, hit, 10.2, hit, "window_high"),
                    )
                conn.commit()

            tracker = FeatureHealthTracker(config)
            records = tracker.refresh()

            self.assertIn("rsi", records)
            self.assertTrue(records["rsi"].disabled)
            self.assertEqual(records["rsi"].status, "DISABLED")

    def test_adaptive_predictor_uses_regime_ensemble_and_online_blend(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_config()
            model_dir = Path(tmpdir) / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            config.model_dir = model_dir
            config.adaptive_metadata_path = model_dir / "adaptive.json"
            config.online_learner_path = model_dir / "online.pkl"
            config.feature_health_path = model_dir / "feature_health.json"
            predictor = AdaptivePredictor(config)
            predictor.regime_ensembles = {"neutral": [_ConstantModel(0.70), _ConstantModel(0.80)]}
            predictor.regime_win_rates = {"neutral": 0.62}
            predictor.active_regime = "neutral"
            predictor.feature_engine._prepare_inference_frame = lambda frame, **kwargs: pd.DataFrame(  # type: ignore[method-assign]
                [{column: 0.0 for column in FEATURE_COLUMNS}]
            )
            predictor.online_learner.predict_proba = lambda regime, features: 0.60  # type: ignore[method-assign]
            predictor.active_market_snapshot = predictor.regime_classifier.build_snapshot(
                price_frames={"SPY": make_ohlcv(periods=90, freq="D")},
                benchmark_history=make_ohlcv(periods=90, freq="D"),
            )

            result = predictor.predict_proba(make_ohlcv(periods=90, freq="D"), breadth_percentile=0.55)

        self.assertGreater(result.probability, 0.70)
        self.assertEqual(result.regime, "neutral")
        self.assertIn(result.confidence_label, {"medium", "high"})

    def test_adaptive_cv_and_backtest_use_smaller_ensembles(self) -> None:
        predictor = AdaptivePredictor(get_config())
        predictor.config.adaptive_regime_min_samples = 1
        dataset = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-06", periods=320, freq="W-MON", tz="UTC"),
                "label_end_date": pd.date_range("2020-01-13", periods=320, freq="W-MON", tz="UTC"),
                "target": [0, 1] * 160,
                "future_return": [0.01, 0.05] * 160,
                "regime": ["neutral"] * 320,
                **{column: [0.1] * 320 for column in predictor.feature_columns},
            }
        )
        calls: list[int] = []

        def fake_fit_regime_models(frame, *, model_count=None):
            calls.append(int(model_count or 0))
            return {"neutral": [_ConstantModel(0.6)]}

        predictor._fit_regime_models = fake_fit_regime_models  # type: ignore[method-assign]
        predictor._predict_regime_probabilities = lambda frame, models: pd.Series([0.6] * len(frame)).to_numpy()  # type: ignore[method-assign]
        predictor.feature_engine._evaluate_trade_basket = lambda frame, probs: {  # type: ignore[method-assign]
            "trade_win_rate": 0.5,
            "trade_average_return": 0.02,
        }
        predictor.feature_engine.build_training_frame = lambda *args, **kwargs: dataset.copy()  # type: ignore[method-assign]
        predictor._augment_dataset = lambda frame, **kwargs: frame.copy()  # type: ignore[method-assign]

        predictor._walk_forward_cv(dataset)
        predictor.walk_forward_backtest(
            {"AAA": make_ohlcv(periods=260, freq="D")},
            benchmark_history=make_ohlcv(periods=260, freq="D"),
            weeks=4,
        )

        self.assertIn(predictor.config.adaptive_cv_models, calls)
        self.assertIn(predictor.config.adaptive_backtest_models, calls)

    def test_fetch_macro_history_skips_blank_optional_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_config()
            cache = SQLiteCache(Path(tmpdir) / "cache.sqlite3")
            fetcher = MarketDataFetcher(config, cache)
            result = fetcher.fetch_macro_history("")

        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
