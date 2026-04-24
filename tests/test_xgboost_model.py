from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from stock_predictor.config import get_config
from stock_predictor.models.xgboost_model import FEATURE_COLUMNS, XGBoostPredictor
from tests.helpers import make_ohlcv


class _FakeModel:
    def __init__(self) -> None:
        self.received = None

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        self.received = features
        return np.array([[0.2, 0.8]])


class _ConstantProbModel:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.column_stack([np.full(len(features), 1 - self.probability), np.full(len(features), self.probability)])


class XGBoostPredictorTests(unittest.TestCase):
    def test_predict_proba_uses_named_feature_frame_for_trained_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_config()
            model_dir = Path(tmpdir) / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            config.model_dir = model_dir
            config.xgb_model_path = model_dir / "xgboost_model.pkl"
            config.xgb_metadata_path = model_dir / "xgboost_metadata.json"
            predictor = XGBoostPredictor(config)
            predictor.model = _FakeModel()
            predictor.enabled = True
            predictor.feature_columns = ["return_1", "rsi"]

            predictor._prepare_inference_frame = lambda frame, **kwargs: pd.DataFrame(  # type: ignore[method-assign]
                [{"return_1": 0.12, "rsi": 61.0}]
            )

            result = predictor.predict_proba(pd.DataFrame({"close": [1.0]}))

        self.assertEqual(result.status, "trained")
        self.assertEqual(list(predictor.model.received.columns), ["return_1", "rsi"])

    def test_predict_proba_uses_saved_blend_weights(self) -> None:
        predictor = XGBoostPredictor(get_config())
        predictor.model = _ConstantProbModel(0.80)
        predictor.lightgbm_model = _ConstantProbModel(0.20)
        predictor.model_calibrator = None
        predictor.lightgbm_calibrator = None
        predictor.enabled = True
        predictor.feature_columns = ["return_1", "rsi"]
        predictor.blend_weights = {"xgb": 0.9, "lgbm": 0.1}
        predictor.last_report = None

        predictor._prepare_inference_frame = lambda frame, **kwargs: pd.DataFrame(  # type: ignore[method-assign]
            [{"return_1": 0.05, "rsi": 58.0}]
        )

        result = predictor.predict_proba(pd.DataFrame({"close": [1.0]}))

        self.assertAlmostEqual(result.probability, 0.74, places=4)
        self.assertEqual(result.blend_weights, {"xgb": 0.9, "lgbm": 0.1})

    def test_build_training_frame_spreads_samples_across_history(self) -> None:
        predictor = XGBoostPredictor(get_config())
        daily = make_ohlcv(periods=520, freq="D", start_price=50.0, drift=0.0005, volatility=0.01)
        benchmark = make_ohlcv(periods=520, freq="D", start_price=400.0, drift=0.0002, volatility=0.007)

        frame = predictor.build_training_frame(
            {"ABC": daily},
            max_samples_per_ticker=10,
            benchmark_history=benchmark,
        )

        self.assertGreaterEqual(len(frame), 8)
        self.assertLessEqual(len(frame), 10)
        dates = pd.to_datetime(frame["date"], utc=True)
        self.assertTrue((dates.dt.dayofweek == 0).all())
        self.assertGreater((dates.max() - dates.min()).days, 60)
        label_starts = pd.to_datetime(frame["label_start_date"], utc=True)
        label_ends = pd.to_datetime(frame["label_end_date"], utc=True)
        self.assertTrue((label_starts.dt.dayofweek == 4).all())
        self.assertTrue((label_ends.dt.dayofweek == 4).all())
        self.assertNotIn("day_of_week", FEATURE_COLUMNS)

    def test_purge_train_rows_applies_business_day_embargo(self) -> None:
        predictor = XGBoostPredictor(get_config())
        frame = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA", "AAA"],
                "date": pd.to_datetime(["2026-01-05", "2026-01-12", "2026-01-19"], utc=True),
                "label_end_date": pd.to_datetime(["2026-01-09", "2026-01-16", "2026-01-23"], utc=True),
                "target": [1, 0, 1],
            }
        )

        purged = predictor._purge_train_rows(frame, test_start=pd.Timestamp("2026-01-19", tz="UTC"))

        self.assertEqual(purged["date"].dt.strftime("%Y-%m-%d").tolist(), ["2026-01-05"])

    def test_prepare_inference_frame_appends_synthetic_next_bar(self) -> None:
        predictor = XGBoostPredictor(get_config())
        captured = {}

        def fake_prepare(frame: pd.DataFrame, **kwargs) -> pd.DataFrame:
            captured["index"] = frame.index
            return pd.DataFrame([{"return_1": 0.1, "rsi": 55.0}])

        predictor._prepare_frame = fake_prepare  # type: ignore[method-assign]
        frame = make_ohlcv(periods=5, freq="D")

        predictor._prepare_inference_frame(frame)

        self.assertEqual(len(captured["index"]), len(frame) + 1)
        self.assertGreater(captured["index"][-1], captured["index"][-2])

    def test_blend_weights_strongly_favor_clear_auc_winner(self) -> None:
        predictor = XGBoostPredictor(get_config())

        self.assertEqual(
            predictor._blend_weights_from_aucs(0.64, 0.60, lightgbm_available=True),
            {"xgb": 0.9, "lgbm": 0.1},
        )
        self.assertEqual(
            predictor._blend_weights_from_aucs(0.60, 0.63, lightgbm_available=True),
            {"xgb": 0.1, "lgbm": 0.9},
        )

    def test_blend_weights_use_moderate_tilt_for_small_auc_edge(self) -> None:
        predictor = XGBoostPredictor(get_config())

        self.assertEqual(
            predictor._blend_weights_from_aucs(0.615, 0.603, lightgbm_available=True),
            {"xgb": 0.8, "lgbm": 0.2},
        )
        self.assertEqual(
            predictor._blend_weights_from_aucs(0.604, 0.615, lightgbm_available=True),
            {"xgb": 0.2, "lgbm": 0.8},
        )

    def test_profile_blend_override_takes_precedence_when_available(self) -> None:
        predictor = XGBoostPredictor(get_config())
        profile = {"blend_weights": {"xgb": 0.85, "lgbm": 0.15}}

        self.assertEqual(
            predictor._resolve_blend_weights(profile, 0.62, 0.61, lightgbm_available=True),
            {"xgb": 0.85, "lgbm": 0.15},
        )
        self.assertEqual(
            predictor._resolve_blend_weights(profile, 0.62, 0.61, lightgbm_available=False),
            {"xgb": 1.0, "lgbm": 0.0},
        )

    def test_recency_sample_weights_favor_newer_rows(self) -> None:
        predictor = XGBoostPredictor(get_config())
        dates = pd.Series(pd.to_datetime(["2026-01-06", "2026-03-03", "2026-04-07"], utc=True))

        weights = predictor._recency_sample_weights(dates, len(dates))

        self.assertIsNotNone(weights)
        self.assertLess(weights[0], weights[1])
        self.assertLess(weights[1], weights[2])
        self.assertLessEqual(weights[2], 1.0)
        self.assertGreaterEqual(weights[0], predictor.config.training_recency_weight_floor)

    def test_fit_kwargs_include_recency_weights(self) -> None:
        predictor = XGBoostPredictor(get_config())
        dates = pd.Series(pd.to_datetime(["2026-01-06", "2026-03-03", "2026-04-07"], utc=True))
        target = pd.Series([0, 1, 1])

        kwargs = predictor._fit_kwargs(_FakeModel(), target, 4.0, sample_dates=dates)

        self.assertIn("sample_weight", kwargs)
        self.assertEqual(len(kwargs["sample_weight"]), 3)

    def test_profile_objective_rewards_recent_fold_strength(self) -> None:
        predictor = XGBoostPredictor(get_config())
        stale_profile = {
            "ensemble_auc": 0.620,
            "xgb_auc": 0.615,
            "rows": [
                {"auc": 0.70},
                {"auc": 0.69},
                {"auc": 0.52},
                {"auc": 0.50},
            ],
        }
        stronger_recent_profile = {
            "ensemble_auc": 0.618,
            "xgb_auc": 0.616,
            "rows": [
                {"auc": 0.60},
                {"auc": 0.61},
                {"auc": 0.64},
                {"auc": 0.66},
            ],
        }

        self.assertGreater(
            predictor._profile_objective(stronger_recent_profile),
            predictor._profile_objective(stale_profile),
        )

    def test_profile_objective_rewards_trade_quality(self) -> None:
        predictor = XGBoostPredictor(get_config())
        weaker_trade_profile = {
            "ensemble_auc": 0.626,
            "xgb_auc": 0.624,
            "trade_win_rate": 0.28,
            "trade_average_return": -0.008,
            "rows": [
                {"auc": 0.62},
                {"auc": 0.63},
                {"auc": 0.63},
                {"auc": 0.62},
            ],
        }
        stronger_trade_profile = {
            "ensemble_auc": 0.624,
            "xgb_auc": 0.623,
            "trade_win_rate": 0.40,
            "trade_average_return": 0.004,
            "rows": [
                {"auc": 0.62},
                {"auc": 0.63},
                {"auc": 0.62},
                {"auc": 0.63},
            ],
        }

        self.assertGreater(
            predictor._profile_objective(stronger_trade_profile),
            predictor._profile_objective(weaker_trade_profile),
        )

    def test_walk_forward_backtest_uses_active_profile_configuration(self) -> None:
        predictor = XGBoostPredictor(get_config())
        dataset = pd.DataFrame(
            {
                "ticker": ["AAA"] * 210,
                "date": pd.date_range("2024-01-01", periods=210, freq="7D", tz="UTC"),
                "label_end_date": pd.date_range("2024-01-05", periods=210, freq="7D", tz="UTC"),
                "target": ([0, 1] * 105),
                "future_return": np.linspace(-0.02, 0.04, 210),
                "future_excess_return": np.linspace(-0.03, 0.03, 210),
                "return_1": 0.1,
                "rsi": 55.0,
            }
        )
        for feature in FEATURE_COLUMNS:
            if feature not in dataset.columns:
                dataset[feature] = 0.0

        predictor.build_training_frame = lambda *args, **kwargs: dataset.copy()  # type: ignore[method-assign]
        predictor.active_profile_config = {
            "xgb_params": {"max_depth": 3},
            "lgbm_params": {"num_leaves": 31},
            "use_lightgbm": True,
            "blend_weights": {"xgb": 0.85, "lgbm": 0.15},
        }

        seen = {}

        class _FitModel(_ConstantProbModel):
            def fit(self, x, y, **kwargs):
                return self

        def fake_build_model(*, scale_pos_weight=1.0, overrides=None):
            seen["xgb_overrides"] = overrides
            return _FitModel(0.8)

        def fake_build_lightgbm_model(*, scale_pos_weight=1.0, overrides=None, enabled=True):
            if not enabled:
                return None
            seen["lgbm_overrides"] = overrides
            return _FitModel(0.2)

        predictor._build_model = fake_build_model  # type: ignore[method-assign]
        predictor._build_lightgbm_model = fake_build_lightgbm_model  # type: ignore[method-assign]
        predictor._fit_kwargs = lambda *args, **kwargs: {}  # type: ignore[method-assign]
        predictor._predict_model_probabilities = lambda model, features, calibrator: np.full(len(features), model.probability)  # type: ignore[method-assign]
        predictor._resolve_blend_weights = lambda profile, xgb_auc, lightgbm_auc, lightgbm_available: profile["blend_weights"]  # type: ignore[method-assign]

        result = predictor.walk_forward_backtest({"AAA": pd.DataFrame()}, weeks=2)

        self.assertEqual(seen["xgb_overrides"], {"max_depth": 3})
        self.assertEqual(seen["lgbm_overrides"], {"num_leaves": 31})
        self.assertIn("summary", result)

    def test_search_training_profile_selects_highest_objective(self) -> None:
        predictor = XGBoostPredictor(get_config())
        dataset = pd.DataFrame({"date": pd.to_datetime(["2026-01-05"], utc=True), "target": [1]})

        predictor._candidate_training_profiles = lambda: [  # type: ignore[method-assign]
            {"name": "baseline", "xgb_params": {}, "lgbm_params": {}, "use_lightgbm": True},
            {"name": "candidate_a", "xgb_params": {}, "lgbm_params": {}, "use_lightgbm": True},
            {"name": "candidate_b", "xgb_params": {}, "lgbm_params": {}, "use_lightgbm": False},
        ]

        def fake_cv(_dataset, *, profile=None):
            name = (profile or {}).get("name", "baseline")
            mapping = {
                "baseline": {
                    "ensemble_auc": 0.610,
                    "xgb_auc": 0.608,
                    "rows": [{"auc": 0.60}, {"auc": 0.61}, {"auc": 0.60}, {"auc": 0.61}],
                },
                "candidate_a": {
                    "ensemble_auc": 0.615,
                    "xgb_auc": 0.612,
                    "rows": [{"auc": 0.61}, {"auc": 0.61}, {"auc": 0.62}, {"auc": 0.61}],
                },
                "candidate_b": {
                    "ensemble_auc": 0.613,
                    "xgb_auc": 0.614,
                    "rows": [{"auc": 0.60}, {"auc": 0.62}, {"auc": 0.66}, {"auc": 0.67}],
                },
            }
            return mapping[name]

        predictor._walk_forward_cv = fake_cv  # type: ignore[method-assign]

        selected_profile, cv_result = predictor._search_training_profile(dataset)

        self.assertEqual(selected_profile["name"], "candidate_b")
        self.assertEqual(cv_result["ensemble_auc"], 0.613)


if __name__ == "__main__":
    unittest.main()
