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


if __name__ == "__main__":
    unittest.main()
