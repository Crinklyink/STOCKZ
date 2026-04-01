from __future__ import annotations

import unittest

from stock_predictor.models.anomaly_model import AnomalyDetector
from tests.helpers import make_ohlcv


class AnomalyModelTests(unittest.TestCase):
    def test_fit_and_score_return_structured_result(self) -> None:
        detector = AnomalyDetector()
        training = {
            "AAA": make_ohlcv(periods=420, freq="D", drift=0.002, volatility=0.005),
            "BBB": make_ohlcv(periods=420, freq="D", drift=0.0025, volatility=0.005),
            "CCC": make_ohlcv(periods=420, freq="D", drift=0.0018, volatility=0.005),
            "DDD": make_ohlcv(periods=420, freq="D", drift=0.0022, volatility=0.005),
        }
        detector.fit(training)
        frame = make_ohlcv(periods=420, freq="D", drift=0.003, last_boost=0.10, volatility=0.005)
        frame.loc[frame.index[-1], "volume"] = frame["volume"].iloc[:-1].mean() * 4.0
        result = detector.score(frame)
        self.assertIsNotNone(detector.model)
        self.assertGreaterEqual(result.score, 0.0)
        self.assertIn(result.adjustment_multiplier, {0.9, 1.0, 1.15})


if __name__ == "__main__":
    unittest.main()
