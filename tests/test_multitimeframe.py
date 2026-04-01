from __future__ import annotations

import unittest
from unittest.mock import patch

from stock_predictor.analysis.multitimeframe import evaluate_multi_timeframe_alignment
from tests.helpers import make_ohlcv


class MultiTimeframeTests(unittest.TestCase):
    def test_full_alignment_qualifies(self) -> None:
        daily = make_ohlcv(periods=260, freq="D", drift=0.0025, last_boost=0.03)
        hourly = make_ohlcv(periods=240, freq="h", drift=0.0010, last_boost=0.02)
        with patch(
            "stock_predictor.analysis.multitimeframe.timeframe_signal",
            side_effect=[0.72, 0.68, 0.74],
        ):
            result = evaluate_multi_timeframe_alignment(daily, hourly)
        self.assertTrue(result.qualifies)
        self.assertFalse(result.contradicts)
        self.assertEqual(result.agreement_count, 3)
        self.assertEqual(result.penalty_factor, 1.0)

    def test_contradiction_rejects(self) -> None:
        daily = make_ohlcv(periods=260, freq="D")
        hourly = make_ohlcv(periods=240, freq="h")
        with patch(
            "stock_predictor.analysis.multitimeframe.timeframe_signal",
            side_effect=[0.72, 0.30, 0.71],
        ):
            result = evaluate_multi_timeframe_alignment(daily, hourly)
        self.assertFalse(result.qualifies)
        self.assertTrue(result.contradicts)
        self.assertEqual(result.penalty_factor, 0.0)


if __name__ == "__main__":
    unittest.main()
