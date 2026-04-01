from __future__ import annotations

import unittest

from stock_predictor.analysis.smart_money import evaluate_smart_money_divergence


class SmartMoneyTests(unittest.TestCase):
    def test_institutions_bullish_retail_bearish_is_strongest_signal(self) -> None:
        result = evaluate_smart_money_divergence(0.30, 0.75, 0.80)
        self.assertFalse(result.reject)
        self.assertEqual(result.label, "strongest")
        self.assertGreater(result.adjustment_points, 0)

    def test_retail_bullish_but_institutions_weak_is_rejected(self) -> None:
        result = evaluate_smart_money_divergence(0.80, 0.20, 0.25, bearish_flow_ratio=0.3)
        self.assertTrue(result.reject)
        self.assertEqual(result.label, "reject")
        self.assertLess(result.adjustment_points, 0)


if __name__ == "__main__":
    unittest.main()
