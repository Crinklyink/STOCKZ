from __future__ import annotations

import unittest
from unittest.mock import patch

from stock_predictor.analysis.pattern_history import PatternWinRateAnalyzer


class PatternHistoryTests(unittest.TestCase):
    def test_pattern_history_reports_win_rate_and_qualification(self) -> None:
        price_frames = {"AAA": object(), "BBB": object()}
        analyzer = PatternWinRateAnalyzer(lookback_occurrences=20, future_days=5, step=5)

        with patch(
            "stock_predictor.analysis.pattern_history.PatternWinRateAnalyzer._collect_outcomes",
            side_effect=[
                [True] * 16 + [False] * 4,
            ],
        ):
            result = analyzer.analyze(
                "bull_flag",
                "Technology",
                {"Technology": ["AAA", "BBB"]},
                price_frames,
                current_regime="risk_on",
                pattern_threshold=7.0,
            )

        self.assertGreaterEqual(result.sample_size, 10)
        self.assertGreaterEqual(result.win_rate, 0.60)
        self.assertTrue(result.qualified)
        self.assertIn("win rate", result.label.lower())


if __name__ == "__main__":
    unittest.main()
