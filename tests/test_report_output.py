from __future__ import annotations

import unittest

from stock_predictor.output.report import render_terminal_report
from tests.helpers import make_candidate


class ReportOutputTests(unittest.TestCase):
    def test_terminal_report_uses_above_threshold_wording(self) -> None:
        candidate = make_candidate("APA")
        report = render_terminal_report(
            [candidate],
            threshold_used=60.0,
            regime_label="neutral",
            qualified_count=1,
            candidate_pool_size=10,
            summary_header={
                "date": "2026-03-22",
                "universe_total": 10,
                "runtime_seconds": 12.0,
                "regime": "neutral",
                "vix": 21.5,
                "spy_week_return": -0.8,
                "model_family": "XGB+LGBM",
                "model_training_samples": 5000,
                "model_auc": 0.621,
                "threshold_used": 60.0,
                "top_sector": "Energy",
                "worst_sector": "Real Estate",
                "stage1_survivors": 5,
                "stage1_runtime_seconds": 0.8,
                "cache_warm": 10,
                "cache_total": 10,
                "cache_hit_rate": 100.0,
                "cache_saved_seconds_estimate": 240.0,
            },
        )

        self.assertIn("Above score threshold: 1/10", report)
        self.assertIn("ABOVE SCORE THRESHOLD", report)
        self.assertNotIn("Qualified above threshold", report)
        self.assertNotIn("QUALIFIED", report)


if __name__ == "__main__":
    unittest.main()
