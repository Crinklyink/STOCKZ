from __future__ import annotations

import unittest

from stock_predictor.output.html_report import render_weekly_html_report


class HtmlReportTests(unittest.TestCase):
    def test_render_weekly_html_report_shows_watchlist_when_no_official_picks(self) -> None:
        payload = {
            "scan_summary": {
                "date": "2026-03-22",
                "regime": "neutral",
                "vix": 26.8,
                "spy_week_return": -2.1,
                "selection_warning": "High VIX regime active.",
                "qualified_count": 0,
            },
            "macro_summary": "Market breadth is weak.",
            "selected": [],
            "display_candidates": [
                {
                    "ticker": "APA",
                    "company_name": "APA Corporation",
                    "sector": "Energy",
                    "sector_temperature_tag": "🔥 HOT SECTOR",
                    "current_price": 39.11,
                    "stop_loss": 36.96,
                    "targets": {"tp2": 41.26},
                    "technical_score": 73.0,
                    "rs_score": 89.0,
                    "volume_momentum_score": 100.0,
                    "ml_score": 69.0,
                    "pattern_score": 8.0,
                    "probability_4pct_5d": 69.0,
                    "confluence_count": 5,
                }
            ],
        }

        html = render_weekly_html_report(
            payload,
            last_week_summary={
                "picks": 3,
                "target_hits": 1,
                "target_hit_rate": 33.0,
                "positive_return_rate": 67.0,
                "avg_return": 1.2,
                "best_text": "APA +4.8% (target hit)",
                "worst_text": "T -1.2% (below target)",
            },
            rolling_summary={
                "target_hit_rate": 42.0,
                "positive_return_rate": 58.0,
                "average_return": 0.7,
            },
            weekly_rows=[
                {"week_label": "03/22", "target_hit_rate": 33.0},
            ],
        )

        self.assertIn("No official picks this week", html)
        self.assertIn("APA", html)
        self.assertIn("Market breadth is weak.", html)
        self.assertIn("Above score threshold", html)
        self.assertIn("Target Hit Rate", html)
        self.assertIn("Positive Return Rate", html)


if __name__ == "__main__":
    unittest.main()
