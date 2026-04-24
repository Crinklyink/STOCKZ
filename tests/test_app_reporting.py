from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
import json

from app_window import AppDataService


class AppReportingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = AppDataService(Path("/Users/crinklyink/Desktop/idk project"))

    def test_model_metadata_merges_active_ensemble_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            adaptive_path = tmp / "adaptive.json"
            xgb_path = tmp / "xgb.json"
            lgbm_path = tmp / "lgbm.json"
            adaptive_path.write_text(
                json.dumps(
                    {
                        "trained": True,
                        "trained_at": "2026-04-12T00:00:00+00:00",
                        "training_samples": 6400,
                        "ensemble_auc": 0.631,
                        "selected_profile": "adaptive_regime_ensemble",
                        "regime_summary": {"neutral": {"weeks": 6, "win_rate": 58.0, "average_return": 2.1}},
                    }
                ),
                encoding="utf-8",
            )
            xgb_path.write_text(
                json.dumps(
                    {
                        "trained": True,
                        "trained_at": "2026-04-10T02:15:26.086424+00:00",
                        "training_samples": 53683,
                        "auc": 0.6247615384615385,
                        "ensemble_auc": 0.6247615384615385,
                        "xgb_auc": 0.6242461538461538,
                        "selected_profile": "regularized_ensemble",
                    }
                ),
                encoding="utf-8",
            )
            lgbm_path.write_text(
                json.dumps(
                    {
                        "auc": 0.616376923076923,
                        "ensemble_weights": {"xgb": 0.6, "lgbm": 0.4},
                    }
                ),
                encoding="utf-8",
            )

            original_adaptive = self.service.config.adaptive_metadata_path
            original_xgb = self.service.config.xgb_metadata_path
            original_lgbm = self.service.config.lgbm_metadata_path
            self.service.config.adaptive_metadata_path = adaptive_path
            self.service.config.xgb_metadata_path = xgb_path
            self.service.config.lgbm_metadata_path = lgbm_path
            try:
                meta = self.service._load_model_metadata({"training_report": {}})
            finally:
                self.service.config.adaptive_metadata_path = original_adaptive
                self.service.config.xgb_metadata_path = original_xgb
                self.service.config.lgbm_metadata_path = original_lgbm

            self.assertEqual(meta["model_stack"], "XGBoost + LightGBM")
            self.assertEqual(meta["selected_profile"], "adaptive_regime_ensemble")
            self.assertEqual(meta["profile_label"], "adaptive regime ensemble")
            self.assertAlmostEqual(float(meta["auc"]), 0.631)
            self.assertAlmostEqual(float(meta["lightgbm_auc"]), 0.616376923076923)
            self.assertEqual(meta["ensemble_weights"], {"xgb": 0.6, "lgbm": 0.4})

    def test_weekly_rows_use_target_hits_not_positive_returns(self) -> None:
        details = [
            {
                "week_key": "2026-03-16 00:00:00",
                "week_label": "Mar 22",
                "ticker": "AAA",
                "realized_return_pct": 1.5,
                "hit_target": False,
            },
            {
                "week_key": "2026-03-16 00:00:00",
                "week_label": "Mar 22",
                "ticker": "BBB",
                "realized_return_pct": -2.0,
                "hit_target": False,
            },
        ]

        weekly_rows = self.service._build_weekly_rows(details)

        self.assertEqual(len(weekly_rows), 1)
        row = weekly_rows[0]
        self.assertEqual(row["target_hits"], 0)
        self.assertEqual(row["target_hit_rate"], 0.0)
        self.assertEqual(row["positive_return_rate"], 50.0)
        self.assertEqual(row["winners"], 0)
        self.assertEqual(row["hit_rate"], 0.0)

    def test_rolling_summary_separates_target_hit_rate_from_positive_return_rate(self) -> None:
        details = [
            {
                "week_key": "2026-03-16 00:00:00",
                "week_label": "Mar 22",
                "ticker": "AAA",
                "realized_return_pct": 1.5,
                "hit_target": False,
            },
            {
                "week_key": "2026-03-16 00:00:00",
                "week_label": "Mar 22",
                "ticker": "BBB",
                "realized_return_pct": 4.8,
                "hit_target": True,
            },
            {
                "week_key": "2026-03-09 00:00:00",
                "week_label": "Mar 15",
                "ticker": "CCC",
                "realized_return_pct": -1.0,
                "hit_target": False,
            },
        ]

        weekly_rows = self.service._build_weekly_rows(details)
        summary = self.service._build_rolling_summary(details, weekly_rows)

        self.assertAlmostEqual(float(summary["target_hit_rate"]), 33.3333, places=2)
        self.assertAlmostEqual(float(summary["positive_return_rate"]), 66.6667, places=2)
        self.assertAlmostEqual(float(summary["win_rate"]), float(summary["target_hit_rate"]), places=4)

    def test_history_rows_are_grouped_by_run_not_calendar_week(self) -> None:
        details = [
            {
                "run_id": "run-a",
                "run_label": "Mar 25 · 4:01PM",
                "created_ts": "2026-03-25T20:01:00Z",
                "week_key": "2026-03-23 00:00:00",
                "week_label": "Mar 25",
                "ticker": "DOCN",
                "realized_return_pct": 4.2,
                "hit_target": True,
                "final_score": 82.0,
            },
            {
                "run_id": "run-b",
                "run_label": "Mar 25 · 4:26PM",
                "created_ts": "2026-03-25T20:26:00Z",
                "week_key": "2026-03-23 00:00:00",
                "week_label": "Mar 25",
                "ticker": "ARM",
                "realized_return_pct": None,
                "hit_target": False,
                "final_score": 79.0,
            },
        ]

        history_rows = self.service._build_history_rows(details)

        self.assertEqual(len(history_rows), 2)
        self.assertEqual(history_rows[0]["run_id"], "run-b")
        self.assertEqual(history_rows[1]["run_id"], "run-a")
        self.assertEqual(history_rows[0]["target_hit_rate_label"], "pending")
        self.assertEqual(history_rows[1]["target_hit_rate_label"], "100%")

    def test_weekly_rows_use_latest_run_per_week_for_tracking(self) -> None:
        details = [
            {
                "run_id": "run-a",
                "created_ts": "2026-03-25T20:01:00Z",
                "week_key": "2026-03-23 00:00:00",
                "week_label": "Mar 25",
                "ticker": "DOCN",
                "realized_return_pct": -2.0,
                "hit_target": False,
            },
            {
                "run_id": "run-b",
                "created_ts": "2026-03-25T20:26:00Z",
                "week_key": "2026-03-23 00:00:00",
                "week_label": "Mar 25",
                "ticker": "ARM",
                "realized_return_pct": 4.8,
                "hit_target": True,
            },
        ]

        weekly_rows = self.service._build_weekly_rows(details)

        self.assertEqual(len(weekly_rows), 1)
        self.assertEqual(weekly_rows[0]["target_hits"], 1)
        self.assertEqual(weekly_rows[0]["target_hit_rate_label"], "100%")
        self.assertEqual(weekly_rows[0]["best_pick"], "ARM +4.8%")


if __name__ == "__main__":
    unittest.main()
