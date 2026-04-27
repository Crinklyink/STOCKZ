from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from stock_predictor.output.backtest import BacktestTracker, SignalAttributionTracker
from tests.helpers import make_candidate


class BacktestTrackerTests(unittest.TestCase):
    def _insert_paper_prediction(self, db_path: Path, *, run_id: str, ticker: str, created_at: datetime, target_price: float = 104.0) -> None:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO paper_predictions
                (run_id, ticker, created_at, entry_price, target_price, final_score, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, ticker, created_at.isoformat(), 100.0, target_price, 72.0, "{}"),
            )
            conn.commit()

    def _price_path(self, created_at: datetime, highs: list[float], closes: list[float] | None = None) -> pd.DataFrame:
        closes = closes or highs
        index = pd.date_range(
            start=created_at.replace(hour=0, minute=0, second=0, microsecond=0),
            periods=len(highs),
            freq="D",
            tz="UTC",
        )
        return pd.DataFrame(
            {
                "open": closes,
                "high": highs,
                "low": [min(high, close) - 1.0 for high, close in zip(highs, closes)],
                "close": closes,
                "volume": [1_000_000] * len(highs),
            },
            index=index,
        )

    def test_paper_trade_results_summary_uses_stated_target_not_positive_finish(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "paper.db"
            tracker = BacktestTracker(db_path)
            now = datetime.now(timezone.utc) - timedelta(days=8)
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO paper_predictions
                    (run_id, ticker, created_at, entry_price, target_price, final_score, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("run-1", "POS", now.isoformat(), 100.0, 105.0, 72.0, "{}"),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO paper_evaluations
                    (run_id, ticker, evaluated_at, latest_price, realized_return, hit_target)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    ("run-1", "POS", now.isoformat(), 101.0, 0.01, 0),
                )
                conn.commit()

            summary = tracker.paper_trade_results_summary()

        self.assertEqual(float(summary["target_hit_rate"]), 0.0)
        self.assertEqual(float(summary["positive_return_rate"]), 100.0)
        self.assertEqual(float(summary["win_rate"]), 0.0)

    def test_path_aware_evaluation_counts_intrawindow_touch_as_target_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "paper.db"
            tracker = BacktestTracker(db_path)
            created_at = datetime.now(timezone.utc) - timedelta(days=8)
            self._insert_paper_prediction(db_path, run_id="run-touch", ticker="TOUCH", created_at=created_at)

            tracker.evaluate_due_paper_predictions(
                {"TOUCH": 101.0},
                price_paths={"TOUCH": self._price_path(created_at, highs=[102.0, 105.0, 101.0], closes=[101.0, 102.0, 101.0])},
            )
            frame = tracker.paper_results_frame()

        row = frame.iloc[0]
        self.assertEqual(float(row["current_price"]), 101.0)
        self.assertEqual(float(row["window_high_price"]), 105.0)
        self.assertTrue(bool(row["resolved_target_hit"]))
        self.assertEqual(str(row["resolution_method"]), "window_high")

    def test_path_aware_evaluation_rejects_green_finish_without_touch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "paper.db"
            tracker = BacktestTracker(db_path)
            created_at = datetime.now(timezone.utc) - timedelta(days=8)
            self._insert_paper_prediction(db_path, run_id="run-green", ticker="GREEN", created_at=created_at)

            tracker.evaluate_due_paper_predictions(
                {"GREEN": 101.0},
                price_paths={"GREEN": self._price_path(created_at, highs=[101.5, 103.5, 102.0], closes=[100.5, 101.0, 101.0])},
            )
            frame = tracker.paper_results_frame()

        row = frame.iloc[0]
        self.assertFalse(bool(row["resolved_target_hit"]))
        self.assertEqual(str(row["resolution_method"]), "window_high")

    def test_path_aware_evaluation_counts_red_finish_after_touch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "paper.db"
            tracker = BacktestTracker(db_path)
            created_at = datetime.now(timezone.utc) - timedelta(days=8)
            self._insert_paper_prediction(db_path, run_id="run-red", ticker="RED", created_at=created_at)

            tracker.evaluate_due_paper_predictions(
                {"RED": 97.0},
                price_paths={"RED": self._price_path(created_at, highs=[104.5, 105.0, 99.0], closes=[103.0, 100.0, 97.0])},
            )
            frame = tracker.paper_results_frame()

        row = frame.iloc[0]
        self.assertLess(float(row["current_price"]), 100.0)
        self.assertTrue(bool(row["resolved_target_hit"]))
        self.assertEqual(str(row["resolution_method"]), "window_high")

    def test_path_aware_evaluation_falls_back_deterministically_when_high_data_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "paper.db"
            tracker = BacktestTracker(db_path)
            created_at = datetime.now(timezone.utc) - timedelta(days=8)
            self._insert_paper_prediction(db_path, run_id="run-fallback", ticker="FALL", created_at=created_at)

            tracker.evaluate_due_paper_predictions({"FALL": 105.0}, price_paths={})
            frame = tracker.paper_results_frame()

        row = frame.iloc[0]
        self.assertTrue(bool(row["resolved_target_hit"]))
        self.assertEqual(str(row["resolution_method"]), "latest_price_fallback")

    def test_threshold_recommendation_uses_completed_paper_trades(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "paper.db"
            tracker = BacktestTracker(db_path)
            now = datetime.now(timezone.utc)
            with sqlite3.connect(db_path) as conn:
                for offset in range(4):
                    created_at = (now - timedelta(days=7 * (offset + 1))).isoformat()
                    week_label = f"run-{offset}"
                    for slot in range(5):
                        high_ticker = f"HIGH{offset}{slot}"
                        low_ticker = f"LOW{offset}{slot}"
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO paper_predictions
                            (run_id, ticker, created_at, entry_price, target_price, final_score, payload_json)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (week_label, high_ticker, created_at, 100.0, 104.0, 72.0, "{}"),
                        )
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO paper_predictions
                            (run_id, ticker, created_at, entry_price, target_price, final_score, payload_json)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (week_label, low_ticker, created_at, 100.0, 104.0, 54.0, "{}"),
                        )
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO paper_evaluations
                            (run_id, ticker, evaluated_at, latest_price, realized_return, hit_target)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (week_label, high_ticker, created_at, 106.0, 0.06, 1),
                        )
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO paper_evaluations
                            (run_id, ticker, evaluated_at, latest_price, realized_return, hit_target)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (week_label, low_ticker, created_at, 99.0, -0.01, 0),
                        )
                conn.commit()

            recommendation = tracker.threshold_recommendation(min_weeks=4)

        self.assertIsNotNone(recommendation)
        self.assertGreater(float(recommendation["threshold"]), 54.0)
        self.assertGreaterEqual(float(recommendation["win_rate"]), 90.0)

    def test_regime_memory_summary_matches_bucket_and_trend(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "paper.db"
            tracker = BacktestTracker(db_path)
            now = datetime.now(timezone.utc) - timedelta(days=8)
            candidate = make_candidate("XOM", sector="Energy")
            tracker.record_paper_predictions("run-energy", [candidate])
            tracker.record_regime_context(
                "run-energy",
                vix=27.0,
                spy_week_return=-0.03,
                sector_leaders=["Energy", "Healthcare"],
            )
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "UPDATE paper_predictions SET created_at = ? WHERE run_id = ?",
                    (now.isoformat(), "run-energy"),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO paper_evaluations
                    (run_id, ticker, evaluated_at, latest_price, realized_return, hit_target)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    ("run-energy", "XOM", now.isoformat(), 106.0, 0.06, 1),
                )
                conn.commit()

            summary = tracker.regime_memory_summary(
                vix=26.0,
                spy_week_return=-0.025,
                sector_leaders=["Energy", "Utilities"],
            )

        self.assertEqual(summary["vix_bucket"], "25-30")
        self.assertEqual(summary["spy_trend"], "falling")
        self.assertGreaterEqual(float(summary["win_rate"]), 100.0)


class SignalAttributionTrackerTests(unittest.TestCase):
    def test_signal_summary_reports_recent_signal_win_rate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "signals.db"
            tracker = SignalAttributionTracker(db_path)
            candidate = make_candidate("TEST")
            tracker.record_predictions("run-1", [candidate])
            old_timestamp = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
            with sqlite3.connect(db_path) as conn:
                conn.execute("UPDATE signal_outcomes SET created_at = ?", (old_timestamp,))
                conn.commit()

            tracker.evaluate_due_predictions({"TEST": 106.0})
            summary = tracker.signal_summary(weeks=4)

        self.assertTrue(summary)
        self.assertGreaterEqual(float(summary["best_signal_win_rate"]), 100.0)
        self.assertGreater(len(summary["rows"]), 0)


if __name__ == "__main__":
    unittest.main()
