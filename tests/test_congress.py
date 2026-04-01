from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from stock_predictor.config import AppConfig
from stock_predictor.data.cache import SQLiteCache
from stock_predictor.data.congress import CongressTradeTracker


class CongressTrackerTests(unittest.TestCase):
    def test_fetch_recent_trades_uses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recent_date = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
            config = AppConfig(
                quiver_endpoint="https://example.com/quiver",
                quiver_token="token",
            )
            cache = SQLiteCache(Path(tmpdir) / "cache.sqlite3")
            tracker = CongressTradeTracker(config, cache)
            response = SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: [{"ticker": "NVDA", "transactionDate": recent_date, "type": "Purchase"}],
            )
            with patch("stock_predictor.data.congress.requests.get", return_value=response) as mocked_get:
                first = tracker.fetch_recent_trades()
                second = tracker.fetch_recent_trades()
            self.assertEqual(len(first), 1)
            self.assertEqual(first, second)
            self.assertEqual(mocked_get.call_count, 1)

    def test_score_ticker_counts_recent_buys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = CongressTradeTracker(AppConfig(), SQLiteCache(Path(tmpdir) / "cache.sqlite3"))
            recent_1 = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
            recent_2 = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
            recent_3 = (datetime.now(timezone.utc) - timedelta(days=6)).isoformat()
            with patch.object(
                tracker,
                "fetch_recent_trades",
                return_value=[
                    {"ticker": "NVDA", "transactionDate": recent_1, "type": "Purchase"},
                    {"ticker": "NVDA", "transactionDate": recent_2, "type": "Purchase"},
                    {"ticker": "NVDA", "transactionDate": recent_3, "type": "Sale"},
                ],
            ):
                result = tracker.score_ticker("NVDA")
        self.assertEqual(result.recent_buys, 2)
        self.assertEqual(result.recent_sells, 1)
        self.assertGreater(result.score, 0.5)


if __name__ == "__main__":
    unittest.main()
