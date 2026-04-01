from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stock_predictor.config import get_config
from stock_predictor.data.cache import SQLiteCache
from stock_predictor.data.dark_pool import FlowDetector


class _RaisingFetcher:
    def fetch_options_chain(self, ticker: str, fresh: bool = False):  # pragma: no cover - should not be called
        raise AssertionError("fetch_options_chain should not run for neutral fallback")


class DarkPoolTests(unittest.TestCase):
    def test_score_ticker_uses_neutral_fallback_without_option_providers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_config()
            config.unusual_whales_endpoint = None
            config.tradytics_endpoint = None
            cache = SQLiteCache(Path(tmpdir) / "cache.sqlite3")
            detector = FlowDetector(config, cache, _RaisingFetcher())

            score = detector.score_ticker("AAPL", 190.0, fresh=False)

        self.assertTrue(score["defaulted"])
        self.assertEqual(score["options_score"], config.default_missing_signal_value / 100.0)


if __name__ == "__main__":
    unittest.main()
