from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from stock_predictor.config import get_config
from stock_predictor.data.cache import SQLiteCache
from stock_predictor.data.fetcher import MarketDataFetcher


class FetcherCacheTests(unittest.TestCase):
    def test_postprocess_history_restores_datetime_index_from_cached_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_config()
            cache = SQLiteCache(Path(tmpdir) / "cache.sqlite3")
            fetcher = MarketDataFetcher(config, cache)
            cached = pd.DataFrame(
                [
                    {
                        "Date": "2026-03-19T00:00:00+00:00",
                        "Open": 100.0,
                        "High": 101.0,
                        "Low": 99.5,
                        "Close": 100.5,
                        "Volume": 1_000_000,
                    },
                    {
                        "Date": "2026-03-20T00:00:00+00:00",
                        "Open": 101.0,
                        "High": 102.0,
                        "Low": 100.0,
                        "Close": 101.5,
                        "Volume": 1_200_000,
                    },
                ]
            )

            result = fetcher._postprocess_history(cached)

            self.assertIsInstance(result.index, pd.DatetimeIndex)
            self.assertEqual(str(result.index.tz), "UTC")
            self.assertListEqual(list(result.columns), ["open", "high", "low", "close", "volume"])
            self.assertEqual(result.index[-1].isoformat(), "2026-03-20T00:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
