from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from stock_predictor.config import get_config
from stock_predictor.data.cache import SQLiteCache
from stock_predictor.data.fetcher import MarketDataFetcher


class _MockResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class UniverseLoaderTests(unittest.TestCase):
    def test_cached_universe_table_selects_first_table_with_ticker_column(self) -> None:
        html = """
        <html>
          <body>
            <table>
              <tr><th>Name</th><th>Value</th></tr>
              <tr><td>Noise</td><td>1</td></tr>
            </table>
            <table>
              <tr><th>Ticker</th><th>ICB Industry[14]</th></tr>
              <tr><td>BRK.B</td><td>Financials</td></tr>
              <tr><td>MSFT</td><td>Technology</td></tr>
            </table>
          </body>
        </html>
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_config()
            cache = SQLiteCache(Path(tmpdir) / "cache.sqlite3")
            fetcher = MarketDataFetcher(config, cache)
            with patch("stock_predictor.data.fetcher.requests.get", return_value=_MockResponse(html)):
                result = fetcher._cached_universe_table("nasdaq100-universe", "https://example.com")

        self.assertEqual(sorted(result["Financials"]), ["BRK-B"])
        self.assertEqual(sorted(result["Technology"]), ["MSFT"])


if __name__ == "__main__":
    unittest.main()
