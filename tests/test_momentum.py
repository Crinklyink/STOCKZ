from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from stock_predictor.analysis.momentum import build_momentum_watchlist
from tests.helpers import make_ohlcv


class Bundle:
    def __init__(self, sector: str, daily):
        self.sector = sector
        self.daily = daily


class MomentumWatchlistTests(unittest.TestCase):
    def test_build_momentum_watchlist_marks_persistent_tickers(self) -> None:
        benchmark = make_ohlcv(periods=80, drift=0.0005)
        bundles = {
            "AAA": Bundle("Technology", make_ohlcv(periods=80, drift=0.004, last_boost=0.08)),
            "BBB": Bundle("Energy", make_ohlcv(periods=80, drift=0.003, last_boost=0.05)),
            "CCC": Bundle("Utilities", make_ohlcv(periods=80, drift=0.0002)),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "momentum_watchlist.json"
            path.write_text(
                json.dumps(
                    {
                        "top_50": [{"ticker": "AAA"}],
                        "history": [],
                    }
                ),
                encoding="utf-8",
            )
            result = build_momentum_watchlist(benchmark=benchmark, bundles=bundles, path=path)

        self.assertTrue(any(entry.ticker == "AAA" and entry.persistent for entry in result.top_50))
        self.assertIn("AAA", result.persistent_tickers)


if __name__ == "__main__":
    unittest.main()
