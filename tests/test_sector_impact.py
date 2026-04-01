from __future__ import annotations

import unittest

from stock_predictor.analysis.sector_impact import SectorImpactEngine, trailing_return
from stock_predictor.config import AppConfig
from tests.helpers import make_ohlcv


class FakeFetcher:
    def fetch_macro_history(self, ticker: str, fresh: bool = False):
        mapping = {
            "CL=F": make_ohlcv(periods=20, freq="D", drift=0.01, last_boost=0.04),
            "NG=F": make_ohlcv(periods=20, freq="D", drift=0.008, last_boost=0.03),
            "SPY": make_ohlcv(periods=20, freq="D", drift=0.002),
        }
        return mapping.get(ticker, make_ohlcv(periods=20, freq="D", drift=0.001))


class SectorImpactTests(unittest.TestCase):
    def test_energy_sector_gets_positive_tailwind(self) -> None:
        engine = SectorImpactEngine(AppConfig(fred_api_key=None), FakeFetcher())
        result = engine.score_sector("Energy")
        self.assertGreater(result.points, 0)
        self.assertGreater(result.normalized_score, 0.5)

    def test_trailing_return_handles_regular_frame(self) -> None:
        frame = make_ohlcv(periods=10, freq="D", drift=0.01)
        self.assertGreater(trailing_return(frame, periods=5), 0)


if __name__ == "__main__":
    unittest.main()
