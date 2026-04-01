from __future__ import annotations

import unittest

from stock_predictor.data.supply_chain import compute_supply_chain_signal
from tests.helpers import make_ohlcv


class SupplyChainTests(unittest.TestCase):
    def test_related_movers_boost_supply_chain_signal(self) -> None:
        price_frames = {
            "TSM": make_ohlcv(periods=30, freq="D", drift=0.005, last_boost=0.08),
            "ASML": make_ohlcv(periods=30, freq="D", drift=0.004, last_boost=0.07),
            "AMAT": make_ohlcv(periods=30, freq="D", drift=0.004, last_boost=0.05),
            "LRCX": make_ohlcv(periods=30, freq="D", drift=0.003),
            "MU": make_ohlcv(periods=30, freq="D", drift=0.003),
        }
        result = compute_supply_chain_signal("NVDA", price_frames)
        self.assertGreater(result.score, 0.5)
        self.assertIn("TSM", result.related_movers)


if __name__ == "__main__":
    unittest.main()
