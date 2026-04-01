from __future__ import annotations

import unittest

from stock_predictor.analysis.squeeze import calculate_squeeze_probability
from tests.helpers import make_info, make_ohlcv


class SqueezeTests(unittest.TestCase):
    def test_high_short_interest_and_volume_surge_qualify(self) -> None:
        frame = make_ohlcv(periods=60, freq="D", drift=0.003, last_boost=0.08)
        frame.loc[frame.index[-5:], "volume"] = frame["volume"].iloc[-20:-5].mean() * 3.0
        info = make_info(shortPercentOfFloat=0.30, shortRatio=6.5, borrowRate=0.14)
        result = calculate_squeeze_probability(info, frame)
        self.assertGreater(result.score, 70)
        self.assertTrue(result.qualifying)


if __name__ == "__main__":
    unittest.main()
