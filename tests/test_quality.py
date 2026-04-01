from __future__ import annotations

import unittest

from stock_predictor.data.quality import validate_ticker_data
from tests.helpers import make_info, make_ohlcv


class DataQualityTests(unittest.TestCase):
    def test_valid_data_passes(self) -> None:
        result = validate_ticker_data(
            info=make_info(),
            daily=make_ohlcv(periods=120, freq="D"),
            hourly=make_ohlcv(periods=120, freq="h"),
            minimum_score=0.6,
        )
        self.assertTrue(result.is_valid)
        self.assertGreaterEqual(result.score, 0.6)

    def test_missing_fields_fail(self) -> None:
        result = validate_ticker_data(
            info={"averageVolume": None},
            daily=make_ohlcv(periods=120, freq="D"),
            hourly=make_ohlcv(periods=0),
            minimum_score=0.8,
        )
        self.assertFalse(result.is_valid)
        self.assertTrue(result.issues)


if __name__ == "__main__":
    unittest.main()
