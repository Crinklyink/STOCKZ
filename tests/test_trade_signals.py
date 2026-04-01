from __future__ import annotations

import unittest

from stock_predictor.analysis.trade_signals import confirm_breakout, detect_relative_volume_alert
from tests.helpers import make_ohlcv


class TradeSignalTests(unittest.TestCase):
    def test_confirm_breakout_sets_bonus_when_all_checks_pass(self) -> None:
        frame = make_ohlcv(periods=260, drift=0.0025, last_boost=0.08, volume_base=1_500_000)
        breakout_level = frame["high"].iloc[:-2].max() * 1.03
        frame.iloc[-2:, frame.columns.get_loc("close")] = breakout_level
        frame.iloc[-2:, frame.columns.get_loc("high")] = breakout_level * 1.001
        frame.iloc[-1, frame.columns.get_loc("volume")] = frame["volume"].tail(20).mean() * 2.2

        result = confirm_breakout(frame, bonus_points=10.0)

        self.assertTrue(result.confirmed)
        self.assertEqual(result.bonus_points, 10.0)

    def test_relative_volume_alert_flags_large_intraday_ratio(self) -> None:
        frame = make_ohlcv(periods=240, freq="h", drift=0.001, volume_base=100_000)
        latest_day = frame.index[-1].normalize()
        frame.loc[frame.index.normalize() == latest_day, "volume"] *= 4.0

        result = detect_relative_volume_alert(frame, bonus_points=8.0)

        self.assertTrue(result.triggered)
        self.assertEqual(result.bonus_points, 8.0)


if __name__ == "__main__":
    unittest.main()
