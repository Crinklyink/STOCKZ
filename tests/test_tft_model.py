from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stock_predictor.config import AppConfig
from stock_predictor.models.tft_model import TemporalFusionPredictor
from tests.helpers import make_ohlcv


class TFTModelTests(unittest.TestCase):
    def test_predictor_falls_back_to_heuristic_when_heavy_deps_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(checkpoint_dir=Path(tmpdir))
            predictor = TemporalFusionPredictor(config)
            predictor.enabled = False
            predictor.fit({"AAA": make_ohlcv(periods=260, freq="D", drift=0.002)})
            output = predictor.predict_proba(make_ohlcv(periods=260, freq="D", drift=0.003))
        self.assertEqual(output.status, "heuristic")
        self.assertGreaterEqual(output.probability, 0.0)
        self.assertLessEqual(output.probability, 1.0)


if __name__ == "__main__":
    unittest.main()
