from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from main import persist_adaptive_regime_summary
from stock_predictor.config import get_config


class MainAdaptiveReportingTests(unittest.TestCase):
    def test_persist_adaptive_regime_summary_updates_metadata_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_config()
            adaptive_path = Path(tmpdir) / "adaptive.json"
            xgb_path = Path(tmpdir) / "xgb.json"
            config.adaptive_metadata_path = adaptive_path
            config.xgb_metadata_path = xgb_path

            persist_adaptive_regime_summary(
                config,
                {"neutral": {"weeks": 4, "win_rate": 55.0, "average_return": 1.8}},
                summary={"weeks": 4, "win_rate": 55.0},
                source="adaptive_backtest",
            )

            adaptive_meta = json.loads(adaptive_path.read_text(encoding="utf-8"))
            xgb_meta = json.loads(xgb_path.read_text(encoding="utf-8"))

            self.assertEqual(adaptive_meta["adaptive_regime_summary_source"], "adaptive_backtest")
            self.assertEqual(adaptive_meta["regime_summary"]["neutral"]["weeks"], 4)
            self.assertEqual(xgb_meta["regime_summary"]["neutral"]["win_rate"], 55.0)


if __name__ == "__main__":
    unittest.main()
