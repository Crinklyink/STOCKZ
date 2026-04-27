from __future__ import annotations

import unittest

from stock_predictor.config import AppConfig
from stock_predictor.data.sentiment import SentimentEngine


class SentimentEngineTests(unittest.TestCase):
    def test_finbert_is_opt_in_by_default(self) -> None:
        engine = SentimentEngine(AppConfig(), cache=None)  # type: ignore[arg-type]

        scores = engine._score_finbert_batch({"AAPL": ["Apple beats earnings estimates."]})

        self.assertEqual(scores, {"AAPL": 0.0})
        self.assertIs(engine._finbert, False)


if __name__ == "__main__":
    unittest.main()
