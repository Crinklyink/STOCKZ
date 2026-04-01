from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from stock_predictor.config import AppConfig
from stock_predictor.data.cache import SQLiteCache
from stock_predictor.scoring.news_reasoning import GPTNewsReasoner


class NewsReasoningTests(unittest.TestCase):
    def test_reasoning_is_cached(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            reasoner = GPTNewsReasoner(AppConfig(openai_api_key=None), SQLiteCache(Path(tmpdir) / "cache.sqlite3"))
            headlines = ["Company beats expectations and raises guidance"]
            with patch.object(reasoner, "_heuristic_reasoning", wraps=reasoner._heuristic_reasoning) as mocked:
                first = reasoner.reason_about_news("NVDA", headlines)
                second = reasoner.reason_about_news("NVDA", headlines)
            self.assertEqual(first, second)
            self.assertGreater(first["score"], 50.0)
            self.assertEqual(mocked.call_count, 1)


if __name__ == "__main__":
    unittest.main()
