from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from main import update_single_analysis_history
from ui.widgets import should_ignore_analysis_output_line


class SingleAnalysisTests(unittest.TestCase):
    def test_history_keeps_latest_unique_tickers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SimpleNamespace(single_analysis_history_path=Path(tmpdir) / "single_analysis_history.json")
            update_single_analysis_history(
                config,
                {"ticker": "AAPL", "generated_at": "2026-03-31T20:00:00Z", "candidate": {"final_score": 50.0}},
            )
            update_single_analysis_history(
                config,
                {"ticker": "MSFT", "generated_at": "2026-03-31T20:01:00Z", "candidate": {"final_score": 60.0}},
            )
            update_single_analysis_history(
                config,
                {"ticker": "AAPL", "generated_at": "2026-03-31T20:02:00Z", "candidate": {"final_score": 70.0}},
            )

            payload = json.loads(config.single_analysis_history_path.read_text(encoding="utf-8"))
            items = payload["items"]
            self.assertEqual([item["ticker"] for item in items], ["AAPL", "MSFT"])
            self.assertEqual(items[0]["generated_at"], "2026-03-31T20:02:00Z")

    def test_noise_filter_hides_transport_and_model_chatter(self) -> None:
        noisy_lines = [
            "RequestsDependencyWarning: urllib3 (2.6.3) doesn't match a supported version!",
            '2026-03-31 20:02:58,692 | INFO | httpx | HTTP Request: GET https://huggingface.co/api/models/ProsusAI/finbert "HTTP/1.1 200 OK"',
            "Loading weights: 100%|██████████| 201/201",
        ]
        clean_lines = [
            "Analyzing AAPL…",
            "Regime neutral; VIX 25.25",
            "Below current threshold (54.0)",
        ]

        for line in noisy_lines:
            self.assertTrue(should_ignore_analysis_output_line(line))
        for line in clean_lines:
            self.assertFalse(should_ignore_analysis_output_line(line))


if __name__ == "__main__":
    unittest.main()
