from __future__ import annotations

import unittest

from stock_predictor.scoring.adaptive import compute_signal_accuracies, evolve_weights


class AdaptiveWeightsTests(unittest.TestCase):
    def test_evolve_weights_rewards_accurate_signals_and_penalizes_wrong_streaks(self) -> None:
        rows = [
            {"signal_name": "options", "week": "2026-W08", "signal_score": 80, "hit_target": 1},
            {"signal_name": "options", "week": "2026-W09", "signal_score": 82, "hit_target": 1},
            {"signal_name": "options", "week": "2026-W10", "signal_score": 85, "hit_target": 1},
            {"signal_name": "options", "week": "2026-W11", "signal_score": 79, "hit_target": 0},
            {"signal_name": "sentiment", "week": "2026-W09", "signal_score": 75, "hit_target": 0},
            {"signal_name": "sentiment", "week": "2026-W10", "signal_score": 76, "hit_target": 0},
            {"signal_name": "sentiment", "week": "2026-W11", "signal_score": 77, "hit_target": 0},
        ]
        base = {
            "ml": 0.25,
            "technical": 0.20,
            "options": 0.18,
            "sentiment": 0.12,
            "rs": 0.10,
            "institutional": 0.08,
            "news": 0.07,
        }

        rolling, streaks = compute_signal_accuracies(rows)
        result = evolve_weights(base, rows)

        self.assertGreater(rolling["options"], rolling["sentiment"])
        self.assertEqual(streaks["sentiment"], 3)
        self.assertAlmostEqual(sum(result.weights.values()), 1.0, places=6)
        self.assertGreater(result.weights["options"], result.weights["sentiment"])


if __name__ == "__main__":
    unittest.main()
