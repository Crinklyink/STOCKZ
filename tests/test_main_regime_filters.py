from __future__ import annotations

import unittest

from main import _single_candidate_eligibility, build_report_candidates, rebalance_stage1_survivors
from stock_predictor.analysis.prefilter import PrefilterEntry, PrefilterResult
from stock_predictor.config import get_config
from tests.helpers import make_candidate, make_ohlcv


class MainRegimeFilterTests(unittest.TestCase):
    def test_rebalance_stage1_survivors_includes_defensive_names_in_downtrend(self) -> None:
        config = get_config()
        benchmark = make_ohlcv(periods=80, freq="D", start_price=500.0, drift=-0.01, volatility=0.002)
        rows = [
            PrefilterEntry(ticker=f"TECH{i}", sector="Technology", rs_score=90, momentum_5d_score=90, volume_ratio_score=80, stage1_score=90 - i)
            for i in range(5)
        ] + [
            PrefilterEntry(ticker="XLV1", sector="Healthcare", rs_score=55, momentum_5d_score=45, volume_ratio_score=60, stage1_score=40),
            PrefilterEntry(ticker="XLP1", sector="Consumer Staples", rs_score=54, momentum_5d_score=44, volume_ratio_score=59, stage1_score=39),
            PrefilterEntry(ticker="XLU1", sector="Utilities", rs_score=53, momentum_5d_score=43, volume_ratio_score=58, stage1_score=38),
        ]
        prefilter = PrefilterResult(
            total_tickers=len(rows),
            warm_tickers=len(rows),
            cache_hit_rate=1.0,
            survivors=[row.ticker for row in rows[:5]],
            top_rows=rows,
        )

        survivors = rebalance_stage1_survivors(config, prefilter, benchmark, survivor_limit=8)

        self.assertIn("XLV1", survivors)
        self.assertIn("XLP1", survivors)
        self.assertIn("XLU1", survivors)

    def test_build_report_candidates_falls_back_to_defensive_near_misses(self) -> None:
        config = get_config()
        benchmark = make_ohlcv(periods=80, freq="D", start_price=500.0, drift=-0.01, volatility=0.002)
        tech = make_candidate("TECH", "Technology")
        tech.final_score = 82.0
        health = make_candidate("XLV1", "Healthcare")
        health.final_score = 42.0
        health.meets_threshold = False
        health.tier_label = "🥉 TIER 3"
        health.confluence_count = 3
        health.risk_reward = 1.2
        staples = make_candidate("XLP1", "Consumer Staples")
        staples.final_score = 41.0
        staples.meets_threshold = False
        staples.tier_label = "🥉 TIER 3"
        staples.confluence_count = 3
        staples.risk_reward = 1.1

        report_candidates = build_report_candidates(
            config,
            selected=[],
            candidates=[tech, health, staples],
            benchmark=benchmark,
            top_n=5,
            vix=26.5,
        )

        tickers = [candidate.ticker for candidate in report_candidates]
        self.assertNotIn("TECH", tickers)
        self.assertIn("XLV1", tickers)
        self.assertIn("XLP1", tickers)

    def test_single_candidate_eligibility_explains_rejection_reasons(self) -> None:
        config = get_config()
        benchmark = make_ohlcv(periods=80, freq="D", start_price=500.0, drift=-0.01, volatility=0.002)
        candidate = make_candidate("TECH", "Technology")
        candidate.final_score = 63.0
        candidate.ml_score = 61.0
        candidate.confluence_count = 2
        candidate.risk_reward = 0.8
        candidate.tier_label = "🥈 TIER 2"

        eligible, reasons = _single_candidate_eligibility(
            config,
            candidate,
            threshold_used=70.0,
            vix=26.8,
            benchmark=benchmark,
        )

        self.assertFalse(eligible)
        self.assertIn("Below current threshold (70.0)", reasons)
        self.assertIn("Fewer than 3/5 core signals aligned", reasons)
        self.assertIn("Risk/reward below 1.0", reasons)
        self.assertIn("High-VIX rule requires ML >= 70", reasons)


if __name__ == "__main__":
    unittest.main()
