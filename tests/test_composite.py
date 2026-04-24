from __future__ import annotations

import unittest

from stock_predictor.analysis.multitimeframe import MultiTimeframeResult
from stock_predictor.analysis.pattern_history import PatternWinRateResult
from stock_predictor.analysis.patterns import PatternResult
from stock_predictor.analysis.sector_impact import SectorImpactResult
from stock_predictor.analysis.smart_money import SmartMoneyResult
from stock_predictor.analysis.squeeze import SqueezeResult
from stock_predictor.analysis.trade_signals import BreakoutResult, RelativeVolumeResult
from stock_predictor.config import get_config
from stock_predictor.data.congress import CongressSignal
from stock_predictor.data.quality import DataQualityResult
from stock_predictor.data.supply_chain import SupplyChainSignal
from stock_predictor.models.anomaly_model import AnomalyResult
from stock_predictor.models.ensemble import EnsembleOutput
from stock_predictor.scoring.composite import build_candidate_score
from tests.helpers import make_info, make_ohlcv


class CompositeScoreTests(unittest.TestCase):
    def _build_candidate(
        self,
        *,
        pattern: PatternResult,
        pattern_history: PatternWinRateResult,
        options_metrics: dict | None = None,
        sentiment_metrics: dict | None = None,
    ):
        config = get_config()
        daily = make_ohlcv(periods=280, drift=0.002, last_boost=0.04)
        hourly = make_ohlcv(periods=200, freq="h", drift=0.0004, last_boost=0.02)
        return build_candidate_score(
            config=config,
            ticker="TEST",
            sector="Technology",
            info=make_info(currentPrice=float(daily["close"].iloc[-1]), shortName="Test Corp"),
            daily_frame=daily,
            hourly_frame=hourly,
            pattern=pattern,
            pattern_history=pattern_history,
            ensemble_output=EnsembleOutput(0.72, 0.0, 0.72, None, "trained", 0.0, 4.0, "high"),
            options_metrics=options_metrics or {"options_score": 0.65, "has_data": True},
            sentiment_metrics=sentiment_metrics or {"sentiment_score": 0.62, "has_data": True},
            rs_metrics={"rs_rating": 72.0, "weighted_excess_return": 0.08},
            macro_sector_score=0.65,
            news_metrics={"score": 0.62, "summary": "positive", "label": "bullish"},
            gpt_reasoning={"score": 64.0, "reason": "positive news"},
            institutional_metrics={"institutional_score": 0.68, "insider_buy_signal": 0.0, "insider_sell_signal": 0.0},
            pre_earnings_metrics={"runner_bonus_points": 0.0, "setup_active": 0.0},
            multi_timeframe=MultiTimeframeResult(
                True,
                False,
                3,
                1.0,
                {"daily": "bullish", "4h": "bullish", "1h": "bullish"},
                {"daily": 0.7, "4h": 0.7, "1h": 0.7},
                "Aligned",
            ),
            smart_money=SmartMoneyResult(0.55, 0.0, False, "neutral", "Neutral"),
            anomaly_result=AnomalyResult(0.0, False, False, 1.0, "No anomaly"),
            supply_chain_signal=SupplyChainSignal(0.0, [], "No signal"),
            congress_signal=CongressSignal(0.0, 0, 0, "No buys"),
            sector_impact=SectorImpactResult(0.0, 0.5, "Neutral"),
            squeeze_result=SqueezeResult(0.0, False, "No squeeze"),
            breakout_result=BreakoutResult(0.0, False, False, False, False, "", 0.0, "No breakout"),
            relative_volume_result=RelativeVolumeResult(1.0, False, 0.0, "No alert"),
            data_quality=DataQualityResult(1.0, True, []),
            weights=config.signal_weights.as_dict(),
            model_training_samples=500,
            threshold_used=54.0,
            sector_temperature_bonus=0.0,
            sector_temperature_tag="",
            persistent_momentum_bonus=0.0,
            float_rotation_bonus=0.0,
        )

    def test_unqualified_pattern_does_not_count_toward_score_or_confluence(self) -> None:
        qualified = self._build_candidate(
            pattern=PatternResult("bull_flag", 8.4, "Strong flag"),
            pattern_history=PatternWinRateResult("bull_flag", 0.68, 30, True),
        )
        weak = self._build_candidate(
            pattern=PatternResult("bull_flag", 8.4, "Strong flag"),
            pattern_history=PatternWinRateResult("bull_flag", 0.28, 30, False),
        )

        self.assertGreater(qualified.pattern_score, 0.0)
        self.assertEqual(weak.pattern_score, 0.0)
        self.assertEqual(qualified.confluence_count, weak.confluence_count + 1)
        self.assertTrue(
            any("historical win rate is too weak" in note for note in weak.notes),
            msg=weak.notes,
        )

    def test_missing_pattern_does_not_award_free_confluence(self) -> None:
        candidate = self._build_candidate(
            pattern=PatternResult("none", 0.0, "No pattern"),
            pattern_history=PatternWinRateResult("none", 0.0, 0, False),
        )

        self.assertEqual(candidate.pattern_score, 0.0)
        self.assertLess(candidate.confluence_count, 3)

    def test_defaulted_optional_signals_reduce_score_and_confidence(self) -> None:
        covered = self._build_candidate(
            pattern=PatternResult("bull_flag", 8.4, "Strong flag"),
            pattern_history=PatternWinRateResult("bull_flag", 0.68, 30, True),
        )
        defaulted = self._build_candidate(
            pattern=PatternResult("bull_flag", 8.4, "Strong flag"),
            pattern_history=PatternWinRateResult("bull_flag", 0.68, 30, True),
            options_metrics={"options_score": None, "has_data": False, "defaulted": True},
            sentiment_metrics={"sentiment_score": None, "has_data": False, "defaulted": True},
        )

        self.assertLess(defaulted.final_score, covered.final_score)
        self.assertGreater(defaulted.score_uncertainty, covered.score_uncertainty)
        self.assertIn(defaulted.confidence_label, {"medium", "low"})
        self.assertTrue(
            any("Optional signals defaulted to neutral" in note for note in defaulted.notes),
            msg=defaulted.notes,
        )


if __name__ == "__main__":
    unittest.main()
