"""Composite scoring engine."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

import pandas as pd

from stock_predictor.analysis.indicators import add_indicators
from stock_predictor.analysis.multitimeframe import MultiTimeframeResult
from stock_predictor.analysis.pattern_history import PatternWinRateResult
from stock_predictor.analysis.patterns import PatternResult
from stock_predictor.analysis.sector_impact import SectorImpactResult
from stock_predictor.analysis.smart_money import SmartMoneyResult
from stock_predictor.analysis.squeeze import SqueezeResult
from stock_predictor.analysis.trade_signals import BreakoutResult, RelativeVolumeResult
from stock_predictor.config import AppConfig
from stock_predictor.data.congress import CongressSignal
from stock_predictor.data.quality import DataQualityResult
from stock_predictor.data.supply_chain import SupplyChainSignal
from stock_predictor.models.anomaly_model import AnomalyResult
from stock_predictor.models.ensemble import EnsembleOutput
from stock_predictor.utils import clamp, coerce_float, normalize

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


@dataclass(slots=True)
class CandidateScore:
    ticker: str
    sector: str
    company_name: str
    current_price: float
    market_cap: float
    final_score: float
    score_low: float
    score_high: float
    score_uncertainty: float
    confidence_label: str
    ml_score: float
    technical_score: float
    volume_momentum_score: float
    options_score: float
    sentiment_score: float
    rs_score: float
    institutional_score: float
    news_score: float
    probability_4pct_5d: float
    pattern_name: str
    pattern_score: float
    pattern_win_rate: float
    pattern_win_rate_label: str
    confluence_count: int
    risk_reward: float
    stop_loss: float
    targets: Dict[str, float]
    position_size_pct: float
    kelly_size_pct: float
    data_quality_score: float
    smart_money_score: float
    squeeze_score: float
    gpt_news_score: float
    gpt_news_reason: str
    anomaly_score: float
    sector_tailwind_points: float
    congress_score: float
    supply_chain_score: float
    tier_label: str
    sector_temperature_tag: str
    meets_threshold: bool
    threshold_used: float
    defaulted_signals: Dict[str, bool]
    ai_explanation: str
    notes: List[str]
    diagnostics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NewsCatalystClassifier:
    """Classify catalyst strength with OpenAI or heuristics."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key) if (OpenAI and config.openai_api_key) else None

    def score(self, ticker: str, news_items: Iterable[Dict[str, Any]], sec_filings: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        texts = self._extract_texts(news_items, sec_filings)
        if not texts:
            return {"score": 0.5, "label": "neutral", "impact": 5, "summary": "No fresh catalyst text"}
        if self.client is not None:
            result = self._score_with_openai(ticker, texts)
            if result:
                return result
        return self._heuristic_score(texts)

    def _extract_texts(
        self,
        news_items: Iterable[Dict[str, Any]],
        sec_filings: Iterable[Dict[str, Any]],
    ) -> List[str]:
        texts = []
        floor = datetime.now(timezone.utc).timestamp() - 7 * 24 * 60 * 60
        for item in news_items:
            if item.get("providerPublishTime") and item["providerPublishTime"] < floor:
                continue
            title = item.get("title", "")
            summary = item.get("summary", "")
            if title or summary:
                texts.append(f"{title}\n{summary}".strip())
        for filing in sec_filings:
            form = filing.get("form")
            if form not in {"8-K", "6-K", "SC 13D", "SC 13G", "4"}:
                continue
            texts.append(f"SEC filing {form} accession {filing.get('accessionNumber')}")
        return texts[:20]

    def _score_with_openai(self, ticker: str, texts: List[str]) -> Dict[str, Any] | None:
        prompt = (
            "Classify the stock catalyst tone from these recent news items.\n"
            "Return strict JSON with keys: label, impact, summary.\n"
            "label must be bullish, bearish, or neutral.\n"
            "impact must be an integer 1-10.\n"
            f"Ticker: {ticker}\n"
            f"Items: {json.dumps(texts)}"
        )
        try:
            if hasattr(self.client, "responses"):
                response = self.client.responses.create(
                    model=self.config.openai_model,
                    input=prompt,
                )
                content = response.output_text
            else:  # pragma: no cover
                response = self.client.chat.completions.create(
                    model=self.config.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                content = response.choices[0].message.content
            payload = json.loads(content)
            label = payload.get("label", "neutral")
            impact = int(payload.get("impact", 5))
            return {
                "score": clamp(impact / 10.0, 0.0, 1.0),
                "label": label,
                "impact": impact,
                "summary": payload.get("summary", ""),
            }
        except Exception:
            LOGGER.debug("OpenAI catalyst scoring failed for %s", ticker, exc_info=True)
            return None

    def _heuristic_score(self, texts: List[str]) -> Dict[str, Any]:
        bullish_terms = {
            "buyback",
            "guidance raise",
            "approval",
            "contract",
            "award",
            "acquisition",
            "partner",
            "expansion",
            "beats",
            "record",
            "dividend increase",
        }
        bearish_terms = {
            "investigation",
            "miss",
            "downgrade",
            "delay",
            "lawsuit",
            "offering",
            "recall",
            "dilution",
            "fraud",
        }
        score = 0
        joined = " ".join(text.lower() for text in texts)
        for term in bullish_terms:
            if term in joined:
                score += 1
        for term in bearish_terms:
            if term in joined:
                score -= 1
        label = "neutral"
        if score >= 2:
            label = "bullish"
        elif score <= -2:
            label = "bearish"
        impact = int(clamp(5 + score, 1, 10))
        return {
            "score": clamp(impact / 10.0, 0.0, 1.0),
            "label": label,
            "impact": impact,
            "summary": texts[0][:200],
        }


def build_candidate_score(
    *,
    config: AppConfig,
    ticker: str,
    sector: str,
    info: Dict[str, Any],
    daily_frame: pd.DataFrame,
    hourly_frame: pd.DataFrame,
    pattern: PatternResult,
    pattern_history: PatternWinRateResult,
    ensemble_output: EnsembleOutput,
    options_metrics: Dict[str, float],
    sentiment_metrics: Dict[str, float],
    rs_metrics: Dict[str, float],
    macro_sector_score: float,
    news_metrics: Dict[str, Any],
    gpt_reasoning: Dict[str, Any],
    institutional_metrics: Dict[str, float],
    pre_earnings_metrics: Dict[str, float],
    multi_timeframe: MultiTimeframeResult,
    smart_money: SmartMoneyResult,
    anomaly_result: AnomalyResult,
    supply_chain_signal: SupplyChainSignal,
    congress_signal: CongressSignal,
    sector_impact: SectorImpactResult,
    squeeze_result: SqueezeResult,
    breakout_result: BreakoutResult,
    relative_volume_result: RelativeVolumeResult,
    data_quality: DataQualityResult,
    weights: Dict[str, float],
    model_training_samples: int,
    threshold_used: float,
    sector_temperature_bonus: float,
    sector_temperature_tag: str,
    persistent_momentum_bonus: float,
    float_rotation_bonus: float,
) -> CandidateScore:
    daily_indicators = add_indicators(daily_frame).dropna()
    current_price = coerce_float(info.get("currentPrice") or info.get("lastPrice") or daily_frame["close"].iloc[-1])
    atr_value = coerce_float(daily_indicators["atr"].iloc[-1] if not daily_indicators.empty else 0.0)
    stop_loss = max(0.01, current_price - config.thresholds.stop_atr_multiplier * max(atr_value, current_price * 0.02))
    downside = max(current_price - stop_loss, current_price * 0.005)
    targets = {
        "tp1": round(current_price * 1.02, 2),
        "tp2": round(max(current_price * 1.04, current_price + downside), 2),
        "tp3": round(max(current_price * 1.07, current_price + downside * 2.0), 2),
    }
    upside = targets["tp2"] - current_price
    risk_reward = upside / downside

    technical_signal_scores = _technical_subscores(daily_indicators, rs_metrics)
    technical_score = sum(technical_signal_scores.values()) / max(len(technical_signal_scores), 1)
    volume_momentum_score = _volume_momentum_score(daily_indicators, rs_metrics)
    pattern_signal_active = _pattern_signal_active(config, pattern, pattern_history)
    pattern_component_score = _pattern_component_score(
        config,
        pattern,
        pattern_history,
        active=pattern_signal_active,
    )
    defaulted_signals = {
        "sentiment": bool(sentiment_metrics.get("defaulted")),
        "flow": bool(options_metrics.get("defaulted")),
    }
    defaulted_signal_count = int(sum(defaulted_signals.values()))
    ml_score = normalize(ensemble_output.probability, 0.0, 1.0)
    if model_training_samples < config.thresholds.cold_start_min_samples:
        ml_score = round(technical_score * 0.85, 2)
        ensemble_output.model_status = "cold_start_fallback"
    rs_score = clamp(
        0.8 * coerce_float(rs_metrics.get("rs_rating"), config.default_missing_signal_value)
        + 0.2 * normalize(macro_sector_score, 0.0, 1.0),
        0.0,
        100.0,
    )
    options_score = _optional_base_score(
        raw_score=options_metrics.get("options_score"),
        available=bool(options_metrics.get("has_data")),
        default_value=config.default_missing_signal_value,
    )
    social_sentiment_score = _optional_base_score(
        raw_score=sentiment_metrics.get("sentiment_score"),
        available=bool(sentiment_metrics.get("has_data")),
        default_value=config.default_missing_signal_value,
    )
    gpt_news_score = clamp(coerce_float(gpt_reasoning.get("score"), config.default_missing_signal_value), 0.0, 100.0)
    news_component_score = _optional_base_score(
        raw_score=(coerce_float(news_metrics.get("score"), 0.5) * 100.0 + gpt_news_score) / 2.0,
        available=bool(news_metrics.get("summary") or gpt_reasoning.get("reason")),
        default_value=config.default_missing_signal_value,
        raw_is_percent=True,
    )
    sentiment_score = round((social_sentiment_score + news_component_score) / 2.0, 2)
    institutional_score = clamp(
        normalize(coerce_float(institutional_metrics.get("institutional_score"), 0.5), 0.0, 1.0),
        0.0,
        100.0,
    )
    news_score = news_component_score
    earnings_bonus = coerce_float(pre_earnings_metrics.get("runner_bonus_points"), 0.0)
    confluence_checks = {
        "technical": technical_score >= 65.0,
        "rs": rs_score >= 50.0,
        "volume": volume_momentum_score >= 60.0,
        "ml": ml_score >= 65.0,
        "pattern": pattern_signal_active,
    }
    confluence_count = int(sum(confluence_checks.values()))

    weighted_sum = (
        weights["ml"] * ml_score
        + weights["technical"] * technical_score
        + weights["rs"] * rs_score
        + weights["pattern"] * pattern_component_score
        + weights["volume"] * volume_momentum_score
        + weights["sentiment"] * sentiment_score
        + weights["options"] * options_score
    )
    base_score = clamp(weighted_sum, 0.0, 100.0)
    signal_coverage_penalty = 0.0
    if defaulted_signals["sentiment"]:
        signal_coverage_penalty += config.thresholds.defaulted_sentiment_penalty_points
    if defaulted_signals["flow"]:
        signal_coverage_penalty += config.thresholds.defaulted_flow_penalty_points
    if defaulted_signal_count >= 2:
        signal_coverage_penalty += config.thresholds.multiple_defaulted_signals_penalty_points
    optional_bonus = min(
        10.0,
        4.0 * clamp(supply_chain_signal.score, 0.0, 1.0)
        + 3.0 * clamp(congress_signal.score, 0.0, 1.0)
        + 3.0 * clamp(squeeze_result.score / 100.0, 0.0, 1.0),
    )
    anomaly_points = 5.0 * clamp(anomaly_result.score, 0.0, 1.0) if anomaly_result.is_anomaly and anomaly_result.direction_aligned else -3.0 if anomaly_result.is_anomaly else 0.0
    sector_points = clamp(sector_impact.points * 0.35, -3.0, 3.0)
    smart_money_points = smart_money.adjustment_points
    final_score = (
        base_score
        + optional_bonus
        + earnings_bonus
        + breakout_result.bonus_points
        + relative_volume_result.bonus_points
        + float_rotation_bonus
        + anomaly_points
        + sector_points
        + smart_money_points
        + sector_temperature_bonus
        + persistent_momentum_bonus
    )
    final_score -= signal_coverage_penalty
    if risk_reward < 1.0:
        final_score -= 3.0
    elif risk_reward < 1.5:
        final_score -= 1.0
    final_score -= (1.0 - data_quality.score) * 8.0
    timeframe_penalty = multi_timeframe.penalty_factor
    if multi_timeframe.contradicts:
        timeframe_penalty = 0.35
    elif timeframe_penalty == 0.0:
        timeframe_penalty = 0.65
    final_score *= timeframe_penalty
    final_score = clamp(final_score, 0.0, 100.0)
    assert 0.0 <= final_score <= 100.0

    uncertainty = (
        ensemble_output.score_uncertainty
        + (1.0 - data_quality.score) * 4.0
        + defaulted_signal_count * config.thresholds.defaulted_signal_uncertainty_points
    )
    uncertainty = clamp(uncertainty, 2.0, 15.0)
    confidence_label = ensemble_output.confidence_label
    if defaulted_signal_count >= 2:
        confidence_label = "medium" if confidence_label == "high" else "low"
    elif defaulted_signal_count == 1 and confidence_label == "high":
        confidence_label = "medium"
    if data_quality.score < 0.7:
        confidence_label = "low"
    if final_score < threshold_used:
        confidence_label = "low"

    notes = build_notes(
        config=config,
        pattern=pattern,
        pattern_history=pattern_history,
        sentiment_metrics=sentiment_metrics,
        options_metrics=options_metrics,
        news_metrics=news_metrics,
        institutional_metrics=institutional_metrics,
        pre_earnings_metrics=pre_earnings_metrics,
        multi_timeframe=multi_timeframe,
        smart_money=smart_money,
        anomaly_result=anomaly_result,
        supply_chain_signal=supply_chain_signal,
        congress_signal=congress_signal,
        squeeze_result=squeeze_result,
        breakout_result=breakout_result,
        relative_volume_result=relative_volume_result,
        sector_temperature_tag=sector_temperature_tag,
        persistent_momentum_bonus=persistent_momentum_bonus,
        float_rotation_bonus=float_rotation_bonus,
        data_quality=data_quality,
        defaulted_signals=defaulted_signals,
    )
    if model_training_samples < config.thresholds.cold_start_min_samples:
        notes.append("Model untrained; ML score is using technical fallback")
    if confluence_count < 3:
        notes.append(f"Only {confluence_count}/5 core signals are aligned")
    if risk_reward < 1.0:
        notes.append("Risk/reward below 1.0 after ATR stop sizing")
    if final_score < threshold_used:
        notes.append(f"Below active threshold of {threshold_used:.1f}")
    ai_explanation = build_confidence_explanation(
        technical_score=technical_score,
        volume_momentum_score=volume_momentum_score,
        breakout_result=breakout_result,
        pre_earnings_metrics=pre_earnings_metrics,
        sector_temperature_tag=sector_temperature_tag,
        relative_volume_result=relative_volume_result,
    )
    position_size_pct = clamp(0.08 * (current_price / max(atr_value * 5, current_price * 0.05)), 0.01, 0.12)
    price_chart_frame = (
        daily_frame.tail(30)
        .reset_index()
        .rename(columns={daily_frame.index.name or "index": "date"})
    )
    price_chart_frame.columns = [
        str(column).lower() if str(column).lower() != "index" else "date"
        for column in price_chart_frame.columns
    ]
    price_chart_frame = price_chart_frame.loc[:, ~pd.Index(price_chart_frame.columns).duplicated(keep="last")]
    analyst_view = build_analyst_explainability(
        ticker=ticker,
        final_score=final_score,
        current_price=current_price,
        stop_loss=stop_loss,
        targets=targets,
        risk_reward=risk_reward,
        ensemble_output=ensemble_output,
        data_quality=data_quality,
        pattern_history=pattern_history,
        confluence_checks=confluence_checks,
        technical_scores=technical_signal_scores,
        notes=notes,
        meets_threshold=final_score >= threshold_used,
        threshold_used=threshold_used,
    )

    return CandidateScore(
        ticker=ticker,
        sector=sector,
        company_name=info.get("shortName") or info.get("longName") or ticker,
        current_price=round(current_price, 2),
        market_cap=coerce_float(info.get("marketCap")),
        final_score=round(final_score, 2),
        score_low=round(max(0.0, final_score - uncertainty), 2),
        score_high=round(min(100.0, final_score + uncertainty), 2),
        score_uncertainty=round(uncertainty, 2),
        confidence_label=confidence_label,
        ml_score=round(ml_score, 2),
        technical_score=round(technical_score, 2),
        volume_momentum_score=round(volume_momentum_score, 2),
        options_score=round(options_score, 2),
        sentiment_score=round(sentiment_score, 2),
        rs_score=round(rs_score, 2),
        institutional_score=round(institutional_score, 2),
        news_score=round(news_score, 2),
        probability_4pct_5d=round(normalize(ensemble_output.probability, 0.0, 1.0), 2),
        pattern_name=pattern.name,
        pattern_score=round(pattern_component_score, 2),
        pattern_win_rate=round(pattern_history.win_rate * 100, 2),
        pattern_win_rate_label=pattern_history.label,
        confluence_count=confluence_count,
        risk_reward=round(risk_reward, 2),
        stop_loss=round(stop_loss, 2),
        targets=targets,
        position_size_pct=round(position_size_pct * 100, 2),
        kelly_size_pct=round(position_size_pct * 100, 2),
        data_quality_score=round(data_quality.score * 100, 2),
        smart_money_score=round(smart_money.score * 100, 2),
        squeeze_score=round(squeeze_result.score, 2),
        gpt_news_score=round(gpt_news_score, 2),
        gpt_news_reason=str(gpt_reasoning.get("reason", "")),
        anomaly_score=round(anomaly_result.score * 100, 2),
        sector_tailwind_points=round(sector_impact.points, 2),
        congress_score=round(congress_signal.score * 100, 2),
        supply_chain_score=round(supply_chain_signal.score * 100, 2),
        tier_label="Tier 3",
        sector_temperature_tag=sector_temperature_tag,
        meets_threshold=final_score >= threshold_used,
        threshold_used=threshold_used,
        defaulted_signals=defaulted_signals,
        ai_explanation=ai_explanation,
        notes=notes,
        diagnostics={
            "options": options_metrics,
            "sentiment": sentiment_metrics,
            "rs": rs_metrics,
            "news": news_metrics,
            "institutional": institutional_metrics,
            "pre_earnings": pre_earnings_metrics,
            "multi_timeframe": asdict(multi_timeframe),
            "smart_money": asdict(smart_money),
            "pattern_history": asdict(pattern_history),
            "anomaly": asdict(anomaly_result),
            "supply_chain": asdict(supply_chain_signal),
            "congress": asdict(congress_signal),
            "sector_impact": asdict(sector_impact),
            "squeeze": asdict(squeeze_result),
            "breakout": asdict(breakout_result),
            "relative_volume": asdict(relative_volume_result),
            "persistent_momentum_bonus": persistent_momentum_bonus,
            "float_rotation_bonus": float_rotation_bonus,
            "sector_temperature_bonus": sector_temperature_bonus,
            "data_quality": asdict(data_quality),
            "ensemble": asdict(ensemble_output),
            "gpt_reasoning": gpt_reasoning,
            "analyst": analyst_view,
            "weights": weights,
            "subscores": {
                "technical_signals": technical_signal_scores,
                "pattern_component": round(pattern_component_score, 2),
                "volume_momentum": round(volume_momentum_score, 2),
                "base_score": round(base_score, 2),
                "signal_coverage_penalty": round(signal_coverage_penalty, 2),
                "optional_bonus": round(optional_bonus, 2),
                "confluence_checks": confluence_checks,
                "confluence_count": confluence_count,
            },
            "price_chart": price_chart_frame[["date", "open", "high", "low", "close", "volume"]].to_dict(orient="records"),
        },
    )


def build_analyst_explainability(
    *,
    ticker: str,
    final_score: float,
    current_price: float,
    stop_loss: float,
    targets: Dict[str, float],
    risk_reward: float,
    ensemble_output: EnsembleOutput,
    data_quality: DataQualityResult,
    pattern_history: PatternWinRateResult,
    confluence_checks: Dict[str, bool],
    technical_scores: Dict[str, float],
    notes: List[str],
    meets_threshold: bool,
    threshold_used: float,
) -> Dict[str, Any]:
    """Build compact analyst-facing reasoning for the Swift UI."""

    positive: List[str] = []
    negative: List[str] = []
    for name, passed in sorted(confluence_checks.items()):
        label = name.replace("_", " ").title()
        if passed:
            positive.append(label)
        else:
            negative.append(label)
    strongest = sorted(technical_scores.items(), key=lambda item: item[1], reverse=True)[:3]
    weakest = sorted(technical_scores.items(), key=lambda item: item[1])[:3]
    positive.extend([f"{name.replace('_', ' ').title()} {score:.0f}/100" for name, score in strongest if score >= 55])
    negative.extend([f"{name.replace('_', ' ').title()} {score:.0f}/100" for name, score in weakest if score <= 45])
    data_warnings = list(getattr(data_quality, "issues", []))
    if data_warnings:
        negative.extend([f"Data warning: {warning}" for warning in data_warnings[:2]])
    invalidation = [
        f"Close below stop near ${stop_loss:.2f}",
        "Market regime flips risk-off or VIX expands sharply",
        "Volume dries up below the recent 20-day average",
    ]
    if not meets_threshold:
        why_not = f"Below official threshold by {max(0.0, threshold_used - final_score):.1f} points."
    elif data_quality.score < 0.75:
        why_not = "Eligible score, but data quality needs review before conviction."
    else:
        why_not = "Official candidate if it remains inside portfolio risk limits."
    target = float(targets.get("tp2") or targets.get("tp1") or current_price)
    reward_pct = ((target / current_price) - 1.0) * 100.0 if current_price else 0.0
    risk_pct = ((current_price / stop_loss) - 1.0) * 100.0 if stop_loss else 0.0
    confidence = clamp(float(ensemble_output.probability), 0.0, 1.0)
    return {
        "why_this_pick": positive[:6] or ["Model score is above the current scan floor"],
        "negative_drivers": negative[:6] or ["No major blocker surfaced"],
        "invalidation": invalidation,
        "similar_historical_setups": [
            f"{pattern_history.label} pattern history",
            f"{pattern_history.sample_size} comparable setups tracked",
            f"{pattern_history.win_rate * 100:.0f}% historical win rate",
        ],
        "backtest_confidence": round(confidence * 100.0, 1),
        "model_confidence": getattr(ensemble_output, "confidence_label", "medium"),
        "risk_reward_map": {
            "entry": round(current_price, 2),
            "stop": round(stop_loss, 2),
            "target": round(target, 2),
            "reward_pct": round(reward_pct, 2),
            "risk_pct": round(risk_pct, 2),
            "risk_reward": round(risk_reward, 2),
        },
        "data_freshness": "Current scan artifact",
        "data_quality_warnings": data_warnings[:4],
        "why_not_official": why_not,
        "notes": notes[:4],
    }


def _technical_subscores(
    daily_indicators: pd.DataFrame,
    rs_metrics: Dict[str, float],
) -> Dict[str, float]:
    if daily_indicators.empty:
        return {
            "rsi": 50.0,
            "macd": 50.0,
            "volume": 50.0,
            "price_vs_50ma": 50.0,
            "bollinger": 50.0,
            "adx": 50.0,
            "momentum": 50.0,
            "vwap": 50.0,
        }
    latest = daily_indicators.iloc[-1]
    close = coerce_float(latest["close"])
    lower = coerce_float(latest.get("bollinger_low"), close)
    upper = coerce_float(latest.get("bollinger_high"), close)
    band_position = (close - lower) / max(upper - lower, 1e-6)
    volume_ratio = coerce_float(latest["volume"]) / max(coerce_float(daily_indicators["volume"].tail(20).mean()), 1.0)
    return {
        "rsi": round(clamp(100.0 - abs(coerce_float(latest["rsi"], 50.0) - 45.0) * 2.2, 0.0, 100.0), 2),
        "macd": round(normalize(coerce_float(latest["macd_hist"]), -1.5, 1.5), 2),
        "volume": round(normalize(volume_ratio, 0.75, 2.5), 2),
        "price_vs_50ma": round(normalize(close / max(coerce_float(latest["sma_50"], close), 1e-6), 0.94, 1.10), 2),
        "bollinger": round(clamp(100.0 - abs(band_position - 0.35) * 180.0, 0.0, 100.0), 2),
        "adx": round(normalize(coerce_float(latest["adx"], 20.0), 10.0, 45.0), 2),
        "momentum": round(
            clamp(
                0.55 * normalize(coerce_float(latest["return_5"], 0.0), -0.08, 0.10)
                + 0.45 * normalize(coerce_float(rs_metrics.get("weighted_excess_return"), 0.0), -0.15, 0.15),
                0.0,
                100.0,
            ),
            2,
        ),
        "vwap": round(normalize(coerce_float(latest["vwap_distance"], 0.0), -0.03, 0.04), 2),
    }


def _pattern_signal_active(
    config: AppConfig,
    pattern: PatternResult,
    pattern_history: PatternWinRateResult,
) -> bool:
    if pattern.name == "none":
        return False
    if pattern.score < config.thresholds.pattern_min:
        return False
    if pattern_history.sample_size <= 0:
        return False
    if pattern_history.win_rate < config.thresholds.pattern_min_win_rate:
        return False
    return bool(pattern_history.qualified)


def _pattern_component_score(
    config: AppConfig,
    pattern: PatternResult,
    pattern_history: PatternWinRateResult,
    *,
    active: bool,
) -> float:
    if not active:
        return 0.0
    pattern_strength = normalize(pattern.score, config.thresholds.pattern_min, 10.0)
    history_strength = normalize(
        pattern_history.win_rate,
        config.thresholds.pattern_min_win_rate,
        1.0,
    )
    score = 0.60 * pattern_strength + 0.40 * history_strength
    return clamp(score, 0.0, 100.0)


def _volume_momentum_score(daily_indicators: pd.DataFrame, rs_metrics: Dict[str, float]) -> float:
    if daily_indicators.empty:
        return 50.0
    latest = daily_indicators.iloc[-1]
    volume_ratio = coerce_float(latest["volume"]) / max(coerce_float(daily_indicators["volume"].tail(20).mean()), 1.0)
    obv_trend = 0.0
    if len(daily_indicators) >= 10:
        obv_trend = coerce_float(latest["obv"]) - coerce_float(daily_indicators["obv"].iloc[-10])
    score = (
        0.30 * normalize(volume_ratio, 0.75, 2.5)
        + 0.25 * normalize(coerce_float(latest["return_5"], 0.0), -0.08, 0.10)
        + 0.20 * normalize(coerce_float(latest["return_20"], 0.0), -0.15, 0.20)
        + 0.15 * normalize(obv_trend, -5_000_000.0, 5_000_000.0)
        + 0.10 * normalize(coerce_float(rs_metrics.get("weighted_excess_return"), 0.0), -0.15, 0.15)
    )
    return clamp(score, 0.0, 100.0)


def _optional_base_score(
    *,
    raw_score: float | None,
    available: bool,
    default_value: float,
    raw_is_percent: bool = False,
) -> float:
    if not available:
        return default_value
    if raw_score is None:
        return default_value
    if raw_is_percent:
        return clamp(coerce_float(raw_score, default_value), 0.0, 100.0)
    return clamp(normalize(coerce_float(raw_score, default_value / 100.0), 0.0, 1.0), 0.0, 100.0)


def build_notes(
    *,
    config: AppConfig,
    pattern: PatternResult,
    pattern_history: PatternWinRateResult,
    sentiment_metrics: Dict[str, float],
    options_metrics: Dict[str, float],
    news_metrics: Dict[str, Any],
    institutional_metrics: Dict[str, float],
    pre_earnings_metrics: Dict[str, float],
    multi_timeframe: MultiTimeframeResult,
    smart_money: SmartMoneyResult,
    anomaly_result: AnomalyResult,
    supply_chain_signal: SupplyChainSignal,
    congress_signal: CongressSignal,
    squeeze_result: SqueezeResult,
    breakout_result: BreakoutResult,
    relative_volume_result: RelativeVolumeResult,
    sector_temperature_tag: str,
    persistent_momentum_bonus: float,
    float_rotation_bonus: float,
    data_quality: DataQualityResult,
    defaulted_signals: Dict[str, bool],
) -> List[str]:
    notes: List[str] = []
    pattern_signal_active = _pattern_signal_active(config, pattern, pattern_history)
    if pattern_signal_active:
        notes.append(f"High-quality {pattern.name.replace('_', ' ')} pattern")
        notes.append(pattern_history.label)
    elif pattern.name != "none":
        if pattern.score < config.thresholds.pattern_min:
            notes.append("Pattern detected but its quality score is below the activation threshold")
        elif pattern_history.sample_size <= 0:
            notes.append("Pattern detected but there is not enough historical evidence to use it")
        else:
            notes.append("Pattern detected but its historical win rate is too weak to count")
    if sentiment_metrics.get("mention_velocity", 0.0) >= config.thresholds.sentiment_velocity_bullish:
        notes.append("Social mention velocity is surging")
    if options_metrics.get("put_call_ratio", 1.0) < 0.5:
        notes.append("Bullish put/call skew")
    if news_metrics.get("label") == "bullish":
        notes.append("Recent news catalyst skew is bullish")
    if institutional_metrics.get("insider_buy_signal", 0.0) > 0:
        notes.append("Recent insider buying signal detected")
    if pre_earnings_metrics.get("reliable_runner"):
        notes.append(
            f"PRE-EARNINGS RUNNER - earnings in {int(pre_earnings_metrics.get('days_until_earnings', 0))} days"
        )
    notes.append(multi_timeframe.summary)
    notes.append(smart_money.summary)
    if anomaly_result.is_anomaly:
        notes.append(anomaly_result.summary)
    if breakout_result.confirmed:
        notes.append(breakout_result.summary)
    elif breakout_result.warning:
        notes.append(breakout_result.summary)
    if relative_volume_result.triggered:
        notes.append(relative_volume_result.summary)
    if supply_chain_signal.related_movers:
        notes.append(supply_chain_signal.summary)
    if congress_signal.recent_buys > 0:
        notes.append(congress_signal.summary)
    if squeeze_result.qualifying:
        notes.append("Short-squeeze probability is elevated")
    if sector_temperature_tag:
        notes.append(sector_temperature_tag)
    if persistent_momentum_bonus > 0:
        notes.append("Persistent momentum leader across consecutive weeks")
    if float_rotation_bonus > 0:
        notes.append("Fast float rotation supports momentum follow-through")
    if data_quality.issues:
        notes.append("Data quality warnings: " + "; ".join(data_quality.issues[:2]))
    defaulted = [name for name, active in defaulted_signals.items() if active]
    if defaulted:
        notes.append("Optional signals defaulted to neutral: " + ", ".join(defaulted))
    return notes


def compute_institutional_metrics(
    daily_frame: pd.DataFrame,
    info: Dict[str, Any],
    sec_filings: Iterable[Dict[str, Any]],
) -> Dict[str, float]:
    """Estimate institutional footprint from available public data."""

    data = add_indicators(daily_frame).dropna()
    if data.empty:
        return {"institutional_score": 0.5, "insider_buy_signal": 0.0, "insider_sell_signal": 0.0}
    recent = data.tail(10)
    accumulation_days = int(
        (
            (recent["volume"] > recent["volume"].rolling(5).mean())
            & ((recent["high"] - recent["low"]) / recent["close"] < 0.04)
        ).sum()
    )
    insider_transactions = info.get("insider_transactions", [])
    insider_buy_signal = 0.0
    insider_sell_signal = 0.0
    for transaction in insider_transactions[:10]:
        text = str(transaction).lower()
        if "buy" in text:
            insider_buy_signal = 1.0
        if "sale" in text or "sell" in text:
            insider_sell_signal = 1.0
    filing_signal = 0.0
    for filing in sec_filings:
        if filing.get("form") in {"4", "SC 13D", "SC 13G"}:
            filing_signal += 0.25
    score = clamp(
        0.45 * min(accumulation_days / 5.0, 1.0)
        + 0.30 * insider_buy_signal
        + 0.15 * filing_signal
        - 0.20 * insider_sell_signal,
        0.0,
        1.0,
    )
    return {
        "accumulation_days": float(accumulation_days),
        "insider_buy_signal": insider_buy_signal,
        "insider_sell_signal": insider_sell_signal,
        "institutional_score": score,
    }


def compute_pre_earnings_drift(
    daily_frame: pd.DataFrame,
    earnings_dates: Iterable[Dict[str, Any]],
    *,
    bonus_points: float = 12.0,
) -> Dict[str, float]:
    """Estimate whether a stock tends to drift higher into earnings."""

    if daily_frame.empty:
        return {"drift_score": 0.0, "setup_active": 0.0, "runner_bonus_points": 0.0}
    close = daily_frame["close"]
    volume = daily_frame["volume"]
    historical_drifts = []
    upcoming_within_days = 0.0
    days_until_earnings = None
    for row in earnings_dates:
        raw_date = row.get("earnings_date") or row.get("index") or row.get("date")
        if raw_date is None:
            continue
        try:
            earnings_date = pd.to_datetime(raw_date, utc=True)
        except Exception:
            continue
        if earnings_date > close.index[-1]:
            days_until = (earnings_date - close.index[-1]).days
            if 4 <= days_until <= 8:
                upcoming_within_days = 1.0
                days_until_earnings = days_until
            continue
        subset = close.loc[:earnings_date]
        if len(subset) < 6:
            continue
        drift = subset.iloc[-1] / subset.iloc[-6] - 1
        historical_drifts.append(float(drift))
    recent_history = historical_drifts[-8:]
    average_drift = float(sum(recent_history) / max(len(recent_history), 1))
    positive_hits = sum(drift >= 0.03 for drift in recent_history)
    reliable_runner = len(recent_history) >= 8 and positive_hits >= 6
    recent_return = float(close.iloc[-1] / close.iloc[-4] - 1) if len(close) >= 4 else 0.0
    recent_volume_boost = (
        float(volume.tail(3).mean() / max(volume.tail(20).mean(), 1)) if len(volume) >= 20 else 0.0
    )
    setup_active = 1.0 if (upcoming_within_days and reliable_runner and recent_return >= 0.02 and recent_volume_boost >= 1.2) else 0.0
    drift_score = clamp((average_drift + 0.03) / 0.06, 0.0, 1.0)
    return {
        "average_pre_earnings_drift": average_drift,
        "recent_return_3d": recent_return,
        "recent_volume_boost": recent_volume_boost,
        "setup_active": setup_active,
        "reliable_runner": reliable_runner,
        "runner_hits": positive_hits,
        "runner_samples": len(recent_history),
        "days_until_earnings": float(days_until_earnings or 0.0),
        "runner_bonus_points": bonus_points if setup_active else 0.0,
        "drift_score": drift_score * (0.7 + 0.3 * setup_active),
    }


def build_confidence_explanation(
    *,
    technical_score: float,
    volume_momentum_score: float,
    breakout_result: BreakoutResult,
    pre_earnings_metrics: Dict[str, float],
    sector_temperature_tag: str,
    relative_volume_result: RelativeVolumeResult,
) -> str:
    opening = "Technical momentum is constructive"
    if technical_score >= 70:
        opening = "Technical momentum is strong"
    if breakout_result.confirmed:
        opening = (
            f"{opening} with a confirmed breakout above {breakout_result.resistance_level:.2f}"
        )
    elif breakout_result.warning:
        opening = f"{opening}, but the breakout is still unconfirmed"
    volume_text = f" and volume is tracking {volume_momentum_score:.0f}/100"
    if relative_volume_result.triggered:
        volume_text = (
            f" with unusual early volume running {relative_volume_result.ratio:.1f}x normal"
        )
    macro_text = "Sector conditions are neutral."
    if sector_temperature_tag:
        macro_text = f"{sector_temperature_tag} is adding macro support."
    if pre_earnings_metrics.get("setup_active"):
        macro_text = (
            f"Pre-earnings drift is active with earnings in {int(pre_earnings_metrics.get('days_until_earnings', 0))} days."
        )
    return f"{opening}{volume_text}. {macro_text}"
