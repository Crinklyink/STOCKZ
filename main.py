"""Master orchestrator for the stock prediction system."""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Dict, Iterable, Optional

import pandas as pd

try:  # pragma: no cover - optional runtime dependency
    import schedule
except Exception:  # pragma: no cover
    schedule = None

try:  # pragma: no cover - optional runtime dependency
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

warnings.filterwarnings(
    "ignore",
    message=r".*urllib3.*doesn't match a supported version.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"Warning: You are sending unauthenticated requests to the HF Hub.*",
)
try:  # pragma: no cover - optional dependency warning class
    from requests import RequestsDependencyWarning

    warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
except Exception:
    pass

from stock_predictor.analysis.macro import MacroRegimeAnalyzer
from stock_predictor.analysis.momentum import (
    MomentumWatchlistResult,
    build_momentum_watchlist,
    float_rotation_ratio,
    rank_priority_tickers,
)
from stock_predictor.analysis.multitimeframe import MultiTimeframeResult, evaluate_multi_timeframe_alignment
from stock_predictor.analysis.pattern_history import PatternWinRateAnalyzer, PatternWinRateResult
from stock_predictor.analysis.patterns import detect_patterns
from stock_predictor.analysis.prefilter import PrefilterResult, build_fast_prefilter
from stock_predictor.analysis.rs_rating import compute_rs_scores
from stock_predictor.analysis.sector_impact import SectorImpactEngine, SectorImpactResult
from stock_predictor.analysis.smart_money import SmartMoneyResult, evaluate_smart_money_divergence
from stock_predictor.analysis.squeeze import SqueezeResult, calculate_squeeze_probability
from stock_predictor.analysis.trade_signals import BreakoutResult, RelativeVolumeResult, confirm_breakout, detect_relative_volume_alert
from stock_predictor.config import get_config
from stock_predictor.data.cache import SQLiteCache
from stock_predictor.data.congress import CongressSignal, CongressTradeTracker
from stock_predictor.data.dark_pool import FlowDetector
from stock_predictor.data.fetcher import MarketDataFetcher, filter_recent_news, provider_health_check
from stock_predictor.data.quality import DataQualityResult, validate_ticker_data
from stock_predictor.data.sentiment import SentimentEngine
from stock_predictor.data.supply_chain import SupplyChainSignal, compute_supply_chain_signal
from stock_predictor.models.adaptive_model import AdaptivePredictor
from stock_predictor.models.anomaly_model import AnomalyDetector, AnomalyResult
from stock_predictor.models.ensemble import build_tree_ensemble_output
from stock_predictor.output.alerts import send_alerts
from stock_predictor.output.backtest import BacktestTracker, SignalAttributionTracker, with_resolved_outcomes
from stock_predictor.output.bot import send_endweek_results, send_midweek_update, send_weekly_picks
from stock_predictor.output.pdf_report import generate_pdf_report
from stock_predictor.output.report import export_reports, render_terminal_report
from stock_predictor.scoring.composite import (
    CandidateScore,
    NewsCatalystClassifier,
    build_candidate_score,
    compute_institutional_metrics,
    compute_pre_earnings_drift,
)
from stock_predictor.scoring.news_reasoning import GPTNewsReasoner
from stock_predictor.scoring.portfolio import select_portfolio
from stock_predictor.utils import json_default, save_json, setup_logging

LOGGER = logging.getLogger(__name__)


def configure_quiet_runtime_output() -> None:
    """Reduce third-party noise for interactive single-ticker analysis flows."""

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ["STOCK_PREDICTOR_QUIET_RUNTIME"] = "1"
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore",
        message=r".*urllib3.*doesn't match a supported version.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Warning: You are sending unauthenticated requests to the HF Hub.*",
    )
    try:  # pragma: no cover - optional dependency warning class
        from requests import RequestsDependencyWarning

        warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
    except Exception:
        pass


def update_single_analysis_history(config, payload: Dict[str, object], *, limit: int = 20) -> None:
    """Persist a rolling history of recent single-ticker analyses."""

    history_path = config.single_analysis_history_path
    existing: list[dict[str, object]] = []
    if history_path.exists():
        try:
            history_payload = json.loads(history_path.read_text(encoding="utf-8"))
            raw_items = history_payload.get("items", history_payload)
            if isinstance(raw_items, list):
                existing = [item for item in raw_items if isinstance(item, dict)]
        except Exception:
            existing = []
    ticker = str(payload.get("ticker", "")).upper()
    filtered = [item for item in existing if str(item.get("ticker", "")).upper() != ticker]
    history_items = [payload, *filtered][:limit]
    save_json(history_path, {"items": history_items})


def persist_adaptive_regime_summary(
    config,
    regime_summary: Dict[str, object],
    *,
    summary: Dict[str, object] | None = None,
    source: str = "adaptive_backtest",
) -> None:
    """Persist the latest adaptive regime summary into model metadata for UI/reporting."""

    if not regime_summary:
        return
    metadata: dict[str, object] = {}
    if config.adaptive_metadata_path.exists():
        try:
            metadata = json.loads(config.adaptive_metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
    metadata["regime_summary"] = regime_summary
    metadata["adaptive_regime_summary_source"] = source
    if summary:
        metadata["adaptive_backtest_summary"] = summary
    payload = json.dumps(metadata, indent=2)
    config.adaptive_metadata_path.write_text(payload, encoding="utf-8")
    config.xgb_metadata_path.write_text(payload, encoding="utf-8")


@dataclass(slots=True)
class CandidateContext:
    ticker: str
    sector: str
    info: dict
    daily_frame: pd.DataFrame
    hourly_frame: pd.DataFrame
    earnings_dates: list[dict]
    macro_sector_score: float
    pattern_name: str
    pattern_score: float
    pattern_result: object
    options_metrics: dict
    sentiment_metrics: dict
    rs_metrics: dict
    news_metrics: dict
    institutional_metrics: dict
    pre_earnings_metrics: dict
    multi_timeframe: object
    smart_money: object
    anomaly_result: object
    supply_chain_signal: object
    congress_signal: object
    sector_impact: object
    squeeze_result: object
    breakout_result: object
    relative_volume_result: object
    sector_temperature_bonus: float
    sector_temperature_tag: str
    persistent_momentum_bonus: float
    float_rotation_bonus: float
    data_quality: object
    headlines: list[str]


def run_scan(
    fresh: bool = False,
    *,
    alert_email: str | None = None,
    alert_phone: str | None = None,
    threshold_override: float | None = None,
    top_n: int | None = None,
    universe_mode: str | None = None,
    debug: bool = False,
    paper_trade: bool = False,
    send_notifications: bool = True,
) -> Dict[str, object]:
    """Execute one end-to-end prediction scan."""

    config = get_config()
    if top_n is not None:
        config.top_n = max(1, top_n)
    setup_logging(config.predictions_log)
    started = time.perf_counter()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cache = SQLiteCache(config.cache_db)
    fetcher = MarketDataFetcher(config, cache)
    sentiment_engine = SentimentEngine(config, cache)
    flow_detector = FlowDetector(config, cache, fetcher)
    macro_analyzer = MacroRegimeAnalyzer(config, fetcher)
    news_classifier = NewsCatalystClassifier(config)
    gpt_reasoner = GPTNewsReasoner(config, cache)
    backtest = BacktestTracker(config.backtest_db)
    paper_tracker = BacktestTracker(config.paper_trade_db)
    attribution_tracker = SignalAttributionTracker(config.signal_attribution_db)
    congress_tracker = CongressTradeTracker(config, cache)
    sector_impact_engine = SectorImpactEngine(config, fetcher)

    resolved_mode = universe_mode or config.default_universe
    universe_map = fetcher.resolve_universe(mode=resolved_mode, fresh=fresh)
    all_tickers = sorted({ticker for sector_tickers in universe_map.values() for ticker in sector_tickers})
    sector_by_ticker = {
        ticker: sector
        for sector, tickers in universe_map.items()
        for ticker in tickers
    }
    benchmark = fetcher.download_history(
        config.benchmark_ticker,
        interval="1d",
        period=config.daily_period,
        ttl_seconds=config.cache_ttls.market_history,
        fresh=fresh,
        cache_only=not fresh,
    )
    if benchmark.empty:
        benchmark = fetcher.download_history(
            config.benchmark_ticker,
            interval="1d",
            period=config.daily_period,
            ttl_seconds=config.cache_ttls.market_history,
            fresh=fresh,
            cache_only=False,
        )
    sector_impact_engine.shared_benchmark_return = trailing_return_safe(benchmark, periods=5)
    stage1_started = time.perf_counter()
    stage1_daily_frames = fetcher.fetch_daily_frames_for_tickers(
        all_tickers,
        fresh=fresh,
        cache_only=not fresh,
    )
    macro_snapshot = macro_analyzer.build_snapshot(
        fresh=fresh,
        price_frames=stage1_daily_frames,
    )
    stage1_prefilter = build_fast_prefilter(
        universe=universe_map,
        daily_frames=stage1_daily_frames,
        benchmark=benchmark,
        survivor_limit=config.stage1_limit if config.feature_flags.two_stage_scan else len(all_tickers),
    )
    if len(stage1_prefilter.survivors) < min(config.stage1_limit, len(all_tickers)):
        missing_tickers = [ticker for ticker, frame in stage1_daily_frames.items() if frame.empty]
        if missing_tickers:
            refill_frames = fetcher.fetch_daily_frames_for_tickers(missing_tickers, fresh=fresh, cache_only=False)
            stage1_daily_frames.update(refill_frames)
            stage1_prefilter = build_fast_prefilter(
                universe=universe_map,
                daily_frames=stage1_daily_frames,
                benchmark=benchmark,
                survivor_limit=config.stage1_limit if config.feature_flags.two_stage_scan else len(all_tickers),
            )
    stage1_elapsed = time.perf_counter() - stage1_started
    stage1_limit = config.stage1_limit if config.feature_flags.two_stage_scan else len(all_tickers)
    survivor_tickers = (
        rebalance_stage1_survivors(config, stage1_prefilter, benchmark, survivor_limit=stage1_limit)
        if config.feature_flags.two_stage_scan and stage1_prefilter.survivors
        else all_tickers
    )
    survivor_sector_map = {ticker: sector_by_ticker.get(ticker, "Unknown") for ticker in survivor_tickers}
    survivor_bundles = fetcher.fetch_selected_bundles(
        survivor_sector_map,
        daily_frames=stage1_daily_frames,
        fresh=fresh,
    )
    momentum_watchlist = (
        build_momentum_watchlist(
            benchmark=benchmark,
            bundles={
                ticker: SimpleNamespace(
                    daily=stage1_daily_frames.get(ticker, pd.DataFrame()),
                    sector=sector_by_ticker.get(ticker, "Unknown"),
                )
                for ticker in all_tickers
                if not stage1_daily_frames.get(ticker, pd.DataFrame()).empty
            },
            path=config.momentum_watchlist_path,
        )
        if config.feature_flags.momentum_watchlist
        else None
    )
    universe = survivor_bundles
    all_price_frames = {
        ticker: frame
        for ticker, frame in stage1_daily_frames.items()
        if not frame.empty
    }
    price_frames = {ticker: bundle.daily for ticker, bundle in universe.items()}
    prices = {
        ticker: (
            fetcher.summarize_company(bundle)["price"]
            or float(bundle.daily["close"].iloc[-1])
            if not bundle.daily.empty
            else 0.0
        )
        for ticker, bundle in universe.items()
    }
    float_rotation_bonus_map = compute_float_rotation_bonuses(config, universe)

    backtest.evaluate_due_predictions(prices, price_paths=all_price_frames)
    paper_tracker.evaluate_due_paper_predictions(prices, price_paths=all_price_frames)
    attribution_tracker.evaluate_due_predictions(prices, price_paths=all_price_frames)
    if config.feature_flags.adaptive_weights:
        adaptive_weights = backtest.refresh_adaptive_weights(run_id, config.signal_weights.as_dict())
    else:
        adaptive_weights = config.signal_weights.as_dict()

    sentiment_scores = sentiment_engine.score_sentiment(universe.keys(), fresh=fresh)
    flow_scores = flow_detector.score_universe(universe.keys(), prices, fresh=fresh)
    rs_scores = compute_rs_scores(price_frames, benchmark)

    xgb = AdaptivePredictor(config)
    xgb.active_breadth_percentile = macro_snapshot.breadth_percentile
    adaptive_snapshot = xgb.regime_classifier.build_snapshot(
        price_frames=all_price_frames,
        benchmark_history=benchmark,
        breadth_history=fetcher.calculate_breadth_history(stage1_daily_frames),
        sector_histories=fetcher.fetch_sector_history_for_period(config.daily_period, fresh=fresh),
        vix_history=fetcher.fetch_macro_history(config.vix_ticker, fresh=fresh),
        put_call_history=fetcher.fetch_macro_history(config.put_call_ticker, fresh=fresh),
        hyg_history=fetcher.fetch_macro_history(config.hyg_ticker, fresh=fresh),
        lqd_history=fetcher.fetch_macro_history(config.lqd_ticker, fresh=fresh),
        tlt_history=fetcher.fetch_macro_history(config.tlt_ticker, fresh=fresh),
    )
    xgb.active_regime = adaptive_snapshot.label
    xgb.active_market_snapshot = adaptive_snapshot
    anomaly_detector = AnomalyDetector()
    training_report = xgb.last_report
    if config.feature_flags.auto_training_pipeline and not xgb.is_trained:
        training_frames = fetcher.fetch_daily_frames_for_tickers(
            all_price_frames.keys(),
            fresh=fresh,
            cache_only=False,
            period=config.training_history_period,
        )
        benchmark_history = fetcher.download_history(
            config.benchmark_ticker,
            interval="1d",
            period=config.training_history_period,
            ttl_seconds=config.cache_ttls.market_history,
            fresh=fresh,
            cache_only=False,
        )
        breadth_history = fetcher.calculate_breadth_history(training_frames)
        training_report = xgb.fit(
            training_frames,
            save_model=True,
            sector_map=sector_by_ticker,
            sector_histories=fetcher.fetch_sector_history_for_period(config.training_history_period, fresh=fresh),
            vix_history=fetcher.fetch_macro_history(config.vix_ticker, fresh=fresh),
            earnings_dates_map=fetcher.fetch_earnings_dates_for_tickers(training_frames.keys(), fresh=fresh),
            benchmark_history=benchmark_history,
            breadth_history=breadth_history,
            put_call_history=fetcher.fetch_macro_history(config.put_call_ticker, fresh=fresh),
            hyg_history=fetcher.fetch_macro_history(config.hyg_ticker, fresh=fresh),
            lqd_history=fetcher.fetch_macro_history(config.lqd_ticker, fresh=fresh),
            tlt_history=fetcher.fetch_macro_history(config.tlt_ticker, fresh=fresh),
        )
    if config.feature_flags.anomaly_detection:
        anomaly_detector.fit(all_price_frames)
    feedback_rows = paper_tracker.completed_payload_rows(paper=True, weeks=8)
    feedback_summary = xgb.apply_feedback(xgb.feedback_rows_from_payloads(feedback_rows))
    for signal_name, multiplier in xgb.feature_health.broad_weight_adjustments().items():
        if signal_name in adaptive_weights:
            adaptive_weights[signal_name] *= multiplier
    adaptive_weight_total = sum(adaptive_weights.values()) or 1.0
    adaptive_weights = {
        signal_name: weight / adaptive_weight_total
        for signal_name, weight in adaptive_weights.items()
    }
    model_training_samples = getattr(xgb, "training_samples", 0)
    model_trained = model_training_samples >= config.thresholds.cold_start_min_samples
    config.model_trained = model_trained
    threshold_used = resolve_active_threshold(config, macro_snapshot, threshold_override)
    quality_summary = summarize_data_quality(config, universe)

    contexts = build_candidate_contexts(
        config=config,
        universe=universe,
        fetcher=fetcher,
        sentiment_scores=sentiment_scores,
        flow_scores=flow_scores,
        rs_scores=rs_scores,
        news_classifier=news_classifier,
        congress_tracker=congress_tracker,
        sector_impact_engine=sector_impact_engine,
        macro_analyzer=macro_analyzer,
        macro_snapshot=macro_snapshot,
        price_frames=all_price_frames,
        anomaly_detector=anomaly_detector,
        momentum_watchlist=momentum_watchlist,
        float_rotation_bonus_map=float_rotation_bonus_map,
        fresh=fresh,
    )

    pattern_analyzer = PatternWinRateAnalyzer()
    pattern_history_map: Dict[tuple[str, str], PatternWinRateResult] = {}
    unique_patterns = {(ctx.sector, ctx.pattern_name) for ctx in contexts.values() if ctx.pattern_name != "none"}
    for sector, pattern_name in unique_patterns:
        if not config.feature_flags.historical_pattern_win_rate:
            pattern_history_map[(sector, pattern_name)] = PatternWinRateResult(
                pattern_name=pattern_name,
                win_rate=0.65,
                sample_size=50,
                qualified=True,
            )
            continue
        pattern_history_map[(sector, pattern_name)] = pattern_analyzer.analyze(
            pattern_name,
            sector,
            config.sector_universe,
            all_price_frames,
            current_regime=macro_snapshot.risk_regime,
            pattern_threshold=config.thresholds.pattern_min,
        )

    candidates = score_candidate_contexts(
        config=config,
        contexts=contexts,
        pattern_history_map=pattern_history_map,
        xgb=xgb,
        adaptive_weights=adaptive_weights,
        model_training_samples=model_training_samples,
        threshold_used=threshold_used,
    )

    ranked = sorted(candidates.values(), key=lambda item: item.final_score, reverse=True)
    if config.feature_flags.gpt_news_reasoning:
        for candidate in ranked[: config.gpt_reasoning_top_n]:
            context = contexts[candidate.ticker]
            reasoning = gpt_reasoner.reason_about_news(candidate.ticker, context.headlines, fresh=fresh)
            candidates[candidate.ticker] = score_candidate_context(
                config=config,
                context=context,
                pattern_history=pattern_history_map.get(
                    (context.sector, context.pattern_name),
                    PatternWinRateResult(context.pattern_name, 0.65, 0, False),
                ),
                gpt_reasoning=reasoning,
                xgb=xgb,
                adaptive_weights=adaptive_weights,
                model_training_samples=model_training_samples,
                threshold_used=threshold_used,
            )

    final_candidates = list(candidates.values())
    if config.feature_flags.pick_quality_tiers:
        apply_quality_tiers(final_candidates)
    if debug:
        emit_debug_scores(final_candidates)
    selection_warning = build_selection_warning(config, macro_snapshot, benchmark)
    selected = select_portfolio(
        config,
        final_candidates,
        price_frames,
        threshold_used=threshold_used,
        top_n=config.top_n,
        vix=macro_snapshot.vix,
        benchmark=benchmark,
    )
    report_candidates = build_report_candidates(
        config,
        selected=selected,
        candidates=final_candidates,
        benchmark=benchmark,
        top_n=config.top_n,
        vix=macro_snapshot.vix,
    )
    if not selected and report_candidates:
        selection_warning = (
            f"{selection_warning} "
            "No picks passed the active regime filters; showing near-miss watchlist only."
        ).strip()
    backtest.record_predictions(run_id, selected)
    if config.feature_flags.live_performance_tracker or paper_trade:
        paper_tracker.record_paper_predictions(run_id, selected)
    if config.feature_flags.signal_attribution_tracker:
        attribution_tracker.record_predictions(run_id, selected)
    backtest.record_regime_context(
        run_id,
        vix=macro_snapshot.vix,
        spy_week_return=trailing_return_safe(benchmark, periods=5),
        sector_leaders=macro_snapshot.top_sectors,
    )
    if config.feature_flags.live_performance_tracker or paper_trade:
        paper_tracker.record_regime_context(
            run_id,
            vix=macro_snapshot.vix,
            spy_week_return=trailing_return_safe(benchmark, periods=5),
            sector_leaders=macro_snapshot.top_sectors,
        )

    macro_summary = build_macro_summary(macro_snapshot)
    qualified_count = sum(candidate.meets_threshold for candidate in final_candidates)
    suggestion = suggest_lower_threshold(final_candidates, threshold_used, config.top_n)
    weekly_hit_rate = backtest.weekly_hit_rate()
    last_week_accuracy = None
    if not weekly_hit_rate.empty:
        last_week_accuracy = float(weekly_hit_rate["hit_target"].iloc[-1] * 100)
    rolling_paper_stats = paper_tracker.paper_trade_summary()
    regime_memory = paper_tracker.regime_memory_summary(
        vix=macro_snapshot.vix,
        spy_week_return=trailing_return_safe(benchmark, periods=5),
        sector_leaders=macro_snapshot.top_sectors,
    )
    apply_kelly_sizing(final_candidates, rolling_paper_stats)
    pdf_path = None
    cache_saved_seconds = round(stage1_prefilter.cache_hit_rate * 480.0, 0)
    scan_summary = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "universe_total": len(all_tickers),
        "stage1_survivors": len(survivor_tickers),
        "stage1_runtime_seconds": round(stage1_elapsed, 2),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "vix": round(macro_snapshot.vix, 2),
        "regime": adaptive_snapshot.label,
        "macro_mode": macro_snapshot.risk_regime,
        "model_training_samples": model_training_samples,
        "model_auc": round(training_report.auc, 3) if training_report else 0.0,
        "model_family": training_report.model_family if training_report else "AdaptiveRegimeEnsemble",
        "qualified_count": qualified_count,
        "threshold_used": threshold_used,
        "top_sector": macro_snapshot.top_sectors[0] if macro_snapshot.top_sectors else "Unknown",
        "worst_sector": macro_snapshot.bottom_sectors[0] if macro_snapshot.bottom_sectors else "Unknown",
        "cache_warm": stage1_prefilter.warm_tickers,
        "cache_total": stage1_prefilter.total_tickers,
        "cache_hit_rate": round(stage1_prefilter.cache_hit_rate * 100.0, 1),
        "cache_saved_seconds_estimate": cache_saved_seconds,
        "spy_week_return": round(trailing_return_safe(benchmark, periods=5) * 100.0, 2),
        "selection_warning": selection_warning,
        "regime_memory_text": format_regime_memory_summary(regime_memory),
        "adaptive_feedback": feedback_summary,
    }

    payload = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "macro": asdict(macro_snapshot),
        "macro_summary": macro_summary,
        "regime_label": macro_snapshot.risk_regime,
        "adaptive_regime_label": adaptive_snapshot.label,
        "threshold_used": threshold_used,
        "qualified_count": qualified_count,
        "model_trained": model_trained,
        "model_training_samples": model_training_samples,
        "training_report": training_report.to_dict() if training_report else None,
        "scan_summary": scan_summary,
        "stage1_prefilter": stage1_prefilter.to_dict(),
        "momentum_watchlist": momentum_watchlist.to_dict() if momentum_watchlist else {},
        "data_quality_summary": quality_summary,
        "suggested_threshold": suggestion,
        "adaptive_weights": adaptive_weights,
        "candidates": [candidate.to_dict() for candidate in final_candidates],
        "selected": [candidate.to_dict() for candidate in selected],
        "display_candidates": [candidate.to_dict() for candidate in report_candidates],
        "backtest_rows": backtest.performance_frame().to_dict(orient="records"),
        "weight_history": backtest.weight_history_frame().to_dict(orient="records"),
        "paper_trade_summary": rolling_paper_stats,
        "signal_attribution_summary": attribution_tracker.signal_summary(),
        "terminal_report": render_terminal_report(
            report_candidates,
            threshold_used=threshold_used,
            macro_summary=macro_summary,
            regime_label=macro_snapshot.risk_regime,
            model_trained=model_trained,
            qualified_count=qualified_count,
            candidate_pool_size=len(final_candidates),
            data_quality_summary=quality_summary,
            suggestion=suggestion,
            summary_header=scan_summary,
        ),
        "pdf_report": None,
    }
    footer_lines = []
    if rolling_paper_stats:
        footer_lines.append(
            f"Last 8 weeks: {rolling_paper_stats.get('target_hit_rate', rolling_paper_stats.get('hit_rate', 0)):.0f}% target hit rate | "
            f"{rolling_paper_stats.get('positive_return_rate', 0):.0f}% positive-return rate | "
            f"{rolling_paper_stats.get('average_return', 0):+.1f}% avg return"
        )
    save_json(config.latest_scan_path, payload)
    def _run_exports() -> None:
        nonlocal pdf_path
        export_reports(
            report_candidates,
            config.latest_report_csv,
            config.latest_report_md,
            config.latest_report_html,
            threshold_used=threshold_used,
            macro_summary=macro_summary,
            regime_label=macro_snapshot.risk_regime,
            model_trained=model_trained,
            qualified_count=qualified_count,
            candidate_pool_size=len(final_candidates),
            data_quality_summary=quality_summary,
            suggestion=suggestion,
            summary_header=scan_summary,
        )
        if config.feature_flags.pdf_report_generator:
            pdf_path = generate_pdf_report(config, report_candidates, macro_summary, last_week_accuracy)
        if send_notifications:
            send_alerts(
                config,
                selected,
                alert_email=alert_email,
                alert_phone=alert_phone,
                summary_header=scan_summary,
                footer_lines=footer_lines,
            )
        if send_notifications and config.feature_flags.bot_integration:
            send_weekly_picks(config, selected)

    threading.Thread(target=_run_exports).start()
    payload["export_status"] = "background"
    return payload


def _resolve_ticker_sector(
    fetcher: MarketDataFetcher,
    universe_map: Dict[str, list[str]],
    ticker: str,
    *,
    fresh: bool,
) -> str:
    for sector, tickers in universe_map.items():
        if ticker in tickers:
            return sector
    info = fetcher.fetch_info(ticker, fresh=fresh)
    return str(info.get("sector") or info.get("industryDisp") or info.get("industry") or "Unknown")


def _single_candidate_eligibility(
    config,
    candidate: CandidateScore,
    *,
    threshold_used: float,
    vix: float,
    benchmark: pd.DataFrame,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if candidate.final_score < threshold_used:
        reasons.append(f"Below current threshold ({threshold_used:.1f})")
    if candidate.confluence_count < 3:
        reasons.append("Fewer than 3/5 core signals aligned")
    if candidate.risk_reward < 1.0:
        reasons.append("Risk/reward below 1.0")
    if vix > config.thresholds.kill_switch_vix:
        reasons.append(f"VIX above kill switch ({config.thresholds.kill_switch_vix:.1f})")
    elif vix > 25.0:
        if candidate.final_score < 70.0:
            reasons.append("High-VIX rule requires score >= 70")
        if candidate.ml_score < 70.0:
            reasons.append("High-VIX rule requires ML >= 70")
        if "TIER 1" not in candidate.tier_label:
            reasons.append("High-VIX rule requires Tier 1 quality")
    if is_defensive_market_regime(benchmark):
        allowed = {"healthcare", "consumer defensive", "utilities"}
        if _normalized_sector_name(candidate.sector) not in allowed:
            reasons.append("Current market downtrend only allows defensive sectors")
    return (not reasons), reasons


def run_single_ticker_analysis(
    ticker: str,
    *,
    fresh: bool = False,
    universe_mode: str | None = None,
    threshold_override: float | None = None,
) -> Dict[str, object]:
    """Run the full scoring stack for one ticker and persist the latest result."""

    configure_quiet_runtime_output()
    config = get_config()
    setup_logging(config.predictions_log)
    started = time.perf_counter()
    ticker = ticker.upper().strip()
    if not ticker:
        raise ValueError("Ticker is required for single-stock analysis.")

    cache = SQLiteCache(config.cache_db)
    fetcher = MarketDataFetcher(config, cache)
    sentiment_engine = SentimentEngine(config, cache)
    flow_detector = FlowDetector(config, cache, fetcher)
    macro_analyzer = MacroRegimeAnalyzer(config, fetcher)
    news_classifier = NewsCatalystClassifier(config)
    gpt_reasoner = GPTNewsReasoner(config, cache)
    congress_tracker = CongressTradeTracker(config, cache)
    sector_impact_engine = SectorImpactEngine(config, fetcher)
    anomaly_detector = AnomalyDetector()

    resolved_mode = universe_mode or config.default_universe
    universe_map = fetcher.resolve_universe(mode=resolved_mode, fresh=fresh)
    all_tickers = sorted({symbol for sector_tickers in universe_map.values() for symbol in sector_tickers})
    sector_by_ticker = {
        symbol: sector
        for sector, tickers in universe_map.items()
        for symbol in tickers
    }
    benchmark = fetcher.download_history(
        config.benchmark_ticker,
        interval="1d",
        period=config.daily_period,
        ttl_seconds=config.cache_ttls.market_history,
        fresh=fresh,
        cache_only=not fresh,
    )
    if benchmark.empty:
        benchmark = fetcher.download_history(
            config.benchmark_ticker,
            interval="1d",
            period=config.daily_period,
            ttl_seconds=config.cache_ttls.market_history,
            fresh=fresh,
            cache_only=False,
        )

    universe_daily_frames = fetcher.fetch_daily_frames_for_tickers(
        all_tickers,
        fresh=fresh,
        cache_only=not fresh,
    )
    if ticker not in universe_daily_frames or universe_daily_frames[ticker].empty:
        universe_daily_frames[ticker] = fetcher.download_history(
            ticker,
            interval="1d",
            period=config.daily_period,
            ttl_seconds=config.cache_ttls.market_history,
            fresh=fresh,
            cache_only=not fresh,
        )
        if universe_daily_frames[ticker].empty:
            universe_daily_frames[ticker] = fetcher.download_history(
                ticker,
                interval="1d",
                period=config.daily_period,
                ttl_seconds=config.cache_ttls.market_history,
                fresh=fresh,
                cache_only=False,
            )
    sector = sector_by_ticker.get(ticker) or _resolve_ticker_sector(fetcher, universe_map, ticker, fresh=fresh)
    macro_snapshot = macro_analyzer.build_snapshot(
        fresh=fresh,
        price_frames=universe_daily_frames,
    )
    sector_impact_engine.shared_benchmark_return = trailing_return_safe(benchmark, periods=5)

    bundle_map = fetcher.fetch_selected_bundles(
        {ticker: sector},
        daily_frames=universe_daily_frames,
        fresh=fresh,
    )
    bundle = bundle_map.get(ticker)
    if bundle is None or bundle.daily.empty:
        raise RuntimeError(f"No price history was available for {ticker}.")

    price_frames = {
        symbol: frame
        for symbol, frame in universe_daily_frames.items()
        if frame is not None and not frame.empty
    }
    price_frames[ticker] = bundle.daily
    momentum_watchlist = (
        build_momentum_watchlist(
            benchmark=benchmark,
            bundles={
                symbol: SimpleNamespace(
                    daily=frame,
                    sector=sector_by_ticker.get(symbol, "Unknown"),
                )
                for symbol, frame in price_frames.items()
            },
            path=config.momentum_watchlist_path,
        )
        if config.feature_flags.momentum_watchlist
        else None
    )

    summary = fetcher.summarize_company(bundle)
    if summary["price"] < config.thresholds.min_price:
        raise RuntimeError(f"{ticker} is below the app minimum price filter (${config.thresholds.min_price:.2f}).")
    if summary["average_volume"] < config.thresholds.min_average_volume:
        raise RuntimeError(
            f"{ticker} is below the app minimum average-volume filter ({config.thresholds.min_average_volume:,.0f} shares)."
        )

    latest_price = summary["price"] or float(bundle.daily["close"].iloc[-1])
    sentiment_scores = sentiment_engine.score_sentiment([ticker], fresh=fresh)
    flow_scores = flow_detector.score_universe([ticker], {ticker: latest_price}, fresh=fresh)
    rs_scores = compute_rs_scores(price_frames, benchmark)
    float_rotation_bonus_map = compute_float_rotation_bonuses(config, {ticker: bundle})
    threshold_used = resolve_active_threshold(config, macro_snapshot, threshold_override)

    xgb = AdaptivePredictor(config)
    xgb.active_breadth_percentile = macro_snapshot.breadth_percentile
    adaptive_snapshot = xgb.regime_classifier.build_snapshot(
        price_frames=price_frames,
        benchmark_history=benchmark,
        breadth_history=fetcher.calculate_breadth_history(price_frames),
        sector_histories=fetcher.fetch_sector_history_for_period(config.daily_period, fresh=fresh),
        vix_history=fetcher.fetch_macro_history(config.vix_ticker, fresh=fresh),
        put_call_history=fetcher.fetch_macro_history(config.put_call_ticker, fresh=fresh),
        hyg_history=fetcher.fetch_macro_history(config.hyg_ticker, fresh=fresh),
        lqd_history=fetcher.fetch_macro_history(config.lqd_ticker, fresh=fresh),
        tlt_history=fetcher.fetch_macro_history(config.tlt_ticker, fresh=fresh),
    )
    xgb.active_regime = adaptive_snapshot.label
    xgb.active_market_snapshot = adaptive_snapshot
    training_report = xgb.last_report
    model_training_samples = int((training_report.training_samples if training_report else 0) or 0)
    adaptive_weights = config.signal_weights.as_dict()
    for signal_name, multiplier in xgb.feature_health.broad_weight_adjustments().items():
        if signal_name in adaptive_weights:
            adaptive_weights[signal_name] *= multiplier
    adaptive_weight_total = sum(adaptive_weights.values()) or 1.0
    adaptive_weights = {
        signal_name: weight / adaptive_weight_total
        for signal_name, weight in adaptive_weights.items()
    }

    context = _build_candidate_context(
        config,
        ticker,
        bundle,
        fetcher,
        sentiment_scores.get(ticker, {}),
        flow_scores.get(ticker, {}),
        rs_scores.get(ticker, {}),
        news_classifier,
        congress_tracker,
        sector_impact_engine,
        macro_analyzer,
        macro_snapshot,
        price_frames,
        anomaly_detector,
        momentum_watchlist,
        float_rotation_bonus_map,
        fresh,
    )
    if context is None:
        raise RuntimeError(f"{ticker} did not pass the app's minimum liquidity or data-quality checks.")

    pattern_history = PatternWinRateAnalyzer().analyze(
        context.pattern_name,
        context.sector,
        config.sector_universe,
        price_frames,
        current_regime=macro_snapshot.risk_regime,
        pattern_threshold=config.thresholds.pattern_min,
    )
    reasoning = (
        gpt_reasoner.reason_about_news(ticker, context.headlines, fresh=fresh)
        if config.feature_flags.gpt_news_reasoning
        else {"score": 50.0, "reason": ""}
    )
    candidate = score_candidate_context(
        config=config,
        context=context,
        pattern_history=pattern_history,
        gpt_reasoning=reasoning,
        xgb=xgb,
        adaptive_weights=adaptive_weights,
        model_training_samples=model_training_samples,
        threshold_used=threshold_used,
    )
    if config.feature_flags.pick_quality_tiers:
        apply_quality_tiers([candidate])

    eligible, reasons = _single_candidate_eligibility(
        config,
        candidate,
        threshold_used=threshold_used,
        vix=macro_snapshot.vix,
        benchmark=benchmark,
    )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "universe_mode": resolved_mode,
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "regime_label": macro_snapshot.risk_regime,
        "macro_summary": build_macro_summary(macro_snapshot),
        "threshold_used": threshold_used,
        "officially_eligible": eligible,
        "eligibility_reasons": reasons,
        "scan_summary": {
            "regime": macro_snapshot.risk_regime,
            "vix": round(macro_snapshot.vix, 2),
            "spy_week_return": round(trailing_return_safe(benchmark, periods=5) * 100.0, 2),
            "breadth_percentile": round(macro_snapshot.breadth_percentile * 100.0, 1),
            "top_sector": macro_snapshot.top_sectors[0] if macro_snapshot.top_sectors else "Unknown",
            "worst_sector": macro_snapshot.bottom_sectors[0] if macro_snapshot.bottom_sectors else "Unknown",
        },
        "candidate": candidate.to_dict(),
    }
    save_json(config.latest_single_analysis_path, payload)
    update_single_analysis_history(config, payload)
    return payload


def build_candidate_contexts(
    *,
    config,
    universe,
    fetcher,
    sentiment_scores,
    flow_scores,
    rs_scores,
    news_classifier,
    congress_tracker,
    sector_impact_engine,
    macro_analyzer,
    macro_snapshot,
    price_frames,
    anomaly_detector,
    momentum_watchlist,
    float_rotation_bonus_map,
    fresh,
) -> Dict[str, CandidateContext]:
    """Build candidate contexts in parallel for the current universe."""

    tasks = {}
    contexts: Dict[str, CandidateContext] = {}
    max_workers = config.max_parallel_tickers if config.feature_flags.parallel_processing else 1
    priority_tickers = rank_priority_tickers(universe.keys(), momentum_watchlist)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for ticker in priority_tickers:
            bundle = universe[ticker]
            tasks[
                executor.submit(
                    _build_candidate_context,
                    config,
                    ticker,
                    bundle,
                    fetcher,
                    sentiment_scores.get(ticker, {}),
                    flow_scores.get(ticker, {}),
                    rs_scores.get(ticker, {}),
                    news_classifier,
                    congress_tracker,
                    sector_impact_engine,
                    macro_analyzer,
                    macro_snapshot,
                    price_frames,
                    anomaly_detector,
                    momentum_watchlist,
                    float_rotation_bonus_map,
                    fresh,
                )
            ] = ticker
        if config.progress_bar and tqdm is not None:
            progress = tqdm(total=len(tasks), desc="Scoring tickers", dynamic_ncols=True)
        else:
            progress = None
        for future in as_completed(tasks):
            ticker = tasks[future]
            try:
                context = future.result()
            except Exception:
                LOGGER.exception("Failed to score ticker %s", ticker)
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix_str(f"failed:{ticker}")
                continue
            if context is not None:
                contexts[context.ticker] = context
            if progress is not None:
                progress.update(1)
                progress.set_postfix_str(ticker)
        if progress is not None:
            progress.close()
    return contexts


def _build_candidate_context(
    config,
    ticker,
    bundle,
    fetcher,
    sentiment_metrics,
    options_metrics,
    rs_metrics,
    news_classifier,
    congress_tracker,
    sector_impact_engine,
    macro_analyzer,
    macro_snapshot,
    price_frames,
    anomaly_detector,
    momentum_watchlist,
    float_rotation_bonus_map,
    fresh,
):
    summary = fetcher.summarize_company(bundle)
    if summary["price"] < config.thresholds.min_price:
        return None
    if summary["average_volume"] < config.thresholds.min_average_volume:
        return None
    if config.feature_flags.data_quality_validator:
        quality = validate_ticker_data(
            info=bundle.info,
            daily=bundle.daily,
            hourly=bundle.hourly,
            minimum_score=config.thresholds.minimum_data_quality_score,
        )
        if not quality.is_valid:
            LOGGER.info("Skipping %s due to data quality: %s", ticker, "; ".join(quality.issues))
            return None
    else:
        quality = DataQualityResult(score=1.0, is_valid=True, issues=[])
    recent_news = filter_recent_news(bundle.news, days=7)
    sec_filings = fetcher.fetch_sec_filings(ticker, fresh=fresh)
    earnings_dates = fetcher.fetch_earnings_dates(ticker, fresh=fresh)
    pattern_suite = detect_patterns(bundle.daily)
    pattern = pattern_suite["best_pattern"]
    macro_sector_score = macro_analyzer.sector_score(bundle.sector, macro_snapshot)
    if config.feature_flags.multi_timeframe_confirmation:
        multi_timeframe = evaluate_multi_timeframe_alignment(
            bundle.daily,
            bundle.hourly,
            partial_penalty=config.thresholds.multi_timeframe_partial_penalty,
        )
    else:
        multi_timeframe = MultiTimeframeResult(
            qualifies=True,
            contradicts=False,
            agreement_count=3,
            penalty_factor=1.0,
            directions={"daily": "bullish", "4h": "bullish", "1h": "bullish"},
            scores={"daily": 0.5, "4h": 0.5, "1h": 0.5},
            summary="Multi-timeframe confirmation disabled.",
        )
    news_metrics = news_classifier.score(ticker, recent_news, sec_filings)
    institutional_metrics = compute_institutional_metrics(bundle.daily, bundle.info, sec_filings)
    if config.feature_flags.smart_money_divergence:
        smart_money = evaluate_smart_money_divergence(
            sentiment_metrics.get("sentiment_score", 0.0),
            options_metrics.get("options_score", 0.0),
            institutional_metrics.get("institutional_score", 0.0),
            bearish_flow_ratio=options_metrics.get("bearish_flow_ratio", 0.0),
        )
    else:
        smart_money = SmartMoneyResult(
            score=0.5,
            adjustment_points=0.0,
            reject=False,
            label="disabled",
            summary="Smart-money divergence disabled.",
        )
    pre_earnings_metrics = compute_pre_earnings_drift(
        bundle.daily,
        earnings_dates,
        bonus_points=config.thresholds.pre_earnings_runner_bonus_points,
    )
    if config.feature_flags.anomaly_detection:
        anomaly_result = anomaly_detector.score(bundle.daily)
    else:
        anomaly_result = AnomalyResult(
            score=0.0,
            is_anomaly=False,
            direction_aligned=False,
            adjustment_multiplier=1.0,
            summary="Anomaly detection disabled.",
        )
    if config.feature_flags.supply_chain_tracker:
        supply_chain_signal = compute_supply_chain_signal(ticker, price_frames)
    else:
        supply_chain_signal = SupplyChainSignal(
            score=0.0,
            related_movers=[],
            summary="Supply-chain tracker disabled.",
        )
    if config.feature_flags.congress_tracker:
        congress_signal = congress_tracker.score_ticker(ticker, fresh=fresh)
    else:
        congress_signal = CongressSignal(
            score=0.5,
            recent_buys=0,
            recent_sells=0,
            summary="Congress tracker disabled.",
        )
    if config.feature_flags.weather_commodity_engine:
        sector_impact = sector_impact_engine.score_sector(bundle.sector, fresh=fresh)
    else:
        sector_impact = SectorImpactResult(
            points=0.0,
            normalized_score=0.5,
            summary="Sector impact engine disabled.",
        )
    if config.feature_flags.short_squeeze_probability:
        squeeze_result = calculate_squeeze_probability(bundle.info, bundle.daily)
    else:
        squeeze_result = SqueezeResult(
            score=0.0,
            qualifying=False,
            summary="Short-squeeze probability disabled.",
        )
    if config.feature_flags.breakout_confirmation_engine:
        breakout_result = confirm_breakout(bundle.daily, config.thresholds.confirmed_breakout_bonus_points)
    else:
        breakout_result = BreakoutResult(0.0, False, False, False, False, "", 0.0, "Breakout engine disabled.")
    if config.feature_flags.relative_volume_alert_system:
        relative_volume_result = detect_relative_volume_alert(
            bundle.hourly,
            config.thresholds.unusual_early_volume_bonus_points,
        )
    else:
        relative_volume_result = RelativeVolumeResult(0.0, False, 0.0, "Relative volume alert disabled.")
    if config.feature_flags.sector_etf_signal_booster and bundle.sector in macro_snapshot.top_sectors:
        sector_temperature_bonus = config.thresholds.hot_sector_bonus_points
        sector_temperature_tag = "🔥 HOT SECTOR"
    elif config.feature_flags.sector_etf_signal_booster and bundle.sector in macro_snapshot.bottom_sectors:
        sector_temperature_bonus = -config.thresholds.cold_sector_penalty_points
        sector_temperature_tag = "❄️ COLD SECTOR"
    else:
        sector_temperature_bonus = 0.0
        sector_temperature_tag = ""
    return CandidateContext(
        ticker=ticker,
        sector=bundle.sector,
        info=bundle.info,
        daily_frame=bundle.daily,
        hourly_frame=bundle.hourly,
        earnings_dates=earnings_dates,
        macro_sector_score=macro_sector_score,
        pattern_name=pattern.name,
        pattern_score=pattern.score,
        pattern_result=pattern,
        options_metrics=options_metrics,
        sentiment_metrics=sentiment_metrics,
        rs_metrics=rs_metrics,
        news_metrics=news_metrics,
        institutional_metrics=institutional_metrics,
        pre_earnings_metrics=pre_earnings_metrics,
        multi_timeframe=multi_timeframe,
        smart_money=smart_money,
        anomaly_result=anomaly_result,
        supply_chain_signal=supply_chain_signal,
        congress_signal=congress_signal,
        sector_impact=sector_impact,
        squeeze_result=squeeze_result,
        breakout_result=breakout_result,
        relative_volume_result=relative_volume_result,
        sector_temperature_bonus=sector_temperature_bonus,
        sector_temperature_tag=sector_temperature_tag,
        persistent_momentum_bonus=(
            config.thresholds.persistent_momentum_bonus_points
            if momentum_watchlist and ticker in set(momentum_watchlist.persistent_tickers)
            else 0.0
        ),
        float_rotation_bonus=float_rotation_bonus_map.get(ticker, 0.0),
        data_quality=quality,
        headlines=[item.get("title", "") for item in recent_news[:5]],
    )


def score_candidate_context(
    *,
    config,
    context: CandidateContext,
    pattern_history: PatternWinRateResult,
    gpt_reasoning: dict,
    xgb,
    adaptive_weights: Dict[str, float],
    model_training_samples: int,
    threshold_used: float,
) -> CandidateScore:
    """Build a final candidate score from a precomputed context."""

    days_until_earnings = float(context.pre_earnings_metrics.get("days_until_earnings", 0.0) or 0.0)
    xgb_output = xgb.predict_proba(
        context.daily_frame,
        sector_rs_rank=context.macro_sector_score * 100.0,
        earnings_dates=context.earnings_dates,
        earnings_proximity_score=max(0.0, 1.0 - (days_until_earnings / 30.0)) if days_until_earnings > 0 else 0.0,
        breadth_percentile=getattr(xgb, "active_breadth_percentile", 0.5),
        regime=getattr(xgb, "active_regime", None),
    )
    ensemble = build_tree_ensemble_output(
        probability=xgb_output.probability,
        primary_probability=xgb_output.xgb_probability or xgb_output.probability,
        secondary_probability=xgb_output.lightgbm_probability,
        model_status=xgb_output.status,
        confidence_label=xgb_output.confidence_label,
        model_family=getattr(xgb, "model_family", "AdaptiveRegimeEnsemble"),
        blend_weights=xgb_output.blend_weights,
        model_spread_override=xgb_output.model_spread,
        score_uncertainty_override=xgb_output.score_uncertainty,
    )
    candidate = build_candidate_score(
        config=config,
        ticker=context.ticker,
        sector=context.sector,
        info=context.info,
        daily_frame=context.daily_frame,
        hourly_frame=context.hourly_frame,
        pattern=context.pattern_result,
        pattern_history=pattern_history,
        ensemble_output=ensemble,
        options_metrics=context.options_metrics,
        sentiment_metrics=context.sentiment_metrics,
        rs_metrics=context.rs_metrics,
        macro_sector_score=context.macro_sector_score,
        news_metrics=context.news_metrics,
        gpt_reasoning=gpt_reasoning,
        institutional_metrics=context.institutional_metrics,
        pre_earnings_metrics=context.pre_earnings_metrics,
        multi_timeframe=context.multi_timeframe,
        smart_money=context.smart_money,
        anomaly_result=context.anomaly_result,
        supply_chain_signal=context.supply_chain_signal,
        congress_signal=context.congress_signal,
        sector_impact=context.sector_impact,
        squeeze_result=context.squeeze_result,
        breakout_result=context.breakout_result,
        relative_volume_result=context.relative_volume_result,
        data_quality=context.data_quality,
        weights=adaptive_weights,
        model_training_samples=model_training_samples,
        threshold_used=threshold_used,
        sector_temperature_bonus=context.sector_temperature_bonus,
        sector_temperature_tag=context.sector_temperature_tag,
        persistent_momentum_bonus=context.persistent_momentum_bonus,
        float_rotation_bonus=context.float_rotation_bonus,
    )
    if not config.feature_flags.confidence_interval_display:
        candidate.score_low = candidate.final_score
        candidate.score_high = candidate.final_score
        candidate.score_uncertainty = 0.0
        candidate.confidence_label = "disabled"
    return candidate


def score_candidate_contexts(
    *,
    config,
    contexts: Dict[str, CandidateContext],
    pattern_history_map: Dict[tuple[str, str], PatternWinRateResult],
    xgb,
    adaptive_weights: Dict[str, float],
    model_training_samples: int,
    threshold_used: float,
) -> Dict[str, CandidateScore]:
    if not contexts:
        return {}
    results: Dict[str, CandidateScore] = {}
    tasks = {}
    max_workers = config.max_parallel_tickers if config.feature_flags.parallel_processing else 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for ticker, context in contexts.items():
            tasks[
                executor.submit(
                    score_candidate_context,
                    config=config,
                    context=context,
                    pattern_history=pattern_history_map.get(
                        (context.sector, context.pattern_name),
                        PatternWinRateResult(context.pattern_name, 0.65, 0, False),
                    ),
                    gpt_reasoning={"score": 50.0, "reason": ""},
                    xgb=xgb,
                    adaptive_weights=adaptive_weights,
                    model_training_samples=model_training_samples,
                    threshold_used=threshold_used,
                )
            ] = ticker
        if config.progress_bar and tqdm is not None:
            progress = tqdm(total=len(tasks), desc="Final scoring", dynamic_ncols=True)
        else:
            progress = None
        for future in as_completed(tasks):
            ticker = tasks[future]
            try:
                results[ticker] = future.result()
            except Exception:
                LOGGER.exception("Failed final score build for %s", ticker)
            if progress is not None:
                progress.update(1)
                progress.set_postfix_str(ticker)
        if progress is not None:
            progress.close()
    return results


def build_macro_summary(macro_snapshot) -> str:
    """Create a short macro summary for reports and the dashboard."""

    return (
        f"Regime {macro_snapshot.risk_regime}; VIX {macro_snapshot.vix:.2f}; "
        f"top sectors: {', '.join(macro_snapshot.top_sectors)}; "
        f"DXY 5-day return {macro_snapshot.dxy_5d_return:.2%}; "
        f"2s/10s proxy spread {macro_snapshot.yield_spread_proxy:.2f}; "
        f"breadth {macro_snapshot.breadth_percentile:.0%} above 50DMA."
    )


def trailing_return_safe(frame: pd.DataFrame, periods: int = 5) -> float:
    if frame.empty or len(frame) <= periods:
        return 0.0
    close = frame["close"]
    return float(close.iloc[-1] / close.iloc[-periods - 1] - 1.0)


def resolve_active_threshold(config, macro_snapshot, override: float | None) -> float:
    if override is not None:
        return override
    regime = macro_snapshot.risk_regime
    if regime == "risk_on":
        return config.thresholds.risk_on_threshold
    if regime == "risk_off":
        return config.thresholds.risk_off_threshold
    return config.thresholds.neutral_threshold


def _normalized_sector_name(sector: str) -> str:
    normalized = (sector or "").strip().lower()
    aliases = {
        "consumer staples": "consumer defensive",
        "consumer defensive": "consumer defensive",
        "healthcare": "healthcare",
        "utilities": "utilities",
    }
    return aliases.get(normalized, normalized)


def is_defensive_market_regime(benchmark: pd.DataFrame) -> bool:
    if benchmark.empty or len(benchmark) < 50:
        return False
    close = benchmark["close"].dropna()
    if len(close) < 50:
        return False
    spy_5d = float(close.iloc[-1] / close.iloc[-6] - 1.0) if len(close) >= 6 else 0.0
    spy_above_50ma = bool(close.iloc[-1] > close.rolling(50).mean().iloc[-1])
    return spy_5d < -0.02 and not spy_above_50ma


def rebalance_stage1_survivors(
    config,
    prefilter: PrefilterResult,
    benchmark: pd.DataFrame,
    *,
    survivor_limit: int,
) -> list[str]:
    if not prefilter.survivors:
        return []
    if not config.feature_flags.defensive_stage1_rebalance or not is_defensive_market_regime(benchmark):
        return prefilter.survivors[:survivor_limit]

    defensive_aliases = {"healthcare", "consumer defensive", "utilities"}
    selected: list[str] = []
    seen: set[str] = set()
    primary_limit = max(survivor_limit - config.stage1_defensive_slots, survivor_limit // 2)
    for row in prefilter.top_rows:
        if len(selected) >= primary_limit:
            break
        if row.ticker in seen:
            continue
        selected.append(row.ticker)
        seen.add(row.ticker)

    per_sector = config.stage1_defensive_per_sector
    for sector_name in ["Healthcare", "Consumer Staples", "Consumer Defensive", "Utilities"]:
        added = 0
        normalized_target = _normalized_sector_name(sector_name)
        for row in prefilter.top_rows:
            if added >= per_sector:
                break
            if row.ticker in seen:
                continue
            if _normalized_sector_name(row.sector) != normalized_target:
                continue
            selected.append(row.ticker)
            seen.add(row.ticker)
            added += 1

    for row in prefilter.top_rows:
        if len(selected) >= survivor_limit:
            break
        if row.ticker in seen:
            continue
        if _normalized_sector_name(row.sector) in defensive_aliases:
            selected.append(row.ticker)
            seen.add(row.ticker)

    for row in prefilter.top_rows:
        if len(selected) >= survivor_limit:
            break
        if row.ticker in seen:
            continue
        selected.append(row.ticker)
        seen.add(row.ticker)
    return selected[:survivor_limit]


def build_selection_warning(config, macro_snapshot, benchmark: pd.DataFrame) -> str:
    warnings = []
    if macro_snapshot.vix > config.thresholds.kill_switch_vix:
        warnings.append("⛔ VIX > 30 — no picks this week. Wait for volatility to cool.")
    elif macro_snapshot.vix > 25.0:
        warnings.append("⚠️ High-VIX regime: only Tier 1 setups with 70%+ ML probability are allowed, max 5 picks.")
    if is_defensive_market_regime(benchmark):
        warnings.append("🛡️ Market downtrend confirmed: limiting picks to Healthcare, Consumer Defensive/Staples, and Utilities.")
    return " ".join(warnings)


def format_regime_memory_summary(summary: Dict[str, float | str | int] | None) -> str:
    if not summary:
        return ""
    return (
        f"In similar past conditions (VIX {summary.get('vix_bucket')}, SPY {summary.get('spy_trend')}), "
        f"historical target hit rate: {float(summary.get('target_hit_rate', summary.get('win_rate', 0.0))):.0f}% | "
        f"avg return: {float(summary.get('average_return', 0.0)):+.1f}%"
    )


def summarize_data_quality(config, universe) -> Dict[str, int]:
    total = len(universe)
    valid = 0
    full = 0
    for bundle in universe.values():
        result = validate_ticker_data(
            info=bundle.info,
            daily=bundle.daily,
            hourly=bundle.hourly,
            minimum_score=config.thresholds.minimum_data_quality_score,
        )
        if result.is_valid:
            valid += 1
        if not result.issues:
            full += 1
    return {"full_data": full, "valid_data": valid, "total": total}


def suggest_lower_threshold(candidates: list[CandidateScore], threshold_used: float, top_n: int) -> Dict[str, float | int] | None:
    if not candidates:
        return None
    suggested_threshold = max(0.0, threshold_used - 5.0)
    current = sum(candidate.final_score >= threshold_used for candidate in candidates)
    suggested = sum(candidate.final_score >= suggested_threshold for candidate in candidates)
    if suggested <= current:
        return None
    return {
        "threshold": round(suggested_threshold, 1),
        "additional_picks": suggested - current,
        "qualified_at_suggested": suggested,
        "top_n": top_n,
    }


def build_report_candidates(
    config,
    *,
    selected: list[CandidateScore],
    candidates: list[CandidateScore],
    benchmark: pd.DataFrame,
    top_n: int,
    vix: float,
) -> list[CandidateScore]:
    if selected:
        return selected
    ordered = sorted(candidates, key=lambda item: item.final_score, reverse=True)
    if not ordered:
        return []
    allowed_sectors = set()
    if is_defensive_market_regime(benchmark):
        allowed_sectors = {"healthcare", "consumer defensive", "utilities"}
    display_limit = min(top_n, 5) if vix > 25.0 else top_n
    near_misses: list[CandidateScore] = []
    for candidate in ordered:
        if candidate.final_score < config.thresholds.minimum_display_score:
            continue
        if candidate.confluence_count < 3:
            continue
        if candidate.risk_reward < 1.0:
            continue
        if allowed_sectors and _normalized_sector_name(candidate.sector) not in allowed_sectors:
            continue
        near_misses.append(candidate)
        if len(near_misses) >= display_limit:
            break
    if near_misses:
        return near_misses
    return [candidate for candidate in ordered if candidate.final_score >= config.thresholds.minimum_display_score][:display_limit]


def kelly_position_size(win_rate: float, avg_win: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 0.02
    reward_ratio = abs(avg_win / avg_loss)
    if reward_ratio <= 0:
        return 0.02
    kelly = win_rate - (1.0 - win_rate) / reward_ratio
    half_kelly = max(0.0, kelly / 2.0)
    return min(half_kelly, 0.10)


def apply_kelly_sizing(candidates: list[CandidateScore], summary: Dict[str, float | str | int] | None) -> None:
    if not candidates:
        return
    if not summary:
        for candidate in candidates:
            candidate.kelly_size_pct = 2.0
        return
    win_rate = float(summary.get("hit_rate", 0.0)) / 100.0
    avg_win = float(summary.get("average_win", 0.0)) / 100.0
    avg_loss = float(summary.get("average_loss", 0.0)) / 100.0
    base_size_pct = kelly_position_size(win_rate, avg_win, avg_loss) * 100.0
    tier_multipliers = {
        "🥇 TIER 1": 1.0,
        "🥈 TIER 2": 0.5,
        "🥉 TIER 3": 0.25,
    }
    for candidate in candidates:
        multiplier = tier_multipliers.get(candidate.tier_label, 0.25)
        suggested = max(1.0, base_size_pct * multiplier)
        candidate.kelly_size_pct = round(min(suggested, 10.0), 2)


def compute_float_rotation_bonuses(config, universe) -> Dict[str, float]:
    ratios = {}
    for ticker, bundle in universe.items():
        summary = MarketDataFetcher.summarize_company(bundle)
        shares_outstanding = (
            bundle.info.get("sharesOutstanding")
            or bundle.info.get("floatShares")
            or bundle.info.get("impliedSharesOutstanding")
            or 0.0
        )
        ratios[ticker] = float_rotation_ratio(
            summary["price"],
            summary["average_volume"],
            float(shares_outstanding or 0.0),
        )
    ordered = sorted(ratios.items(), key=lambda item: item[1], reverse=True)
    cutoff = max(int(len(ordered) * 0.2), 1)
    boosted = {ticker for ticker, ratio in ordered[:cutoff] if ratio >= 0.5}
    return {
        ticker: config.thresholds.float_rotation_bonus_points if ticker in boosted else 0.0
        for ticker in ratios
    }


def apply_quality_tiers(candidates: list[CandidateScore]) -> None:
    tier1_assigned = 0
    for candidate in sorted(candidates, key=lambda item: item.final_score, reverse=True):
        multiplier = 0.25
        tier = "🥉 TIER 3"
        if candidate.final_score >= 70 and tier1_assigned < 3:
            tier = "🥇 TIER 1"
            multiplier = 1.0
            tier1_assigned += 1
        elif candidate.final_score >= 60:
            tier = "🥈 TIER 2"
            multiplier = 0.5
        candidate.tier_label = tier
        candidate.position_size_pct = round(candidate.position_size_pct * multiplier, 2)


def emit_debug_scores(candidates: list[CandidateScore]) -> None:
    ordered = sorted(candidates, key=lambda item: item.final_score, reverse=True)
    for candidate in ordered:
        defaults = candidate.defaulted_signals
        print(
            (
                f"DEBUG {candidate.ticker} | Final {candidate.final_score:.2f} | "
                f"Tech {candidate.technical_score:.2f} | RS {candidate.rs_score:.2f} | "
                f"ML {candidate.ml_score:.2f} | Vol {candidate.volume_momentum_score:.2f} | "
                f"Sent {candidate.sentiment_score:.2f}{'*' if defaults.get('sentiment') else ''} | "
                f"Flow {candidate.options_score:.2f}{'*' if defaults.get('flow') else ''} | "
                f"Pattern {candidate.pattern_score:.2f}"
            )
        )
        subscores = candidate.diagnostics.get("subscores", {}).get("technical_signals", {})
        if subscores:
            print("  Technical signals:", json.dumps(subscores, sort_keys=True))


def _build_progress_summary(frame: pd.DataFrame, latest_prices: Dict[str, float]) -> str:
    if frame.empty:
        return "No active picks available."
    lines = []
    for row in frame.itertuples(index=False):
        latest = latest_prices.get(row.ticker, row.entry_price)
        current_return = latest / row.entry_price - 1 if row.entry_price else 0.0
        lines.append(f"{row.ticker}: {current_return:.2%} vs entry, target {row.target_price:.2f}")
    return "\n".join(lines)


def send_midweek_bot_update() -> None:
    config = get_config()
    if not config.feature_flags.bot_integration:
        return
    cache = SQLiteCache(config.cache_db)
    fetcher = MarketDataFetcher(config, cache)
    backtest = BacktestTracker(config.backtest_db)
    latest = backtest.latest_run_frame()
    if latest.empty:
        return
    latest_prices = {
        ticker: fetcher.fetch_info(ticker).get("currentPrice") or fetcher.fetch_info(ticker).get("lastPrice") or 0.0
        for ticker in latest["ticker"].tolist()
    }
    send_midweek_update(config, _build_progress_summary(latest, latest_prices))


def send_endweek_bot_update() -> None:
    config = get_config()
    if not config.feature_flags.bot_integration:
        return
    backtest = BacktestTracker(config.backtest_db)
    performance = with_resolved_outcomes(backtest.performance_frame()).dropna(subset=["realized_return"]).head(10)
    if performance.empty:
        return
    lines = [
        f"{row.ticker}: {'hit' if row.resolved_target_hit else 'miss'} ({row.realized_return:.2%})"
        for row in performance.itertuples(index=False)
    ]
    send_endweek_results(config, "\n".join(lines))


def run_training(*, fresh: bool = False, universe_mode: str | None = None) -> str:
    config = get_config()
    def progress(message: str) -> None:
        print(f"[training] {message}", flush=True)

    progress("Preparing model training job")
    setup_logging(config.predictions_log)
    cache = SQLiteCache(config.cache_db)
    fetcher = MarketDataFetcher(config, cache)
    resolved_mode = universe_mode or config.default_universe
    progress(f"Resolving {resolved_mode} universe")
    universe = fetcher.resolve_universe(mode=resolved_mode, fresh=fresh)
    sector_map = {
        ticker: sector
        for sector, tickers in universe.items()
        for ticker in tickers
    }
    universe_tickers = {
        ticker for sector_tickers in universe.values() for ticker in sector_tickers
    }
    progress("Fetching historical daily price frames")
    price_frames = fetcher.fetch_universe_daily_frames(
        fresh=fresh,
        mode=resolved_mode,
        period=config.training_history_period,
    )
    predictor = AdaptivePredictor(config)
    progress("Fetching sector and macro history")
    sector_histories = fetcher.fetch_sector_history_for_period(config.training_history_period, fresh=fresh)
    vix_history = fetcher.fetch_macro_history(config.vix_ticker, fresh=fresh)
    put_call_history = fetcher.fetch_macro_history(config.put_call_ticker, fresh=fresh)
    hyg_history = fetcher.fetch_macro_history(config.hyg_ticker, fresh=fresh)
    lqd_history = fetcher.fetch_macro_history(config.lqd_ticker, fresh=fresh)
    tlt_history = fetcher.fetch_macro_history(config.tlt_ticker, fresh=fresh)
    progress("Fetching earnings calendar data")
    earnings_dates_map = fetcher.fetch_earnings_dates_for_tickers(universe_tickers, fresh=fresh)
    progress("Fetching benchmark history")
    benchmark_history = fetcher.download_history(
        config.benchmark_ticker,
        interval="1d",
        period=config.training_history_period,
        ttl_seconds=config.cache_ttls.market_history,
        fresh=fresh,
        cache_only=False,
    )
    progress("Calculating market breadth features")
    breadth_history = fetcher.calculate_breadth_history(price_frames)
    progress("Training adaptive regime ensemble")
    report = predictor.fit(
        price_frames,
        save_model=True,
        sector_map=sector_map,
        sector_histories=sector_histories,
        vix_history=vix_history,
        earnings_dates_map=earnings_dates_map,
        benchmark_history=benchmark_history,
        breadth_history=breadth_history,
        put_call_history=put_call_history,
        hyg_history=hyg_history,
        lqd_history=lqd_history,
        tlt_history=tlt_history,
    )
    progress("Writing training report and model artifacts")
    config.model_trained = report.trained
    lines = [
        "Training Report",
        f"Universe: {resolved_mode}",
        f"Universe tickers: {len(universe_tickers)}",
        f"Daily frames with data: {sum(1 for frame in price_frames.values() if not frame.empty)}",
        f"Training samples: {report.training_samples}",
        f"Validation samples: {report.validation_samples}",
        f"Model family: {report.model_family}",
        f"Selected profile: {report.selected_profile}",
        f"Label: {report.label_definition}",
        f"Positive class: {report.positive_ratio * 100:.1f}% | Negative class: {report.negative_ratio * 100:.1f}%",
        f"Scale pos weight: {report.scale_pos_weight:.2f}",
        f"Accuracy: {report.accuracy:.3f}",
        f"Precision: {report.precision:.3f}",
        f"Recall: {report.recall:.3f}",
        f"Walk-forward ensemble AUC: {report.auc:.3f}",
        f"Walk-forward XGBoost AUC: {report.xgb_auc:.3f}",
        f"Walk-forward LightGBM AUC: {report.lightgbm_auc:.3f}",
        f"Validation ensemble AUC: {report.ensemble_auc:.3f}",
    ]
    if report.fold_aucs:
        lines.append(
            "Fold AUCs: "
            + " | ".join(
                f"{row.get('month', row.get('fold'))}: {float(row.get('auc', 0.0)):.3f}"
                for row in report.fold_aucs
            )
        )
    if report.feature_importance:
        top_features = list(report.feature_importance.items())[:10]
        lines.append("Top predictive features:")
        lines.append(
            " | ".join(f"{feature}: {value * 100:.1f}%" for feature, value in top_features)
        )
        if config.feature_importance_png.exists():
            lines.append(f"Feature importance chart: {config.feature_importance_png}")
        else:
            lines.append("Feature importance chart unavailable (matplotlib not installed).")
    if config.feature_flags.walk_forward_backtester:
        progress("Running 12-week walk-forward validation")
        walk_forward = predictor.walk_forward_backtest(
            price_frames,
            benchmark_history=benchmark_history,
            breadth_history=breadth_history,
            sector_histories=sector_histories,
            vix_history=vix_history,
            put_call_history=put_call_history,
            hyg_history=hyg_history,
            lqd_history=lqd_history,
            tlt_history=tlt_history,
            weeks=12,
        )
        progress("Saving walk-forward backtest report")
        config.backtest_report_path.write_text(walk_forward["markdown"], encoding="utf-8")
        summary = walk_forward["summary"]
        if summary:
            lines.append(
                f"Walk-forward: win rate {summary['win_rate']:.2f}%, avg return {summary['average_return']:.2f}%, sharpe {summary['sharpe']:.2f}"
            )
            lines.append(f"Backtest report: {config.backtest_report_path}")
        persist_adaptive_regime_summary(
            config,
            walk_forward.get("regime_summary", {}),
            summary=summary,
            source="training_walk_forward",
        )
    if predictor.regime_ensembles:
        lines.append(f"Adaptive regimes: {', '.join(sorted(predictor.regime_ensembles))}")
    progress("Training complete")
    return "\n".join(lines)


def run_results() -> str:
    config = get_config()
    setup_logging(config.predictions_log)
    cache = SQLiteCache(config.cache_db)
    fetcher = MarketDataFetcher(config, cache)
    tracker = BacktestTracker(config.paper_trade_db)
    attribution_tracker = SignalAttributionTracker(config.signal_attribution_db)
    frame = tracker.paper_results_frame()
    if frame.empty:
        return "No paper trades recorded yet."
    latest_prices = {}
    for ticker in frame["ticker"].dropna().unique().tolist():
        info = fetcher.fetch_info(str(ticker))
        latest_prices[str(ticker)] = info.get("currentPrice") or info.get("lastPrice") or 0.0
    price_paths = fetcher.fetch_daily_frames_for_tickers(
        frame["ticker"].dropna().unique().tolist(),
        fresh=False,
        cache_only=False,
        period="6mo",
    )
    tracker.evaluate_due_paper_predictions(latest_prices, price_paths=price_paths)
    attribution_tracker.evaluate_due_predictions(latest_prices, price_paths=price_paths)
    frame = with_resolved_outcomes(tracker.paper_results_frame().copy())
    frame["current_price"] = frame["current_price"].fillna(frame["ticker"].map(latest_prices))
    frame["pct_change"] = (
        (frame["current_price"].fillna(frame["entry_price"]) / frame["entry_price"] - 1.0) * 100.0
    )
    frame["positive_return"] = frame["pct_change"] > 0
    frame["week_label"] = pd.to_datetime(frame["created_at"], utc=True, errors="coerce").dt.strftime("%m/%d")
    weekly = (
        frame.groupby("week_label", as_index=False)
        .agg(
            picks=("ticker", "count"),
            target_hits=("resolved_target_hit", "sum"),
            target_hit_rate=("resolved_target_hit", "mean"),
            positive_return_rate=("positive_return", "mean"),
            avg_return=("pct_change", "mean"),
        )
        .sort_values("week_label", ascending=False)
        .head(8)
    )
    summary = tracker.paper_trade_results_summary()
    attribution_summary = attribution_tracker.signal_summary()
    threshold_recommendation = tracker.threshold_recommendation()
    metadata = {}
    if config.adaptive_metadata_path.exists():
        try:
            metadata = json.loads(config.adaptive_metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
    lines = [
        "══════════════════════════════════════════════",
        "📈 PAPER TRADE RESULTS — Last 8 Weeks",
        "══════════════════════════════════════════════",
    ]
    if weekly.empty:
        lines.append("No completed paper-trade evaluations yet.")
        return "\n".join(lines)
    display = weekly.copy()
    display["target_hit_rate"] = display["target_hit_rate"].map(lambda value: f"{value * 100:.0f}%")
    display["positive_return_rate"] = display["positive_return_rate"].map(lambda value: f"{value * 100:.0f}%")
    display["avg_return"] = display["avg_return"].map(lambda value: f"{value:+.1f}%")
    display = display.rename(
        columns={
            "week_label": "Week",
            "picks": "Picks",
            "target_hits": "Target Hits",
            "target_hit_rate": "Target Hit Rate",
            "positive_return_rate": "Positive Return Rate",
            "avg_return": "Avg Return",
        }
    )
    lines.extend([display.to_string(index=False), "──────────────────────────────────────────────"])
    if summary:
        best_pick_row = frame.sort_values("pct_change", ascending=False).iloc[0]
        streak = compute_paper_trade_streak(weekly)
        lines.extend(
            [
                (
                    f"8-Week Average: {summary.get('target_hit_rate', summary['win_rate']):.0f}% target hit rate | "
                    f"{summary.get('positive_return_rate', 0):.0f}% positive-return rate | "
                    f"{summary['average_return']:+.1f}% avg return"
                ),
                f"Best Week: {summary['best_week']} ({summary['best_week_return']:+.1f}%)",
                f"Worst Week: {summary['worst_week']} ({summary['worst_week_return']:+.1f}%)",
                (
                f"Best Pick Ever: {best_pick_row['ticker']} {best_pick_row['pct_change']:+.1f}% "
                f"(week of {best_pick_row['week_label']})"
                ),
                f"Current Streak: {streak}",
            ]
        )
    if attribution_summary:
        lines.append(
            (
                f"Best signal: {attribution_summary['best_signal']} "
                f"({attribution_summary['best_signal_win_rate']:.0f}% target-hit rate last 4 weeks)"
            )
        )
    if threshold_recommendation:
        lines.append(
            (
                f"Threshold suggestion: {threshold_recommendation['threshold']:.0f} gave the best "
                f"target-hit rate ({threshold_recommendation['win_rate']:.0f}%) across "
                f"{threshold_recommendation['weeks']} tracked weeks."
            )
        )
    regime_summary = {}
    if isinstance(metadata, dict):
        regime_summary = metadata.get("adaptive_backtest_regime_summary") or metadata.get("regime_summary", {})
    if regime_summary:
        lines.append("Adaptive regime win rates:")
        for regime, metrics in regime_summary.items():
            lines.append(
                f"  {str(regime).upper()}: {float(metrics.get('win_rate', 0.0)):.0f}% target-hit rate | "
                f"{float(metrics.get('average_return', 0.0)):+.1f}% avg return"
            )
    lines.append("══════════════════════════════════════════════")
    return "\n".join(lines)


def run_adaptive_backtest(*, fresh: bool = False, universe_mode: str | None = None) -> str:
    config = get_config()
    def progress(message: str) -> None:
        print(f"[backtest] {message}", flush=True)

    progress("Preparing adaptive walk-forward backtest")
    setup_logging(config.predictions_log)
    cache = SQLiteCache(config.cache_db)
    fetcher = MarketDataFetcher(config, cache)
    resolved_mode = universe_mode or config.default_universe
    progress(f"Resolving {resolved_mode} universe")
    universe = fetcher.resolve_universe(mode=resolved_mode, fresh=fresh)
    universe_tickers = {ticker for sector_tickers in universe.values() for ticker in sector_tickers}
    progress(f"Fetching historical prices for {len(universe_tickers)} tickers")
    price_frames = fetcher.fetch_universe_daily_frames(
        fresh=fresh,
        mode=resolved_mode,
        period=config.training_history_period,
    )
    predictor = AdaptivePredictor(config)
    progress("Fetching benchmark and macro history")
    benchmark_history = fetcher.download_history(
        config.benchmark_ticker,
        interval="1d",
        period=config.training_history_period,
        ttl_seconds=config.cache_ttls.market_history,
        fresh=fresh,
        cache_only=False,
    )
    sector_histories = fetcher.fetch_sector_history_for_period(config.training_history_period, fresh=fresh)
    vix_history = fetcher.fetch_macro_history(config.vix_ticker, fresh=fresh)
    put_call_history = fetcher.fetch_macro_history(config.put_call_ticker, fresh=fresh)
    hyg_history = fetcher.fetch_macro_history(config.hyg_ticker, fresh=fresh)
    lqd_history = fetcher.fetch_macro_history(config.lqd_ticker, fresh=fresh)
    tlt_history = fetcher.fetch_macro_history(config.tlt_ticker, fresh=fresh)
    progress("Calculating breadth and regime features")
    breadth_history = fetcher.calculate_breadth_history(price_frames)
    backtest_weeks = 26 if resolved_mode == "mini" else 52
    progress(f"Running {backtest_weeks}-week rolling backtest")
    result = predictor.walk_forward_backtest(
        price_frames,
        benchmark_history=benchmark_history,
        breadth_history=breadth_history,
        sector_histories=sector_histories,
        vix_history=vix_history,
        put_call_history=put_call_history,
        hyg_history=hyg_history,
        lqd_history=lqd_history,
        tlt_history=tlt_history,
        weeks=backtest_weeks,
    )
    lines = [
        f"REGIME BACKTEST RESULTS ({'6 months' if backtest_weeks == 26 else '12 months'})",
        "═══════════════════════════════════",
    ]
    for regime, metrics in result.get("regime_summary", {}).items():
        lines.append(
            f"{regime.upper():<15} {int(metrics.get('weeks', 0))} weeks  "
            f"Win rate: {float(metrics.get('win_rate', 0.0)):.0f}%  "
            f"Avg return: {float(metrics.get('average_return', 0.0)):+.1f}%"
        )
    summary = result.get("summary", {})
    lines.extend(
        [
            "",
            (
                f"OVERALL: {int(summary.get('weeks', 0))} weeks | "
                f"{float(summary.get('win_rate', 0.0)):.0f}% win rate | "
                f"{float(summary.get('average_return', 0.0)):+.1f}% avg | "
                f"Sharpe {float(summary.get('sharpe', 0.0)):.2f}"
            ),
        ]
    )
    progress("Saving backtest report")
    config.backtest_report_path.write_text(result.get("markdown", ""), encoding="utf-8")
    persist_adaptive_regime_summary(
        config,
        result.get("regime_summary", {}),
        summary=result.get("summary", {}),
        source="adaptive_backtest",
    )
    progress("Backtest complete")
    return "\n".join(lines)


def compute_paper_trade_streak(weekly: pd.DataFrame) -> str:
    hit_rate_column = "target_hit_rate" if "target_hit_rate" in weekly else "hit_rate"
    if weekly.empty or hit_rate_column not in weekly:
        return "No streak data"
    streak_weeks = 0
    direction = None
    ordered = weekly.sort_values("week_label", ascending=False)
    for _, row in ordered.iterrows():
        above_threshold = float(row[hit_rate_column]) >= 0.60
        if direction is None:
            direction = above_threshold
        if above_threshold != direction:
            break
        streak_weeks += 1
    if direction is None or streak_weeks == 0:
        return "No streak data"
    badge = "🔥" if direction else "⚠️"
    label = "above 60% target-hit rate" if direction else "below 60% target-hit rate"
    return f"{streak_weeks} weeks {label} {badge}"


def run_warm_cache(*, fresh: bool = False, universe_mode: str | None = None) -> str:
    config = get_config()
    setup_logging(config.predictions_log)
    fetcher = MarketDataFetcher(config, SQLiteCache(config.cache_db))
    result = fetcher.warm_cache(mode=universe_mode or config.default_universe, fresh=fresh)
    return (
        f"Cache warmed: {result['ready']}/{result['total']} tickers ready "
        f"({result['failed']} failed)"
    )


def build_startup_banner(config, *, universe_mode: str) -> str:
    metadata = {}
    if config.xgb_metadata_path.exists():
        try:
            metadata = json.loads(config.xgb_metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
    latest_scan = {}
    if config.latest_scan_path.exists():
        try:
            latest_scan = json.loads(config.latest_scan_path.read_text(encoding="utf-8"))
        except Exception:
            latest_scan = {}
    summary = latest_scan.get("scan_summary", {})
    selected = latest_scan.get("selected", [])
    top_label = "N/A"
    if selected:
        top_candidate = selected[0]
        top_label = f"{top_candidate.get('ticker', 'N/A')} {float(top_candidate.get('final_score', 0.0)):.2f}"
    trained_at = str(metadata.get("trained_at", "n/a"))[:10]
    auc = float(metadata.get("auc", 0.0))
    model_family = str(metadata.get("model_family", "XGB+LGBM"))
    universe_total = int(summary.get("universe_total", 0))
    if not universe_total:
        universe_total = {"mini": 69, "sp500": 503, "nasdaq100": 101, "full": 565, "us_market": 10000}.get(universe_mode, 0)
    last_scan_date = str(latest_scan.get("generated_at", ""))[:10] or "n/a"
    return "\n".join(
        [
            "╔══════════════════════════════════════════╗",
            "║   📈 STOCK PREDICTOR v3.0               ║",
            f"║   Universe: {universe_total:<3} | Model: {model_family:<11}║",
            f"║   AUC: {auc:.3f} | Trained: {trained_at:<10}║",
            f"║   Last scan: {last_scan_date:<10} | Top: {top_label:<10}║",
            "╚══════════════════════════════════════════╝",
        ]
    )


def build_config_status_line(config, *, universe_mode: str) -> str:
    cache = SQLiteCache(config.cache_db)
    fetcher = MarketDataFetcher(config, cache)
    paper_tracker = BacktestTracker(config.paper_trade_db)

    model_state = "❌ Model missing"
    if config.xgb_metadata_path.exists():
        model_age_hours = (time.time() - config.xgb_metadata_path.stat().st_mtime) / 3600.0
        model_state = "✅ Model fresh" if model_age_hours <= 24 * 7 else "⚠️ Model stale"

    cache_state = "⚠️ Cache cold"
    warm_count = 0
    total_tickers = 0
    try:
        universe = fetcher.resolve_universe(mode=universe_mode, fresh=False)
        tickers = sorted({ticker for sector_tickers in universe.values() for ticker in sector_tickers})
        total_tickers = len(tickers)
        if tickers:
            with cache.connection() as conn:
                placeholders = ",".join("?" for _ in tickers)
                keys = [f"history:{ticker}:1d:{config.daily_period}" for ticker in tickers]
                row = conn.execute(
                    f"""
                    SELECT COUNT(*) AS warm_count
                    FROM cache_entries
                    WHERE cache_key IN ({placeholders})
                      AND (expires_at IS NULL OR expires_at > ?)
                    """,
                    (*keys, time.time()),
                ).fetchone()
            warm_count = int(row["warm_count"]) if row else 0
            warm_ratio = warm_count / max(total_tickers, 1)
            cache_state = (
                f"✅ Cache {warm_ratio * 100:.0f}% warm"
                if warm_ratio >= 0.50
                else f"⚠️ Cache {warm_ratio * 100:.0f}% warm"
            )
    except Exception:
        cache_state = "⚠️ Cache unknown"

    universe_state = "✅ Universe current"
    try:
        needs = []
        normalized = {"nasdaq": "nasdaq100", "custom": "mini"}.get(universe_mode, universe_mode)
        if normalized in {"sp500", "full"}:
            needs.append("sp500-universe")
        if normalized in {"nasdaq100", "full"}:
            needs.append("nasdaq100-universe")
        if normalized == "us_market":
            needs.append("sec-ticker-map")
        if needs:
            with cache.connection() as conn:
                row_map = {
                    row["cache_key"]: row["created_at"]
                    for row in conn.execute(
                        f"SELECT cache_key, created_at FROM cache_entries WHERE cache_key IN ({','.join('?' for _ in needs)})",
                        needs,
                    ).fetchall()
                }
            current = all((time.time() - float(row_map.get(key, 0))) <= 7 * 24 * 3600 for key in needs)
            universe_state = "✅ Universe current" if current else "⚠️ Universe stale"
    except Exception:
        universe_state = "⚠️ Universe unknown"

    weeks_tracked = paper_tracker.weeks_tracked()
    tracking_state = f"📊 {weeks_tracked} weeks tracked" if weeks_tracked else "📊 No weeks tracked"
    return f"{model_state} | {cache_state} | {universe_state} | {tracking_state}"


def scheduler_loop() -> None:
    """Run the scan every Sunday at 6 PM local time."""

    config = get_config()
    if schedule is None:
        raise RuntimeError("The 'schedule' package is required for --schedule mode.")
    if config.feature_flags.smart_cache_warmup:
        schedule.every().sunday.at("18:00").do(run_warm_cache, fresh=False, universe_mode=config.default_universe)
    schedule.every().sunday.at("19:00").do(run_scan, fresh=False)
    if config.feature_flags.bot_integration:
        schedule.every().wednesday.at("12:00").do(send_midweek_bot_update)
        schedule.every().friday.at("16:30").do(send_endweek_bot_update)
    while True:
        schedule.run_pending()
        time.sleep(30)


def main() -> None:
    config = get_config()
    parser = argparse.ArgumentParser(description="Elite stock prediction system")
    parser.add_argument("--fresh", action="store_true", help="Ignore cache and refresh external data")
    parser.add_argument("--schedule", action="store_true", help="Run the weekly Sunday 6 PM scheduler")
    parser.add_argument("--email", help="Optional destination email for the finished report")
    parser.add_argument("--sms", help="Optional destination phone number for the finished report")
    parser.add_argument("--threshold", type=float, help="Override the active score threshold")
    parser.add_argument("--top-n", type=int, default=10, help="Always return the top N ranked picks")
    parser.add_argument(
        "--universe",
        choices=["sp500", "nasdaq100", "full", "mini", "us_market"],
        default=config.default_universe,
        help="Choose the stock universe to scan",
    )
    parser.add_argument("--debug", action="store_true", help="Print sub-score breakdowns for every ticker")
    parser.add_argument("--analyze", metavar="TICKER", help="Analyze a single stock with the live scoring stack")
    parser.add_argument("--paper-trade", action="store_true", help="Log picks to the weekly paper-trading database")
    parser.add_argument("--train", action="store_true", help="Train the XGBoost model and print metrics")
    parser.add_argument("--backtest-adaptive", action="store_true", help="Run the adaptive regime walk-forward backtest")
    parser.add_argument("--results", action="store_true", help="Show live paper-trade results")
    parser.add_argument("--warm-cache", action="store_true", help="Pre-fetch and cache price history for the selected universe")
    parser.add_argument("--health", action="store_true", help="Run provider and cache health checks")
    parser.add_argument("--check", action="store_true", help="Alias for --health")
    args = parser.parse_args()
    if config.feature_flags.startup_banner:
        print(build_startup_banner(config, universe_mode=args.universe))
    if config.feature_flags.config_validation:
        print(build_config_status_line(config, universe_mode=args.universe))
    if args.health or args.check:
        print(provider_health_check(config))
        return
    if args.train:
        print(run_training(fresh=args.fresh, universe_mode=args.universe))
        return
    if args.backtest_adaptive:
        print(run_adaptive_backtest(fresh=args.fresh, universe_mode=args.universe))
        return
    if args.analyze:
        payload = run_single_ticker_analysis(
            args.analyze,
            fresh=args.fresh,
            universe_mode=args.universe,
            threshold_override=args.threshold,
        )
        try:
            print(config.latest_single_analysis_path.read_text(encoding="utf-8"))
        except Exception:
            print(json.dumps(payload, indent=2, default=json_default))
        return
    if args.results:
        print(run_results())
        return
    if args.warm_cache:
        print(run_warm_cache(fresh=args.fresh, universe_mode=args.universe))
        return

    if args.schedule:
        scheduler_loop()
        return

    payload = run_scan(
        fresh=args.fresh,
        alert_email=args.email,
        alert_phone=args.sms,
        threshold_override=args.threshold,
        top_n=args.top_n,
        universe_mode=args.universe,
        debug=args.debug,
        paper_trade=args.paper_trade,
    )
    print(payload["terminal_report"])


if __name__ == "__main__":
    main()
