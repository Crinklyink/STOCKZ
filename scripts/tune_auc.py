#!/usr/bin/env python3
"""Direct AUC tuning entry point that avoids the heavier main.py orchestration."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from stock_predictor.config import get_config
from stock_predictor.data.cache import SQLiteCache
from stock_predictor.data.fetcher import MarketDataFetcher
from stock_predictor.models.xgboost_model import XGBoostPredictor
from stock_predictor.utils import save_json, setup_logging


def run_tuning(universe: str, period: str, trials: int, save_model: bool) -> dict[str, object]:
    config = get_config()
    config.training_search_trials = max(int(trials), 0)
    setup_logging(config.predictions_log)
    cache = SQLiteCache(config.cache_db)
    fetcher = MarketDataFetcher(config, cache)
    universe_map = fetcher.resolve_universe(mode=universe, fresh=False)
    sector_map = {ticker: sector for sector, tickers in universe_map.items() for ticker in tickers}
    tickers = {ticker for tickers in universe_map.values() for ticker in tickers}
    frames = fetcher.fetch_daily_frames_for_tickers(tickers, period=period, fresh=False)
    sector_histories = fetcher.fetch_sector_history_for_period(period, fresh=False)
    vix_history = fetcher.fetch_macro_history(config.vix_ticker, fresh=False)
    benchmark_history = fetcher.download_history(
        config.benchmark_ticker,
        interval="1d",
        period=period,
        ttl_seconds=config.cache_ttls.market_history,
        fresh=False,
        cache_only=False,
    )
    breadth_history = fetcher.calculate_breadth_history(frames)
    earnings_dates_map = fetcher.fetch_earnings_dates_for_tickers(tickers, fresh=False)
    predictor = XGBoostPredictor(config, load_persisted=False)
    report = predictor.fit(
        frames,
        save_model=save_model,
        sector_map=sector_map,
        sector_histories=sector_histories,
        vix_history=vix_history,
        earnings_dates_map=earnings_dates_map,
        benchmark_history=benchmark_history,
        breadth_history=breadth_history,
    )
    walk_forward = predictor.walk_forward_backtest(
        frames,
        benchmark_history=benchmark_history,
        breadth_history=breadth_history,
        weeks=12 if universe == "mini" else 26,
    )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "universe": universe,
        "period": period,
        "search_trials": trials,
        "training_report": report.to_dict(),
        "walk_forward_summary": walk_forward.get("summary", {}),
    }
    save_json(config.latest_scan_path.parent / "auc_tuning_report.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune stock model AUC with direct XGB+LGBM training.")
    parser.add_argument("--universe", choices=["mini", "full", "sp500", "nasdaq100"], default="mini")
    parser.add_argument("--period", default="2y")
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--no-save-model", action="store_true")
    args = parser.parse_args()
    result = run_tuning(
        universe=args.universe,
        period=args.period,
        trials=args.trials,
        save_model=not args.no_save_model,
    )
    report = result["training_report"]
    print(json.dumps(
        {
            "ensemble_auc": report.get("auc", 0.0),
            "xgb_auc": report.get("xgb_auc", 0.0),
            "lightgbm_auc": report.get("lightgbm_auc", 0.0),
            "selected_profile": report.get("selected_profile", "unknown"),
            "training_samples": report.get("training_samples", 0),
            "validation_samples": report.get("validation_samples", 0),
            "walk_forward_summary": result.get("walk_forward_summary", {}),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
