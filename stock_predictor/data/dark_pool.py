"""Unusual options activity and dark-pool signal extraction."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean
from typing import Any, Dict, Iterable, List

import requests

from stock_predictor.config import AppConfig
from stock_predictor.data.cache import SQLiteCache
from stock_predictor.data.fetcher import MarketDataFetcher
from stock_predictor.utils import clamp, coerce_float

LOGGER = logging.getLogger(__name__)


class FlowDetector:
    """Collect options-flow and dark-pool proxies from free and paid sources."""

    def __init__(
        self,
        config: AppConfig,
        cache: SQLiteCache,
        fetcher: MarketDataFetcher,
    ) -> None:
        self.config = config
        self.cache = cache
        self.fetcher = fetcher

    def score_universe(
        self,
        tickers: Iterable[str],
        prices: Dict[str, float],
        fresh: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        tickers = list(tickers)
        if not tickers:
            return {}
        max_workers = self.config.max_parallel_tickers if self.config.feature_flags.parallel_processing else 1
        results: Dict[str, Dict[str, float]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = {
                executor.submit(self.score_ticker, ticker, prices.get(ticker, 0.0), fresh): ticker
                for ticker in tickers
            }
            for future in as_completed(tasks):
                ticker = tasks[future]
                try:
                    results[ticker] = future.result()
                except Exception:
                    LOGGER.debug("options scoring failed for %s", ticker, exc_info=True)
                    results[ticker] = {
                        "put_call_ratio": 1.0,
                        "avg_implied_volatility": 0.0,
                        "iv_rank": 0.5,
                        "gamma_signal": 0.5,
                        "call_sweep_signal": 0.5,
                        "premium_bullish_notional": 0.0,
                        "premium_bearish_notional": 0.0,
                        "premium_darkpool_notional": 0.0,
                        "bearish_flow_ratio": 0.0,
                        "options_score": self.config.default_missing_signal_value / 100.0,
                        "has_data": False,
                        "defaulted": True,
                    }
        return results

    def score_ticker(self, ticker: str, price: float, fresh: bool = False) -> Dict[str, float]:
        if self._should_use_neutral_fallback(ticker, fresh=fresh):
            return self._default_score()
        options_payload = self.fetcher.fetch_options_chain(ticker, fresh=fresh)
        premium_flow = self.fetch_premium_flow(ticker, fresh=fresh)
        calls, puts = [], []
        for expiry, chains in options_payload.get("chains", {}).items():
            expiry_calls = chains.get("calls", [])
            expiry_puts = chains.get("puts", [])
            for record in expiry_calls:
                record["expiry"] = expiry
            for record in expiry_puts:
                record["expiry"] = expiry
            calls.extend(expiry_calls)
            puts.extend(expiry_puts)

        total_call_volume = sum(coerce_float(row.get("volume")) for row in calls)
        total_put_volume = sum(coerce_float(row.get("volume")) for row in puts)
        put_call_ratio = total_put_volume / max(total_call_volume, 1.0)
        avg_iv = mean(
            [
                coerce_float(row.get("impliedVolatility"))
                for row in calls + puts
                if row.get("impliedVolatility") is not None
            ]
            or [0.0]
        )
        near_money_calls = [
            row
            for row in calls
            if abs(coerce_float(row.get("strike")) - price) / max(price, 1.0) <= 0.05
        ]
        call_open_interest = sum(coerce_float(row.get("openInterest")) for row in near_money_calls)
        unusual_call_notional = sum(
            coerce_float(row.get("lastPrice")) * 100.0 * coerce_float(row.get("volume"))
            for row in near_money_calls
            if coerce_float(row.get("volume")) > 0
        )
        gamma_signal = clamp(call_open_interest / max(1_000.0, price * 100.0), 0.0, 1.0)
        call_sweep_signal = clamp(
            unusual_call_notional / max(self.config.thresholds.options_block_notional * 5, 1.0),
            0.0,
            1.0,
        )
        premium_bullish_notional = sum(
            row["premium"]
            for row in premium_flow
            if row.get("side") == "call" and row.get("sentiment") == "bullish"
        )
        premium_bearish_notional = sum(
            row["premium"]
            for row in premium_flow
            if row.get("side") == "put" or row.get("sentiment") == "bearish"
        )
        premium_darkpool_notional = sum(
            row["premium"] for row in premium_flow if row.get("source") == "dark_pool"
        )
        iv_rank = self.estimate_iv_rank(ticker, avg_iv)
        has_data = bool(calls or puts or premium_flow)
        if not has_data:
            return self._default_score()

        score = clamp(
            (
                0.30 * (1.0 - clamp(put_call_ratio / 1.5, 0.0, 1.0))
                + 0.25 * call_sweep_signal
                + 0.2 * gamma_signal
                + 0.15 * clamp((premium_bullish_notional / 2_000_000.0), 0.0, 1.0)
                + 0.1 * clamp((premium_darkpool_notional / 2_000_000.0), 0.0, 1.0)
            ),
            0.0,
            1.0,
        )
        return {
            "put_call_ratio": coerce_float(put_call_ratio, default=1.0),
            "avg_implied_volatility": avg_iv,
            "iv_rank": iv_rank,
            "gamma_signal": gamma_signal,
            "call_sweep_signal": call_sweep_signal,
            "premium_bullish_notional": premium_bullish_notional,
            "premium_bearish_notional": premium_bearish_notional,
            "premium_darkpool_notional": premium_darkpool_notional,
            "bearish_flow_ratio": premium_bearish_notional / max(premium_bullish_notional + premium_darkpool_notional, 1.0),
            "options_score": score,
            "has_data": True,
            "defaulted": False,
        }

    def _should_use_neutral_fallback(self, ticker: str, *, fresh: bool) -> bool:
        if self.config.unusual_whales_endpoint or self.config.tradytics_endpoint:
            return False
        if fresh:
            return True
        return not bool(self.cache.get(f"options:{ticker}") or self.cache.get(f"premium-flow:{ticker}"))

    def _default_score(self) -> Dict[str, float]:
        return {
            "put_call_ratio": 1.0,
            "avg_implied_volatility": 0.0,
            "iv_rank": 0.5,
            "gamma_signal": 0.5,
            "call_sweep_signal": 0.5,
            "premium_bullish_notional": 0.0,
            "premium_bearish_notional": 0.0,
            "premium_darkpool_notional": 0.0,
            "bearish_flow_ratio": 0.0,
            "options_score": self.config.default_missing_signal_value / 100.0,
            "has_data": False,
            "defaulted": True,
        }

    def fetch_premium_flow(self, ticker: str, fresh: bool = False) -> List[Dict[str, Any]]:
        cache_key = f"premium-flow:{ticker}"
        if not fresh:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        endpoints = [
            (self.config.unusual_whales_endpoint, self.config.unusual_whales_token),
            (self.config.tradytics_endpoint, self.config.tradytics_token),
        ]
        combined: List[Dict[str, Any]] = []
        for endpoint, token in endpoints:
            if not endpoint:
                continue
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            try:
                response = requests.get(
                    endpoint,
                    params={"ticker": ticker},
                    headers=headers,
                    timeout=20,
                )
                response.raise_for_status()
                payload = response.json()
                records = payload.get("results") or payload.get("data") or []
                for record in records:
                    combined.append(
                        {
                            "source": record.get("source", "options_flow"),
                            "side": record.get("side", "call"),
                            "sentiment": record.get("sentiment", "bullish"),
                            "premium": coerce_float(
                                record.get("premium")
                                or record.get("notional")
                                or record.get("value")
                            ),
                        }
                    )
            except Exception:
                LOGGER.debug("premium flow endpoint failed for %s", ticker, exc_info=True)
        self.cache.set(cache_key, combined, ttl_seconds=self.config.cache_ttls.premium_flow)
        return combined

    def estimate_iv_rank(self, ticker: str, current_iv: float) -> float:
        cache_key = f"iv-history:{ticker}"
        history = self.cache.get(cache_key) or []
        values = [coerce_float(value) for value in history[-60:]] + [current_iv]
        minimum = min(values or [0.0])
        maximum = max(values or [1.0])
        iv_rank = 0.0 if maximum == minimum else (current_iv - minimum) / (maximum - minimum)
        self.cache.set(cache_key, values[-90:], ttl_seconds=60 * 60 * 24 * 90)
        return clamp(iv_rank, 0.0, 1.0)
