"""Congressional trading signal extraction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List

import requests

from stock_predictor.config import AppConfig
from stock_predictor.data.cache import SQLiteCache
from stock_predictor.utils import clamp


@dataclass(slots=True)
class CongressSignal:
    score: float
    recent_buys: int
    recent_sells: int
    summary: str


class CongressTradeTracker:
    """Fetch congressional trades from configurable providers."""

    def __init__(self, config: AppConfig, cache: SQLiteCache) -> None:
        self.config = config
        self.cache = cache

    def fetch_recent_trades(self, fresh: bool = False) -> List[Dict[str, Any]]:
        cache_key = "congress-trades"
        if not fresh:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        endpoints = [
            (self.config.quiver_endpoint, self.config.quiver_token),
            (self.config.capitol_trades_endpoint, self.config.capitol_trades_token),
        ]
        records: List[Dict[str, Any]] = []
        for endpoint, token in endpoints:
            if not endpoint:
                continue
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            try:
                response = requests.get(endpoint, headers=headers, timeout=20)
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict):
                    records.extend(payload.get("results") or payload.get("data") or [])
                elif isinstance(payload, list):
                    records.extend(payload)
            except Exception:
                continue
        self.cache.set(cache_key, records, ttl_seconds=self.config.cache_ttls.congress_trades)
        return records

    def score_ticker(self, ticker: str, fresh: bool = False) -> CongressSignal:
        recent_cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        buys = 0
        sells = 0
        rows = self.fetch_recent_trades(fresh=fresh)
        for row in rows:
            symbol = str(row.get("ticker") or row.get("symbol") or "").upper()
            if symbol != ticker.upper():
                continue
            raw_date = row.get("transactionDate") or row.get("date") or row.get("published_at")
            try:
                trade_date = datetime.fromisoformat(str(raw_date).replace("Z", "+00:00"))
            except Exception:
                trade_date = datetime.now(timezone.utc)
            if trade_date.tzinfo is None:
                trade_date = trade_date.replace(tzinfo=timezone.utc)
            if trade_date < recent_cutoff:
                continue
            trade_type = str(row.get("type") or row.get("transactionType") or row.get("transaction") or "").lower()
            if "buy" in trade_type or "purchase" in trade_type:
                buys += 1
            elif "sell" in trade_type or "sale" in trade_type:
                sells += 1
        if not rows:
            score = 0.5
        else:
            score = clamp((buys - sells + 2) / 4.0, 0.0, 1.0)
        summary = f"Congress buys {buys}, sells {sells} in last 30 days"
        return CongressSignal(score=score, recent_buys=buys, recent_sells=sells, summary=summary)
