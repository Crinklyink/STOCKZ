"""GPT-powered reasoning over top-candidate news headlines."""

from __future__ import annotations

import json
import logging
from typing import Dict, Iterable, List

from stock_predictor.config import AppConfig
from stock_predictor.data.cache import SQLiteCache
from stock_predictor.utils import clamp

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


class GPTNewsReasoner:
    """Ask GPT whether recent headlines support a 5-day 4%+ move thesis."""

    def __init__(self, config: AppConfig, cache: SQLiteCache) -> None:
        self.config = config
        self.cache = cache
        self.client = OpenAI(api_key=config.openai_api_key) if (OpenAI and config.openai_api_key) else None

    def reason_about_news(self, ticker: str, headlines: Iterable[str], fresh: bool = False) -> Dict[str, object]:
        cache_key = f"gpt-news-reasoning:{ticker}:{hash(tuple(headlines))}"
        if not fresh:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        headlines = [headline for headline in headlines if headline][:5]
        if not headlines:
            result = {"score": 50.0, "reason": "No recent headlines were available."}
            self.cache.set(cache_key, result, ttl_seconds=self.config.cache_ttls.gpt_reasoning)
            return result
        if self.client is None:
            result = self._heuristic_reasoning(headlines)
            self.cache.set(cache_key, result, ttl_seconds=self.config.cache_ttls.gpt_reasoning)
            return result
        prompt = (
            "You are scoring whether a stock is likely to rise at least 4% within the next 5 trading days.\n"
            "Read these recent headlines and return strict JSON with keys: score, reason.\n"
            "score must be a number 0-100.\n"
            "reason must be one concise sentence.\n"
            f"Ticker: {ticker}\n"
            f"Headlines: {json.dumps(headlines)}"
        )
        try:
            if hasattr(self.client, "responses"):
                response = self.client.responses.create(model=self.config.openai_model, input=prompt)
                content = response.output_text
            else:  # pragma: no cover
                response = self.client.chat.completions.create(
                    model=self.config.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                content = response.choices[0].message.content
            payload = json.loads(content)
            result = {
                "score": clamp(float(payload.get("score", 50.0)), 0.0, 100.0),
                "reason": str(payload.get("reason", "")),
            }
            self.cache.set(cache_key, result, ttl_seconds=self.config.cache_ttls.gpt_reasoning)
            return result
        except Exception:
            LOGGER.debug("GPT reasoning failed for %s", ticker, exc_info=True)
            result = self._heuristic_reasoning(headlines)
            self.cache.set(cache_key, result, ttl_seconds=self.config.cache_ttls.gpt_reasoning)
            return result

    def _heuristic_reasoning(self, headlines: List[str]) -> Dict[str, object]:
        joined = " ".join(headlines).lower()
        positive = sum(term in joined for term in ["beats", "contract", "guidance", "buyback", "approval", "record"])
        negative = sum(term in joined for term in ["investigation", "miss", "downgrade", "lawsuit", "offering"])
        score = clamp(50 + (positive - negative) * 8, 0.0, 100.0)
        if positive > negative:
            reason = "Recent headlines skew positive and point to a near-term catalyst."
        elif negative > positive:
            reason = "Recent headlines include negative cues that weaken the weekly upside thesis."
        else:
            reason = "Recent headlines are mixed and do not strongly change the weekly thesis."
        return {"score": score, "reason": reason}
