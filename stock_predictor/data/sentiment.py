"""Social and search sentiment scoring."""

from __future__ import annotations

import logging
import io
import os
import re
from contextlib import redirect_stderr, redirect_stdout
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from importlib.util import find_spec
from typing import Any, Dict, Iterable, List

import requests

try:  # pragma: no cover - optional dependency
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:  # pragma: no cover
    class SentimentIntensityAnalyzer:  # type: ignore[override]
        def polarity_scores(self, text: str) -> Dict[str, float]:
            text = text.lower()
            bullish = sum(word in text for word in ["beat", "bull", "upgrade", "buyback", "contract", "record"])
            bearish = sum(word in text for word in ["miss", "downgrade", "fraud", "lawsuit", "recall", "offering"])
            compound = 0.0
            if bullish or bearish:
                compound = max(min((bullish - bearish) / 5.0, 1.0), -1.0)
            return {"compound": compound}

from stock_predictor.config import AppConfig
from stock_predictor.data.cache import SQLiteCache
from stock_predictor.utils import clamp, coerce_float

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional heavy dependency
    from transformers import pipeline
except Exception:  # pragma: no cover - optional heavy dependency
    pipeline = None

TORCH_AVAILABLE = find_spec("torch") is not None

try:  # pragma: no cover - optional dependency
    import praw
except Exception:  # pragma: no cover
    praw = None

try:  # pragma: no cover - optional dependency
    from pytrends.request import TrendReq
except Exception:  # pragma: no cover
    TrendReq = None


CASHTAG_PATTERN = re.compile(r"\$([A-Z]{1,5})\b")
TOKEN_PATTERN = re.compile(r"\b[A-Z]{2,5}\b")


class SentimentEngine:
    """Gather Reddit, X, and Google Trends signals."""

    def __init__(self, config: AppConfig, cache: SQLiteCache) -> None:
        self.config = config
        self.cache = cache
        self.vader = SentimentIntensityAnalyzer()
        self._finbert = None

    @property
    def finbert(self):  # type: ignore[no-untyped-def]
        if self._finbert is None and (pipeline is not None and TORCH_AVAILABLE):
            try:
                if os.getenv("STOCK_PREDICTOR_QUIET_RUNTIME") == "1":
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        self._finbert = pipeline(
                            "text-classification",
                            model=self.config.finbert_model_name,
                            tokenizer=self.config.finbert_model_name,
                            truncation=True,
                        )
                else:
                    self._finbert = pipeline(
                        "text-classification",
                        model=self.config.finbert_model_name,
                        tokenizer=self.config.finbert_model_name,
                        truncation=True,
                    )
            except Exception:
                LOGGER.debug("FinBERT pipeline unavailable in current environment", exc_info=True)
                self._finbert = False
        elif self._finbert is None:
            self._finbert = False
        return self._finbert

    def fetch_reddit_mentions(self, fresh: bool = False) -> List[Dict[str, Any]]:
        cache_key = "reddit-mentions"
        if not fresh:
            cached = self.cache.get(cache_key)
            if cached:
                return cached

        posts = []
        if self.config.reddit_client_id and self.config.reddit_client_secret and praw is not None:
            posts = self._fetch_reddit_via_praw()
        else:
            posts = self._fetch_reddit_via_json()
        self.cache.set(cache_key, posts, ttl_seconds=self.config.cache_ttls.reddit_mentions)
        return posts

    def _fetch_reddit_via_praw(self) -> List[Dict[str, Any]]:
        reddit = praw.Reddit(
            client_id=self.config.reddit_client_id,
            client_secret=self.config.reddit_client_secret,
            user_agent=self.config.reddit_user_agent,
        )
        posts: List[Dict[str, Any]] = []
        for subreddit_name in self.config.reddit_subreddits:
            subreddit = reddit.subreddit(subreddit_name)
            for submission in subreddit.new(limit=100):
                posts.append(
                    {
                        "source": "reddit",
                        "subreddit": subreddit_name,
                        "title": submission.title,
                        "body": submission.selftext or "",
                        "created_utc": submission.created_utc,
                        "score": submission.score,
                        "num_comments": submission.num_comments,
                    }
                )
        return posts

    def _fetch_reddit_via_json(self) -> List[Dict[str, Any]]:
        headers = {"User-Agent": self.config.reddit_user_agent}
        posts: List[Dict[str, Any]] = []
        for subreddit in self.config.reddit_subreddits:
            url = f"https://www.reddit.com/r/{subreddit}/new.json?limit=100"
            try:
                response = requests.get(url, headers=headers, timeout=20)
                response.raise_for_status()
                children = response.json().get("data", {}).get("children", [])
                for child in children:
                    data = child.get("data", {})
                    posts.append(
                        {
                            "source": "reddit",
                            "subreddit": subreddit,
                            "title": data.get("title", ""),
                            "body": data.get("selftext", ""),
                            "created_utc": data.get("created_utc"),
                            "score": data.get("score"),
                            "num_comments": data.get("num_comments"),
                        }
                    )
            except Exception:
                LOGGER.debug("reddit json fetch failed for %s", subreddit, exc_info=True)
        return posts

    def fetch_x_mentions(self, tickers: Iterable[str], fresh: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        cache_key = "x-mentions"
        if not fresh:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        if not self.config.x_search_endpoint:
            return {ticker: [] for ticker in tickers}
        payload: Dict[str, List[Dict[str, Any]]] = {ticker: [] for ticker in tickers}
        headers = {"Authorization": f"Bearer {self.config.x_search_token}"} if self.config.x_search_token else {}
        for ticker in tickers:
            try:
                response = requests.get(
                    self.config.x_search_endpoint,
                    params={"q": f"${ticker}", "hours": 48},
                    headers=headers,
                    timeout=20,
                )
                response.raise_for_status()
                data = response.json()
                payload[ticker] = data.get("results", [])
            except Exception:
                LOGGER.debug("x mentions unavailable for %s", ticker, exc_info=True)
        self.cache.set(cache_key, payload, ttl_seconds=self.config.cache_ttls.x_mentions)
        return payload

    def fetch_google_trends(self, tickers: Iterable[str], fresh: bool = False) -> Dict[str, float]:
        cache_key = "google-trends"
        if not fresh:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        if TrendReq is None:
            return {ticker: 0.0 for ticker in tickers}
        pytrends = TrendReq(hl="en-US", tz=360)
        result: Dict[str, float] = {}
        for ticker in tickers:
            try:
                pytrends.build_payload([ticker], timeframe="now 7-d", geo="US")
                frame = pytrends.interest_over_time()
                if frame.empty:
                    result[ticker] = 0.0
                    continue
                series = frame[ticker]
                recent = series.tail(12).mean()
                prior = series.head(max(len(series) - 12, 1)).mean()
                result[ticker] = coerce_float((recent - prior) / max(prior, 1))
            except Exception:
                LOGGER.debug("google trends unavailable for %s", ticker, exc_info=True)
                result[ticker] = 0.0
        self.cache.set(cache_key, result, ttl_seconds=self.config.cache_ttls.google_trends)
        return result

    def score_sentiment(
        self,
        tickers: Iterable[str],
        fresh: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        tickers = list(tickers)
        reddit_posts = self.fetch_reddit_mentions(fresh=fresh)
        x_posts = self.fetch_x_mentions(tickers, fresh=fresh)
        google_trends = self.fetch_google_trends(tickers, fresh=fresh)
        now = datetime.now(timezone.utc)
        horizon_24h = now - timedelta(hours=24)
        horizon_48h = now - timedelta(hours=48)

        summaries: Dict[str, Dict[str, float]] = {
            ticker: {
                "vader": 0.0,
                "finbert": 0.0,
                "mention_velocity": 0.0,
                "mention_count_24h": 0.0,
                "mention_count_48h": 0.0,
                "x_mentions": float(len(x_posts.get(ticker, []))),
                "google_trends_delta": google_trends.get(ticker, 0.0),
                "sentiment_score": self.config.default_missing_signal_value / 100.0,
                "velocity_signal": 0.0,
                "has_data": False,
                "defaulted": True,
            }
            for ticker in tickers
        }

        texts_by_ticker: Dict[str, List[str]] = defaultdict(list)
        counts_24: Counter[str] = Counter()
        counts_48: Counter[str] = Counter()
        for post in reddit_posts:
            text = f'{post.get("title", "")}\n{post.get("body", "")}'.strip()
            mentioned = self.extract_tickers(text, tickers)
            created = datetime.fromtimestamp(post.get("created_utc", 0), tz=timezone.utc)
            for ticker in mentioned:
                texts_by_ticker[ticker].append(text)
                counts_48[ticker] += 1
                if created >= horizon_24h:
                    counts_24[ticker] += 1
                elif created >= horizon_48h:
                    counts_48[ticker] += 0

        finbert_scores = self._score_finbert_batch(texts_by_ticker)
        for ticker in tickers:
            texts = texts_by_ticker[ticker]
            vader_score = self._score_vader(texts)
            mention_24 = float(counts_24[ticker] + len(x_posts.get(ticker, [])))
            mention_48 = float(max(counts_48[ticker], 1))
            velocity = mention_24 / max(mention_48 - counts_24[ticker], 1)
            signal = 1.0 if velocity >= self.config.thresholds.sentiment_velocity_bullish else velocity / 3.0
            summaries[ticker].update(
                {
                    "vader": vader_score,
                    "finbert": finbert_scores.get(ticker, 0.0),
                    "mention_velocity": velocity,
                    "mention_count_24h": mention_24,
                    "mention_count_48h": mention_48,
                    "velocity_signal": clamp(signal, 0.0, 1.0),
                }
            )
            has_data = bool(texts or x_posts.get(ticker) or abs(google_trends.get(ticker, 0.0)) > 0.01)
            summaries[ticker]["has_data"] = has_data
            summaries[ticker]["defaulted"] = not has_data
            if not has_data:
                summaries[ticker]["sentiment_score"] = self.config.default_missing_signal_value / 100.0
                continue
            summaries[ticker]["sentiment_score"] = clamp(
                (
                    0.35 * (vader_score + 1.0) / 2.0
                    + 0.35 * (finbert_scores.get(ticker, 0.0) + 1.0) / 2.0
                    + 0.2 * summaries[ticker]["velocity_signal"]
                    + 0.1 * clamp((google_trends.get(ticker, 0.0) + 1.0) / 2.0, 0.0, 1.0)
                ),
                0.0,
                1.0,
            )
        return summaries

    def _score_vader(self, texts: Iterable[str]) -> float:
        scores = [self.vader.polarity_scores(text)["compound"] for text in texts if text]
        if not scores:
            return 0.0
        return float(sum(scores) / len(scores))

    def _score_finbert_batch(self, texts_by_ticker: Dict[str, List[str]]) -> Dict[str, float]:
        if self.finbert in {None, False}:
            return {ticker: 0.0 for ticker in texts_by_ticker}
        results: Dict[str, float] = {}
        for ticker, texts in texts_by_ticker.items():
            if not texts:
                results[ticker] = 0.0
                continue
            try:
                outputs = self.finbert(texts[:20])
            except Exception:
                LOGGER.debug("finbert failed for %s", ticker, exc_info=True)
                results[ticker] = 0.0
                continue
            score = 0.0
            for item in outputs:
                label = str(item["label"]).lower()
                value = coerce_float(item["score"])
                if "positive" in label:
                    score += value
                elif "negative" in label:
                    score -= value
            results[ticker] = clamp(score / max(len(outputs), 1), -1.0, 1.0)
        return results

    @staticmethod
    def extract_tickers(text: str, universe: Iterable[str]) -> List[str]:
        universe_set = {ticker.upper() for ticker in universe}
        found = set(CASHTAG_PATTERN.findall(text))
        found.update(token for token in TOKEN_PATTERN.findall(text) if token in universe_set)
        return sorted(found & universe_set)
