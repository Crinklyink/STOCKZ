"""Market, fundamentals, and news data fetching."""

from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import StringIO
from typing import Any, Dict, Iterable, List

import pandas as pd
import requests
import yfinance as yf
from yfinance import data as yf_data

from stock_predictor.config import AppConfig
from stock_predictor.data.cache import SQLiteCache
from stock_predictor.utils import coerce_float

LOGGER = logging.getLogger(__name__)
YF_LOCK = threading.RLock()


@dataclass(slots=True)
class TickerBundle:
    ticker: str
    sector: str
    info: Dict[str, Any]
    daily: pd.DataFrame
    hourly: pd.DataFrame
    news: List[Dict[str, Any]]


class MarketDataFetcher:
    """Fetch price, options, and news data with caching."""

    def __init__(self, config: AppConfig, cache: SQLiteCache) -> None:
        self.config = config
        self.cache = cache

    def download_history(
        self,
        ticker: str,
        *,
        interval: str,
        period: str,
        ttl_seconds: int,
        fresh: bool = False,
        cache_only: bool = False,
    ) -> pd.DataFrame:
        cache_key = f"history:{ticker}:{interval}:{period}"
        if not fresh:
            cached = self.cache.get(cache_key)
            if cached:
                return pd.DataFrame(cached).pipe(self._postprocess_history)
        if cache_only:
            return pd.DataFrame()

        history = pd.DataFrame()
        delays = [2, 4, 8]
        last_error = ""
        crumb_retry_attempted = False
        for delay in delays:
            try:
                with YF_LOCK:
                    history = yf.download(
                        tickers=ticker,
                        period=period,
                        interval=interval,
                        auto_adjust=False,
                        progress=False,
                        threads=False,
                    )
                if history is not None and not history.empty:
                    break
            except Exception as exc:
                last_error = str(exc)
                self._log_provider_failure("yahoo", ticker, f"download failed [{interval} {period}]: {exc}")
                if self._is_yahoo_auth_error(exc) and not crumb_retry_attempted:
                    self._reset_yfinance_session()
                    crumb_retry_attempted = True
            try:
                with YF_LOCK:
                    history = yf.Ticker(ticker).history(
                        period=period,
                        interval=interval,
                        auto_adjust=False,
                    )
                if history is not None and not history.empty:
                    break
            except Exception as exc:
                last_error = str(exc)
                self._log_provider_failure("yahoo", ticker, f"history failed [{interval} {period}]: {exc}")
                if self._is_yahoo_auth_error(exc) and not crumb_retry_attempted:
                    self._reset_yfinance_session()
                    crumb_retry_attempted = True
            time.sleep(delay)
        if (history is None or history.empty) and self.config.alpha_vantage_api_key:
            history = self._download_history_alpha_vantage(ticker, interval=interval)
            if history.empty:
                self._log_provider_failure("alpha_vantage", ticker, f"no data [{interval} {period}] after yahoo retries: {last_error}")
        if history is None or history.empty:
            return pd.DataFrame()
        history = self._postprocess_history(history)
        self.cache.set(
            cache_key,
            history.reset_index().to_dict(orient="records"),
            ttl_seconds=ttl_seconds,
        )
        return history

    def _postprocess_history(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        normalized = frame.copy()
        if isinstance(normalized.columns, pd.MultiIndex):
            normalized.columns = normalized.columns.get_level_values(0)
        normalized.columns = [str(col).lower() for col in normalized.columns]
        index_column = next(
            (
                column
                for column in ("datetime", "date", "index")
                if column in normalized.columns
            ),
            None,
        )
        if index_column is not None:
            normalized[index_column] = pd.to_datetime(normalized[index_column], utc=True, errors="coerce")
            normalized = normalized.dropna(subset=[index_column]).set_index(index_column)
        normalized = normalized.rename(
            columns={
                "adj close": "adj_close",
                "stock splits": "stock_splits",
            }
        )
        normalized = normalized.loc[:, ~pd.Index(normalized.columns).duplicated(keep="last")]
        normalized.index = pd.to_datetime(normalized.index, utc=True, errors="coerce")
        normalized = normalized[~normalized.index.isna()]
        normalized = normalized[~normalized.index.duplicated(keep="last")]
        return normalized.sort_index()

    def fetch_info(self, ticker: str, fresh: bool = False) -> Dict[str, Any]:
        cache_key = f"info:{ticker}"
        if not fresh:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        info: Dict[str, Any] = {}
        yf_ticker = None
        try:
            with YF_LOCK:
                yf_ticker = yf.Ticker(ticker)
                info.update(yf_ticker.fast_info or {})
        except Exception:  # pragma: no cover - network/provider variability
            LOGGER.debug("fast_info unavailable for %s", ticker, exc_info=True)
            self._log_provider_failure("yahoo", ticker, "fast_info unavailable")
        try:
            with YF_LOCK:
                yf_ticker = yf_ticker or yf.Ticker(ticker)
                info.update(yf_ticker.info or {})
        except Exception:
            LOGGER.debug("info unavailable for %s", ticker, exc_info=True)
            self._log_provider_failure("yahoo", ticker, "info unavailable")
        self.cache.set(cache_key, info, ttl_seconds=self.config.cache_ttls.info)
        return info

    def fetch_news(self, ticker: str, fresh: bool = False) -> List[Dict[str, Any]]:
        cache_key = f"news:{ticker}"
        if not fresh:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        news = []
        try:
            with YF_LOCK:
                news = yf.Ticker(ticker).news or []
        except Exception:
            LOGGER.debug("news unavailable for %s", ticker, exc_info=True)
            self._log_provider_failure("yahoo", ticker, "news unavailable")
        self.cache.set(cache_key, news, ttl_seconds=self.config.cache_ttls.news)
        return news

    def fetch_earnings_dates(self, ticker: str, fresh: bool = False) -> List[Dict[str, Any]]:
        cache_key = f"earnings-dates:{ticker}"
        if not fresh:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        records: List[Dict[str, Any]] = []
        try:
            with YF_LOCK:
                yf_ticker = yf.Ticker(ticker)
                dates = yf_ticker.get_earnings_dates(limit=8)
            if dates is not None and not dates.empty:
                frame = dates.reset_index()
                if "Earnings Date" in frame.columns:
                    frame = frame.rename(columns={"Earnings Date": "earnings_date"})
                records = frame.to_dict(orient="records")
        except Exception:
            LOGGER.debug("earnings dates unavailable for %s", ticker, exc_info=True)
            self._log_provider_failure("yahoo", ticker, "earnings dates unavailable")
        self.cache.set(cache_key, records, ttl_seconds=self.config.cache_ttls.earnings_dates)
        return records

    def fetch_earnings_dates_for_tickers(
        self,
        tickers: Iterable[str],
        *,
        fresh: bool = False,
    ) -> Dict[str, List[Dict[str, Any]]]:
        results: Dict[str, List[Dict[str, Any]]] = {}
        tickers = sorted(set(tickers))
        with ThreadPoolExecutor(max_workers=self.config.max_threads) as executor:
            tasks = {
                executor.submit(self.fetch_earnings_dates, ticker, fresh): ticker
                for ticker in tickers
            }
            for future in as_completed(tasks):
                results[tasks[future]] = future.result()
        return results

    def fetch_options_chain(self, ticker: str, fresh: bool = False) -> Dict[str, Any]:
        cache_key = f"options:{ticker}"
        if not fresh:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        payload: Dict[str, Any] = {"expirations": [], "chains": {}}
        try:
            with YF_LOCK:
                stock = yf.Ticker(ticker)
                expirations = list(stock.options or [])
        except Exception:
            expirations = []
            self._log_provider_failure("yahoo", ticker, "options expirations unavailable")
        payload["expirations"] = expirations[:4]
        for expiry in payload["expirations"]:
            try:
                with YF_LOCK:
                    stock = yf.Ticker(ticker)
                    chain = stock.option_chain(expiry)
            except Exception:
                self._log_provider_failure("yahoo", ticker, f"options chain unavailable for {expiry}")
                continue
            payload["chains"][expiry] = {
                "calls": chain.calls.to_dict(orient="records"),
                "puts": chain.puts.to_dict(orient="records"),
            }
        self.cache.set(cache_key, payload, ttl_seconds=self.config.cache_ttls.premium_flow)
        return payload

    def fetch_sector_history(self, fresh: bool = False) -> Dict[str, pd.DataFrame]:
        histories: Dict[str, pd.DataFrame] = {}
        for sector, etf in self.config.sector_etfs.items():
            histories[sector] = self.download_history(
                etf,
                interval="1d",
                period="6mo",
                ttl_seconds=self.config.cache_ttls.market_history,
                fresh=fresh,
            )
        return histories

    def fetch_sector_history_for_period(self, period: str, fresh: bool = False) -> Dict[str, pd.DataFrame]:
        histories: Dict[str, pd.DataFrame] = {}
        for sector, etf in self.config.sector_etfs.items():
            histories[sector] = self.download_history(
                etf,
                interval="1d",
                period=period,
                ttl_seconds=self.config.cache_ttls.market_history,
                fresh=fresh,
            )
        return histories

    def fetch_macro_history(self, ticker: str, fresh: bool = False, period: str = "6mo") -> pd.DataFrame:
        if not ticker:
            return pd.DataFrame()
        return self.download_history(
            ticker,
            interval="1d",
            period=period,
            ttl_seconds=self.config.cache_ttls.macro,
            fresh=fresh,
        )

    def calculate_market_breadth(self, price_frames: Dict[str, pd.DataFrame]) -> float:
        close_frame = self._build_close_matrix(price_frames)
        if close_frame.empty:
            return 0.5
        sma50 = close_frame.rolling(50, min_periods=50).mean()
        latest_close = close_frame.iloc[-1]
        latest_sma50 = sma50.iloc[-1]
        valid = latest_close.notna() & latest_sma50.notna()
        if not bool(valid.any()):
            return 0.5
        return float((latest_close[valid] > latest_sma50[valid]).mean())

    def calculate_breadth_history(self, price_frames: Dict[str, pd.DataFrame]) -> pd.Series:
        close_frame = self._build_close_matrix(price_frames)
        if close_frame.empty:
            return pd.Series(dtype=float)
        sma50 = close_frame.rolling(50, min_periods=50).mean()
        valid = close_frame.notna() & sma50.notna()
        breadth = (close_frame > sma50).where(valid).mean(axis=1, skipna=True).fillna(0.5)
        return breadth.astype(float).sort_index()

    def _build_close_matrix(self, price_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        series = {
            ticker: frame["close"]
            for ticker, frame in price_frames.items()
            if frame is not None and not frame.empty and "close" in frame
        }
        if not series:
            return pd.DataFrame()
        return pd.DataFrame(series).sort_index().ffill()

    def fetch_sec_filings(self, ticker: str, fresh: bool = False) -> List[Dict[str, Any]]:
        cache_key = f"sec-filings:{ticker}"
        if not fresh:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        headers = {
            "User-Agent": self.config.reddit_user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov",
        }
        cik_map = self.fetch_sec_ticker_map(fresh=fresh)
        record = cik_map.get(ticker.upper())
        if not record:
            return []
        cik = str(record["cik_str"]).zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        filings: List[Dict[str, Any]] = []
        try:
            response = requests.get(url, timeout=20, headers=headers)
            response.raise_for_status()
            recent = response.json().get("filings", {}).get("recent", {})
            if recent:
                frame = pd.DataFrame(recent)
                if not frame.empty:
                    filings = frame.head(30).to_dict(orient="records")
        except Exception:
            LOGGER.debug("SEC filings unavailable for %s", ticker, exc_info=True)
        self.cache.set(cache_key, filings, ttl_seconds=self.config.cache_ttls.sec_filings)
        return filings

    def fetch_sec_ticker_map(self, fresh: bool = False) -> Dict[str, Dict[str, Any]]:
        cache_key = "sec-ticker-map"
        if not fresh:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        email = self.config.alert_email if self.config.alert_email else "user@example.com"
        headers = {"User-Agent": f"StockPredictorApp {email}"}
        url = "https://www.sec.gov/files/company_tickers.json"
        mapping: Dict[str, Dict[str, Any]] = {}
        try:
            response = requests.get(url, timeout=20, headers=headers)
            response.raise_for_status()
            payload = response.json()
            for _, record in payload.items():
                mapping[record["ticker"].upper()] = record
            if mapping:
                self.cache.set(cache_key, mapping, ttl_seconds=60 * 60 * 24 * 7)
        except Exception:
            LOGGER.debug("SEC ticker map unavailable", exc_info=True)
        return mapping

    def _reset_yfinance_session(self) -> None:
        try:
            with YF_LOCK:
                session = yf_data.requests.Session(impersonate="chrome")
                yf_data.YfData(session=session)
            LOGGER.info("Reset yfinance session after Yahoo auth failure")
        except Exception:
            LOGGER.debug("Failed to reset yfinance session", exc_info=True)

    @staticmethod
    def _is_yahoo_auth_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "401" in message or "invalid crumb" in message

    def fetch_ticker_bundle(
        self,
        ticker: str,
        sector: str | None,
        fresh: bool = False,
        daily_frame: pd.DataFrame | None = None,
    ) -> TickerBundle:
        info = self.fetch_info(ticker, fresh=fresh)
        return TickerBundle(
            ticker=ticker,
            sector=str(info.get("sector") or sector or "Unknown"),
            info=info,
            daily=daily_frame if daily_frame is not None else self.download_history(
                ticker,
                interval="1d",
                period=self.config.daily_period,
                ttl_seconds=self.config.cache_ttls.market_history,
                fresh=fresh,
            ),
            hourly=self.download_history(
                ticker,
                interval=self.config.hourly_interval,
                period=self.config.intraday_period,
                ttl_seconds=self.config.cache_ttls.intraday_history,
                fresh=fresh,
            ),
            news=self.fetch_news(ticker, fresh=fresh),
        )

    def fetch_universe_bundles(self, fresh: bool = False, mode: str = "custom") -> Dict[str, TickerBundle]:
        tasks = []
        results: Dict[str, TickerBundle] = {}
        universe = self.resolve_universe(mode=mode, fresh=fresh)
        with ThreadPoolExecutor(max_workers=self.config.max_threads) as executor:
            for sector, tickers in universe.items():
                for ticker in tickers:
                    tasks.append(
                        executor.submit(
                            self.fetch_ticker_bundle,
                            ticker,
                            sector,
                            fresh,
                        )
                    )
            for future in as_completed(tasks):
                bundle = future.result()
                results[bundle.ticker] = bundle
        return results

    def fetch_universe_daily_frames(
        self,
        fresh: bool = False,
        mode: str = "mini",
        *,
        period: str | None = None,
    ) -> Dict[str, pd.DataFrame]:
        return self.fetch_daily_frames_for_tickers(
            sorted(
                {
                    ticker
                    for sector_tickers in self.resolve_universe(mode=mode, fresh=fresh).values()
                    for ticker in sector_tickers
                }
            ),
            fresh=fresh,
            period=period,
        )

    def fetch_daily_frames_for_tickers(
        self,
        tickers: Iterable[str],
        *,
        fresh: bool = False,
        cache_only: bool = False,
        period: str | None = None,
    ) -> Dict[str, pd.DataFrame]:
        frames: Dict[str, pd.DataFrame] = {}
        tickers = sorted(set(tickers))
        resolved_period = period or self.config.daily_period
        with ThreadPoolExecutor(max_workers=self.config.max_threads) as executor:
            tasks = {
                executor.submit(
                    self.download_history,
                    ticker,
                    interval="1d",
                    period=resolved_period,
                    ttl_seconds=self.config.cache_ttls.market_history,
                    fresh=fresh,
                    cache_only=cache_only,
                ): ticker
                for ticker in tickers
            }
            for future in as_completed(tasks):
                frames[tasks[future]] = future.result()
        return frames

    def fetch_selected_bundles(
        self,
        ticker_sector_map: Dict[str, str],
        *,
        daily_frames: Dict[str, pd.DataFrame] | None = None,
        fresh: bool = False,
    ) -> Dict[str, TickerBundle]:
        tasks = {}
        results: Dict[str, TickerBundle] = {}
        with ThreadPoolExecutor(max_workers=self.config.max_threads) as executor:
            for ticker, sector in ticker_sector_map.items():
                tasks[
                    executor.submit(
                        self.fetch_ticker_bundle,
                        ticker,
                        sector,
                        fresh,
                        None if daily_frames is None else daily_frames.get(ticker),
                    )
                ] = ticker
            for future in as_completed(tasks):
                bundle = future.result()
                results[bundle.ticker] = bundle
        return results

    def resolve_universe(self, mode: str = "custom", fresh: bool = False) -> Dict[str, List[str]]:
        normalized_mode = {
            "custom": "mini",
            "mini": "mini",
            "sp500": "sp500",
            "nasdaq": "nasdaq100",
            "nasdaq100": "nasdaq100",
            "full": "full",
            "us_market": "us_market",
        }.get(mode, mode)
        if normalized_mode == "mini":
            return self.config.sector_universe
        if normalized_mode == "sp500":
            return self._cached_universe_table("sp500-universe", "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        if normalized_mode == "nasdaq100":
            return self._cached_universe_table("nasdaq100-universe", "https://en.wikipedia.org/wiki/Nasdaq-100")
        if normalized_mode == "us_market":
            mapping = self.fetch_sec_ticker_map(fresh=fresh)
            return {"US Market": [record["ticker"].upper() for record in mapping.values()]}
        full: Dict[str, List[str]] = {}
        for source in [
            self._cached_universe_table("sp500-universe", "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"),
            self._cached_universe_table("nasdaq100-universe", "https://en.wikipedia.org/wiki/Nasdaq-100"),
            {"Small Caps": self.config.small_cap_universe},
            {"Most Shorted": self.config.most_shorted_universe},
        ]:
            for sector, tickers in source.items():
                full.setdefault(sector, [])
                full[sector].extend(tickers)
        return {sector: sorted(set(tickers)) for sector, tickers in full.items()}

    def _cached_universe_table(self, cache_key: str, url: str) -> Dict[str, List[str]]:
        cached = self.cache.get(cache_key)
        if cached and self._universe_size_ok(cache_key, cached):
            return cached
        result: Dict[str, List[str]] = {}
        try:
            html = self._download_universe_html(url)
            tables = pd.read_html(StringIO(html))
            table = self._select_universe_table(tables)
            if table.empty:
                return {}
            symbol_column = self._find_symbol_column(table)
            sector_column = self._find_sector_column(table)
            if symbol_column is None:
                return {}
            for _, row in table.iterrows():
                ticker = str(row.get(symbol_column, "")).replace(".", "-").upper().strip()
                sector = str(row.get(sector_column, "Expanded")) if sector_column else "Expanded"
                if not ticker or ticker == "NAN":
                    continue
                result.setdefault(sector, []).append(ticker)
        except Exception as exc:
            self._log_provider_failure("wikipedia", cache_key, str(exc))
            return {}
        self.cache.set(cache_key, result, ttl_seconds=self.config.cache_ttls.universe_lists)
        return result

    def _download_universe_html(self, url: str) -> str:
        response = requests.get(
            url,
            timeout=20,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; stock-predictor/1.0)",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        response.raise_for_status()
        return response.text

    def _select_universe_table(self, tables: List[pd.DataFrame]) -> pd.DataFrame:
        for candidate in tables:
            normalized = self._normalize_universe_table(candidate)
            if self._find_symbol_column(normalized):
                return normalized
        return pd.DataFrame()

    def _normalize_universe_table(self, table: pd.DataFrame) -> pd.DataFrame:
        normalized = table.copy()
        if isinstance(normalized.columns, pd.MultiIndex):
            normalized.columns = [
                " ".join(str(part) for part in column if str(part) != "nan").strip()
                for column in normalized.columns
            ]
        normalized.columns = [self._clean_column_name(column) for column in normalized.columns]
        return normalized

    @staticmethod
    def _clean_column_name(column: object) -> str:
        text = re.sub(r"\[\d+\]", "", str(column))
        return re.sub(r"\s+", " ", text).strip()

    def _find_symbol_column(self, table: pd.DataFrame) -> str | None:
        for candidate in table.columns:
            lower = str(candidate).strip().lower()
            if lower in {"symbol", "ticker"} or "symbol" in lower or "ticker" in lower:
                return str(candidate)
        return None

    def _find_sector_column(self, table: pd.DataFrame) -> str | None:
        preferred = [
            "gics sector",
            "icb industry",
            "sector",
            "industry",
        ]
        lowered = {str(column).lower(): str(column) for column in table.columns}
        for key in preferred:
            if key in lowered:
                return lowered[key]
        for candidate in table.columns:
            lower = str(candidate).lower()
            if "sector" in lower or "industry" in lower:
                return str(candidate)
        return None

    @staticmethod
    def _universe_size_ok(cache_key: str, universe: Dict[str, List[str]]) -> bool:
        total = sum(len(tickers) for tickers in universe.values())
        minimum = 0
        if "sp500" in cache_key:
            minimum = 450
        elif "nasdaq100" in cache_key:
            minimum = 80
        return total >= minimum

    def warm_cache(self, mode: str = "full", fresh: bool = False) -> Dict[str, int]:
        tickers = sorted(
            {
                ticker
                for sector_tickers in self.resolve_universe(mode=mode, fresh=fresh).values()
                for ticker in sector_tickers
            }
        )
        ready = 0
        failed = 0
        with ThreadPoolExecutor(max_workers=self.config.max_threads) as executor:
            tasks = {
                executor.submit(
                    self._warm_single_ticker,
                    ticker,
                    fresh,
                ): ticker
                for ticker in tickers
            }
            for future in as_completed(tasks):
                success = future.result()
                if success:
                    ready += 1
                else:
                    failed += 1
        return {"ready": ready, "failed": failed, "total": len(tickers)}

    def _warm_single_ticker(self, ticker: str, fresh: bool) -> bool:
        daily = self.download_history(
            ticker,
            interval="1d",
            period=self.config.daily_period,
            ttl_seconds=self.config.cache_ttls.market_history,
            fresh=fresh,
        )
        hourly = self.download_history(
            ticker,
            interval=self.config.hourly_interval,
            period=self.config.intraday_period,
            ttl_seconds=self.config.cache_ttls.intraday_history,
            fresh=fresh,
        )
        return not daily.empty and not hourly.empty

    def cache_stats(self) -> Dict[str, int]:
        with self.cache.connection() as conn:
            total_rows = conn.execute("SELECT COUNT(*) AS count FROM cache_entries").fetchone()["count"]
            history_rows = conn.execute(
                "SELECT COUNT(*) AS count FROM cache_entries WHERE cache_key LIKE 'history:%'"
            ).fetchone()["count"]
            universe_rows = conn.execute(
                "SELECT COUNT(*) AS count FROM cache_entries WHERE cache_key LIKE '%universe%'"
            ).fetchone()["count"]
        return {
            "cache_rows": int(total_rows),
            "history_rows": int(history_rows),
            "universe_rows": int(universe_rows),
        }

    def _download_history_alpha_vantage(self, ticker: str, *, interval: str) -> pd.DataFrame:
        function = "TIME_SERIES_DAILY_ADJUSTED"
        params = {
            "function": function,
            "symbol": ticker,
            "apikey": self.config.alpha_vantage_api_key,
            "outputsize": "full",
        }
        if interval != "1d":
            params["function"] = "TIME_SERIES_INTRADAY"
            params["interval"] = "60min"
            params["outputsize"] = "full"
        try:
            response = requests.get(self.config.alpha_vantage_base_url, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            key = next((name for name in payload if "Time Series" in name), None)
            if key is None:
                return pd.DataFrame()
            frame = pd.DataFrame.from_dict(payload[key], orient="index")
            frame.index = pd.to_datetime(frame.index, utc=True)
            rename_map = {
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
                "5. volume": "volume",
                "5. adjusted close": "adj_close",
                "6. volume": "volume",
            }
            frame = frame.rename(columns=rename_map)
            for column in ["open", "high", "low", "close", "volume"]:
                if column in frame.columns:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce")
            return frame[["open", "high", "low", "close", "volume"]].dropna().sort_index()
        except Exception as exc:
            self._log_provider_failure("alpha_vantage", ticker, str(exc))
            return pd.DataFrame()

    def _log_provider_failure(self, provider: str, ticker: str, reason: str) -> None:
        try:
            self.config.data_quality_log.parent.mkdir(parents=True, exist_ok=True)
            with self.config.data_quality_log.open("a", encoding="utf-8") as handle:
                handle.write(f"{datetime.now(timezone.utc).isoformat()} | {provider} | {ticker} | {reason}\n")
        except Exception:
            LOGGER.debug("Failed to write provider failure log", exc_info=True)

    @staticmethod
    def summarize_company(bundle: TickerBundle) -> Dict[str, Any]:
        info = bundle.info
        daily = bundle.daily
        price_fallback = coerce_float(daily["close"].iloc[-1]) if not daily.empty else 0.0
        avg_volume_fallback = coerce_float(daily["volume"].tail(20).mean()) if not daily.empty else 0.0
        return {
            "ticker": bundle.ticker,
            "sector": bundle.sector,
            "price": coerce_float(info.get("currentPrice") or info.get("lastPrice"), default=price_fallback),
            "market_cap": coerce_float(info.get("marketCap")),
            "average_volume": coerce_float(
                info.get("averageVolume")
                or info.get("averageDailyVolume3Month")
                or info.get("tenDayAverageVolume")
                or avg_volume_fallback
            ),
            "shares_short_pct": coerce_float(
                info.get("sharesPercentSharesOut") or info.get("shortPercentOfFloat")
            ),
            "beta": coerce_float(info.get("beta")),
            "earnings_timestamp": info.get("earningsTimestamp"),
            "next_earnings_timestamp": info.get("earningsTimestampStart"),
            "target_mean_price": coerce_float(info.get("targetMeanPrice")),
            "recommendation": info.get("recommendationKey"),
        }


def filter_recent_news(news_items: Iterable[Dict[str, Any]], days: int = 7) -> List[Dict[str, Any]]:
    """Keep only recent news items."""

    floor = datetime.now(timezone.utc) - timedelta(days=days)
    filtered = []
    for item in news_items:
        provider_publish_time = item.get("providerPublishTime")
        if provider_publish_time is None:
            filtered.append(item)
            continue
        published = datetime.fromtimestamp(provider_publish_time, tz=timezone.utc)
        if published >= floor:
            filtered.append(item)
    return filtered


def provider_health_check(config: AppConfig) -> str:
    cache = SQLiteCache(config.cache_db)
    fetcher = MarketDataFetcher(config, cache)
    lines = []
    start = time.perf_counter()
    spy = fetcher.download_history("SPY", interval="1d", period="1mo", ttl_seconds=60, fresh=True)
    yahoo_latency = time.perf_counter() - start
    lines.append(f"Yahoo Finance    {'OK' if not spy.empty else 'FAIL'}   (avg {yahoo_latency:.1f}s/ticker)")
    if config.alpha_vantage_api_key:
        start = time.perf_counter()
        av = fetcher._download_history_alpha_vantage("SPY", interval="1d")
        lines.append(f"Alpha Vantage    {'OK' if not av.empty else 'FAIL'}   (avg {time.perf_counter() - start:.1f}s/ticker)")
    else:
        lines.append("Alpha Vantage    NOT CONFIGURED")
    start = time.perf_counter()
    sec_map = fetcher.fetch_sec_ticker_map(fresh=False)
    lines.append(f"Congress / SEC   {'OK' if bool(sec_map) else 'FAIL'}   (avg {time.perf_counter() - start:.1f}s/query)")
    lines.append(f"Reddit API       {'OK' if config.reddit_client_id and config.reddit_client_secret else 'FAIL'}")
    lines.append(f"X endpoint       {'OK' if config.x_search_endpoint else 'FAIL'}")
    sector_history = fetcher.fetch_sector_history(fresh=False)
    lines.append(f"Sector ETFs      {'OK' if any(not frame.empty for frame in sector_history.values()) else 'FAIL'}")
    lines.append(f"ML Model         {'TRAINED' if config.xgb_model_path.exists() else 'UNTRAINED'}")
    try:
        sp500_count = sum(len(tickers) for tickers in fetcher.resolve_universe("sp500").values())
        nasdaq100_count = sum(len(tickers) for tickers in fetcher.resolve_universe("nasdaq100").values())
        full_count = len(
            {
                ticker
                for sector_tickers in fetcher.resolve_universe("full").values()
                for ticker in sector_tickers
            }
        )
        lines.append(f"Universe lists    S&P 500={sp500_count} Nasdaq-100={nasdaq100_count} Full={full_count}")
    except Exception:
        lines.append("Universe lists    FAIL")
    stats = fetcher.cache_stats()
    lines.append(
        f"Cache stats      {stats['history_rows']} history entries, {stats['universe_rows']} universe lists, {stats['cache_rows']} total rows"
    )
    return "\n".join(lines)
