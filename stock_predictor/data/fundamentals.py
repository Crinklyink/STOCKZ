"""Fundamental and macro provider clients used by the research pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

from stock_predictor.config import AppConfig
from stock_predictor.data.cache import SQLiteCache
from stock_predictor.utils import coerce_float

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FundamentalSnapshot:
    ticker: str
    revenue_growth: float = 0.0
    revenue_growth_ttm: float = 0.0
    gross_margin: float = 0.0
    operating_margin: float = 0.0
    free_cash_flow_margin: float = 0.0
    debt_to_assets: float = 0.0
    share_change: float = 0.0
    margin_trend_ttm: float = 0.0
    debt_change_ttm: float = 0.0
    share_dilution_ttm: float = 0.0
    source: str = "unavailable"

    def to_features(self) -> dict[str, float]:
        return {
            "fund_revenue_growth": self.revenue_growth,
            "fund_revenue_growth_ttm": self.revenue_growth_ttm,
            "fund_gross_margin": self.gross_margin,
            "fund_operating_margin": self.operating_margin,
            "fund_free_cash_flow_margin": self.free_cash_flow_margin,
            "fund_debt_to_assets": self.debt_to_assets,
            "fund_share_change": self.share_change,
            "fund_margin_trend_ttm": self.margin_trend_ttm,
            "fund_debt_change_ttm": self.debt_change_ttm,
            "fund_share_dilution_ttm": self.share_dilution_ttm,
        }


class SECCompanyFactsClient:
    """Thin client for SEC companyfacts data with cache and polite headers."""

    def __init__(self, config: AppConfig, cache: SQLiteCache) -> None:
        self.config = config
        self.cache = cache

    def fetch_company_facts(self, cik: str, *, fresh: bool = False) -> dict[str, Any]:
        normalized = str(cik).strip().zfill(10)
        cache_key = f"sec-companyfacts:{normalized}"
        if not fresh:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{normalized}.json"
        headers = {"User-Agent": "stock-predictor/1.0 contact@example.com"}
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        payload = response.json()
        self.cache.set(cache_key, payload, ttl_seconds=self.config.cache_ttls.sec_filings)
        return payload

    def snapshot_from_facts(self, ticker: str, facts: dict[str, Any]) -> FundamentalSnapshot:
        timeline = self.build_feature_timeline(ticker, facts)
        if timeline.empty:
            return FundamentalSnapshot(ticker=ticker, source="sec_companyfacts")
        latest = timeline.iloc[-1]
        return FundamentalSnapshot(
            ticker=ticker,
            revenue_growth=coerce_float(latest.get("fund_revenue_growth"), 0.0),
            revenue_growth_ttm=coerce_float(latest.get("fund_revenue_growth_ttm"), 0.0),
            gross_margin=coerce_float(latest.get("fund_gross_margin"), 0.0),
            operating_margin=coerce_float(latest.get("fund_operating_margin"), 0.0),
            free_cash_flow_margin=coerce_float(latest.get("fund_free_cash_flow_margin"), 0.0),
            debt_to_assets=coerce_float(latest.get("fund_debt_to_assets"), 0.0),
            share_change=coerce_float(latest.get("fund_share_change"), 0.0),
            margin_trend_ttm=coerce_float(latest.get("fund_margin_trend_ttm"), 0.0),
            debt_change_ttm=coerce_float(latest.get("fund_debt_change_ttm"), 0.0),
            share_dilution_ttm=coerce_float(latest.get("fund_share_dilution_ttm"), 0.0),
            source="sec_companyfacts",
        )

    def build_feature_timeline(self, ticker: str, facts: dict[str, Any]) -> pd.DataFrame:
        """Return quarterly/TTM fundamental feature history indexed by period end date."""

        revenue = self._series_from_tags(
            facts,
            ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
        )
        gross_profit = self._series_from_tags(facts, ["GrossProfit"])
        operating_income = self._series_from_tags(facts, ["OperatingIncomeLoss"])
        cfo = self._series_from_tags(facts, ["NetCashProvidedByUsedInOperatingActivities"])
        capex = self._series_from_tags(facts, ["PaymentsToAcquirePropertyPlantAndEquipment"])
        assets = self._series_from_tags(facts, ["Assets"])
        debt = self._series_from_tags(facts, ["LongTermDebt", "ShortTermBorrowings"])
        shares = self._series_from_tags(
            facts,
            ["EntityCommonStockSharesOutstanding", "WeightedAverageNumberOfDilutedSharesOutstanding"],
        )
        index = sorted(
            set(revenue.index)
            | set(gross_profit.index)
            | set(operating_income.index)
            | set(cfo.index)
            | set(capex.index)
            | set(assets.index)
            | set(debt.index)
            | set(shares.index)
        )
        if not index:
            return pd.DataFrame(
                columns=[
                    "fund_revenue_growth",
                    "fund_revenue_growth_ttm",
                    "fund_gross_margin",
                    "fund_operating_margin",
                    "fund_free_cash_flow_margin",
                    "fund_debt_to_assets",
                    "fund_share_change",
                    "fund_margin_trend_ttm",
                    "fund_debt_change_ttm",
                    "fund_share_dilution_ttm",
                ]
            )
        frame = pd.DataFrame(index=pd.DatetimeIndex(index))
        frame["revenue"] = revenue.reindex(frame.index).ffill()
        frame["gross_profit"] = gross_profit.reindex(frame.index).ffill()
        frame["operating_income"] = operating_income.reindex(frame.index).ffill()
        frame["cfo"] = cfo.reindex(frame.index).ffill()
        frame["capex"] = capex.reindex(frame.index).ffill()
        frame["assets"] = assets.reindex(frame.index).ffill()
        frame["debt"] = debt.reindex(frame.index).ffill()
        frame["shares"] = shares.reindex(frame.index).ffill()
        frame["fcf"] = frame["cfo"] - frame["capex"].fillna(0.0)
        frame["ttm_revenue"] = frame["revenue"].rolling(4, min_periods=2).sum()
        gross_margin = frame["gross_profit"] / frame["revenue"].replace(0, pd.NA)
        operating_margin = frame["operating_income"] / frame["revenue"].replace(0, pd.NA)
        free_cash_flow_margin = frame["fcf"] / frame["revenue"].replace(0, pd.NA)
        debt_to_assets = frame["debt"] / frame["assets"].replace(0, pd.NA)
        output = pd.DataFrame(index=frame.index)
        output["fund_revenue_growth"] = frame["revenue"].pct_change(4, fill_method=None)
        output["fund_revenue_growth_ttm"] = frame["ttm_revenue"].pct_change(4, fill_method=None)
        output["fund_gross_margin"] = gross_margin
        output["fund_operating_margin"] = operating_margin
        output["fund_free_cash_flow_margin"] = free_cash_flow_margin
        output["fund_debt_to_assets"] = debt_to_assets
        output["fund_share_change"] = frame["shares"].pct_change(4, fill_method=None)
        output["fund_margin_trend_ttm"] = gross_margin.rolling(4, min_periods=2).mean().diff(4)
        output["fund_debt_change_ttm"] = debt_to_assets.diff(4)
        output["fund_share_dilution_ttm"] = frame["shares"].pct_change(4, fill_method=None)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            try:
                pd.set_option('future.no_silent_downcasting', True)
            except Exception:
                pass
            return output.replace([pd.NA, float("inf"), float("-inf")], 0.0).fillna(0.0).sort_index()

    def _series_from_tags(self, facts: dict[str, Any], tags: list[str]) -> pd.Series:
        us_gaap = facts.get("facts", {}).get("us-gaap", {}) if isinstance(facts, dict) else {}
        rows: list[tuple[pd.Timestamp, float]] = []
        for tag in tags:
            units = us_gaap.get(tag, {}).get("units", {})
            for values in units.values():
                if not isinstance(values, list):
                    continue
                for item in values:
                    if not isinstance(item, dict) or item.get("val") is None:
                        continue
                    end = pd.to_datetime(item.get("end"), utc=True, errors="coerce")
                    if pd.isna(end):
                        continue
                    rows.append((end, coerce_float(item.get("val"), 0.0)))
        if not rows:
            return pd.Series(dtype=float)
        frame = pd.DataFrame(rows, columns=["end", "value"]).sort_values("end")
        series = frame.groupby("end", as_index=True)["value"].last().sort_index()
        series.index = pd.to_datetime(series.index, utc=True, errors="coerce")
        return series.dropna()


class FREDMacroClient:
    """FRED series fetcher for macro/regime features."""

    def __init__(self, config: AppConfig, cache: SQLiteCache) -> None:
        self.config = config
        self.cache = cache

    def fetch_observations(self, series_id: str, *, fresh: bool = False) -> list[dict[str, Any]]:
        if not self.config.fred_api_key:
            return []
        cache_key = f"fred:{series_id}"
        if not fresh:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        response = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": series_id,
                "api_key": self.config.fred_api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 260,
            },
            timeout=20,
        )
        response.raise_for_status()
        observations = response.json().get("observations", [])
        self.cache.set(cache_key, observations, ttl_seconds=self.config.cache_ttls.macro)
        return observations
