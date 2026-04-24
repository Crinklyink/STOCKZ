"""Market regime detection and historical regime labeling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np
import pandas as pd

from stock_predictor.config import AppConfig, get_config
from stock_predictor.utils import clamp


REGIME_LABELS = (
    "bull_quiet",
    "bull_volatile",
    "neutral",
    "bear_volatile",
    "crisis",
)


@dataclass(slots=True)
class RegimeSnapshot:
    label: str
    vix: float
    vix_trend: float
    spy_20d_momentum: float
    breadth: float
    sector_dispersion: float
    put_call_trend: float
    credit_spread_proxy: float
    momentum_regime_score: float
    rotation_speed: float
    earnings_quality: float
    liquidity_stress: float
    intermarket_divergence: float

    def to_dict(self) -> Dict[str, float | str]:
        return asdict(self)


class RegimeClassifier:
    """Classify the current market regime and build historical regime labels."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()

    def classify(
        self,
        *,
        vix: float,
        vix_trend: float,
        spy_20d_momentum: float,
        breadth: float,
        sector_dispersion: float,
        put_call_trend: float,
        credit_spread_proxy: float,
    ) -> str:
        if vix > 35.0:
            return "crisis"
        if vix < 18.0 and spy_20d_momentum > 0.02 and breadth > 0.60:
            return "bull_quiet"
        if 18.0 <= vix < 25.0 and spy_20d_momentum > 0.01:
            return "bull_volatile"
        if vix >= 25.0 and (spy_20d_momentum < -0.01 or breadth < 0.40):
            return "bear_volatile"
        if put_call_trend > 0.10 and credit_spread_proxy < -0.01 and vix_trend > 0:
            return "bear_volatile"
        if sector_dispersion > 0.07 and abs(spy_20d_momentum) < 0.02:
            return "neutral"
        return "neutral"

    def build_snapshot(
        self,
        *,
        price_frames: Dict[str, pd.DataFrame],
        benchmark_history: pd.DataFrame,
        breadth_history: pd.Series | pd.DataFrame | None = None,
        sector_histories: Dict[str, pd.DataFrame] | None = None,
        vix_history: pd.DataFrame | None = None,
        put_call_history: pd.DataFrame | None = None,
        hyg_history: pd.DataFrame | None = None,
        lqd_history: pd.DataFrame | None = None,
        tlt_history: pd.DataFrame | None = None,
    ) -> RegimeSnapshot:
        history = self.build_regime_history(
            price_frames=price_frames,
            benchmark_history=benchmark_history,
            breadth_history=breadth_history,
            sector_histories=sector_histories,
            vix_history=vix_history,
            put_call_history=put_call_history,
            hyg_history=hyg_history,
            lqd_history=lqd_history,
            tlt_history=tlt_history,
        )
        if history.empty:
            return RegimeSnapshot(
                label="neutral",
                vix=20.0,
                vix_trend=0.0,
                spy_20d_momentum=0.0,
                breadth=0.5,
                sector_dispersion=0.04,
                put_call_trend=0.0,
                credit_spread_proxy=0.0,
                momentum_regime_score=0.5,
                rotation_speed=0.04,
                earnings_quality=0.5,
                liquidity_stress=0.02,
                intermarket_divergence=0.02,
            )
        latest = history.iloc[-1]
        return RegimeSnapshot(
            label=str(latest["regime"]),
            vix=float(latest["vix"]),
            vix_trend=float(latest["vix_trend"]),
            spy_20d_momentum=float(latest["spy_20d_momentum"]),
            breadth=float(latest["breadth"]),
            sector_dispersion=float(latest["sector_dispersion"]),
            put_call_trend=float(latest["put_call_trend"]),
            credit_spread_proxy=float(latest["credit_spread_proxy"]),
            momentum_regime_score=float(latest["momentum_regime_score"]),
            rotation_speed=float(latest["rotation_speed"]),
            earnings_quality=float(latest["earnings_quality"]),
            liquidity_stress=float(latest["liquidity_stress"]),
            intermarket_divergence=float(latest["intermarket_divergence"]),
        )

    def build_regime_history(
        self,
        *,
        price_frames: Dict[str, pd.DataFrame],
        benchmark_history: pd.DataFrame,
        breadth_history: pd.Series | pd.DataFrame | None = None,
        sector_histories: Dict[str, pd.DataFrame] | None = None,
        vix_history: pd.DataFrame | None = None,
        put_call_history: pd.DataFrame | None = None,
        hyg_history: pd.DataFrame | None = None,
        lqd_history: pd.DataFrame | None = None,
        tlt_history: pd.DataFrame | None = None,
        earnings_quality_series: pd.Series | None = None,
    ) -> pd.DataFrame:
        if benchmark_history is None or benchmark_history.empty or "close" not in benchmark_history:
            return pd.DataFrame()
        benchmark_close = benchmark_history["close"].sort_index().ffill()
        index = benchmark_close.index
        breadth_series = self._normalize_series(breadth_history, index)
        if breadth_series.empty:
            breadth_series = self._compute_breadth_history(price_frames, index)
        sector_dispersion = self._compute_sector_dispersion(sector_histories or {}, index)
        put_call_series = self._close_series(put_call_history, index, default=1.0)
        vix_series = self._close_series(vix_history, index, default=20.0)
        hyg_series = self._close_series(hyg_history, index, default=80.0)
        lqd_series = self._close_series(lqd_history, index, default=100.0)
        tlt_series = self._close_series(tlt_history, index, default=100.0)
        momentum_regime_score = self._compute_momentum_regime(price_frames, index)
        liquidity_stress = self._compute_liquidity_stress(price_frames, index)
        earnings_quality = (
            earnings_quality_series.reindex(index, method="ffill").fillna(0.5)
            if earnings_quality_series is not None and not earnings_quality_series.empty
            else pd.Series(0.5, index=index)
        )

        spy_20d_momentum = benchmark_close.pct_change(20).fillna(0.0)
        vix_trend = vix_series.pct_change(10).fillna(0.0)
        put_call_trend = put_call_series.pct_change(10).fillna(0.0)
        credit_spread_proxy = (hyg_series / lqd_series.replace(0, np.nan)).pct_change(10).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        rotation_speed = sector_dispersion.rolling(5, min_periods=1).mean()
        intermarket_divergence = (benchmark_close.pct_change(5) - (-tlt_series.pct_change(5))).abs().fillna(0.0)

        frame = pd.DataFrame(
            {
                "vix": vix_series,
                "vix_trend": vix_trend,
                "spy_20d_momentum": spy_20d_momentum,
                "breadth": breadth_series.clip(0.0, 1.0),
                "sector_dispersion": sector_dispersion.fillna(0.04),
                "put_call_trend": put_call_trend,
                "credit_spread_proxy": credit_spread_proxy,
                "momentum_regime_score": momentum_regime_score.clip(0.0, 1.0),
                "rotation_speed": rotation_speed.fillna(0.04),
                "earnings_quality": earnings_quality.clip(0.0, 1.0),
                "liquidity_stress": liquidity_stress.fillna(0.02),
                "intermarket_divergence": intermarket_divergence.fillna(0.02),
            },
            index=index,
        ).replace([np.inf, -np.inf], np.nan).ffill().fillna(
            {
                "vix": 20.0,
                "vix_trend": 0.0,
                "spy_20d_momentum": 0.0,
                "breadth": 0.5,
                "sector_dispersion": 0.04,
                "put_call_trend": 0.0,
                "credit_spread_proxy": 0.0,
                "momentum_regime_score": 0.5,
                "rotation_speed": 0.04,
                "earnings_quality": 0.5,
                "liquidity_stress": 0.02,
                "intermarket_divergence": 0.02,
            }
        )
        frame["regime"] = [
            self.classify(
                vix=float(row.vix),
                vix_trend=float(row.vix_trend),
                spy_20d_momentum=float(row.spy_20d_momentum),
                breadth=float(row.breadth),
                sector_dispersion=float(row.sector_dispersion),
                put_call_trend=float(row.put_call_trend),
                credit_spread_proxy=float(row.credit_spread_proxy),
            )
            for row in frame.itertuples()
        ]
        return frame

    def _close_series(
        self,
        frame: pd.DataFrame | None,
        index: pd.Index,
        *,
        default: float,
    ) -> pd.Series:
        if frame is None or frame.empty or "close" not in frame:
            return pd.Series(default, index=index, dtype=float)
        return frame["close"].sort_index().reindex(index, method="ffill").fillna(default)

    def _normalize_series(self, series: pd.Series | pd.DataFrame | None, index: pd.Index) -> pd.Series:
        if series is None:
            return pd.Series(dtype=float)
        if isinstance(series, pd.DataFrame):
            if series.empty:
                return pd.Series(dtype=float)
            if "close" in series:
                base = series["close"]
            else:
                base = series.iloc[:, 0]
        else:
            base = series
        return pd.to_numeric(base, errors="coerce").sort_index().reindex(index, method="ffill").clip(0.0, 1.0)

    def _compute_breadth_history(self, price_frames: Dict[str, pd.DataFrame], index: pd.Index) -> pd.Series:
        breadth = {}
        usable = {
            ticker: frame[["close"]].sort_index()
            for ticker, frame in price_frames.items()
            if not frame.empty and "close" in frame
        }
        for date in index:
            above = 0
            total = 0
            for frame in usable.values():
                hist = frame.loc[:date]
                if len(hist) < 50:
                    continue
                sma50 = hist["close"].rolling(50).mean().iloc[-1]
                current = hist["close"].iloc[-1]
                if pd.isna(sma50) or pd.isna(current):
                    continue
                above += int(current > sma50)
                total += 1
            breadth[date] = above / total if total > 0 else 0.5
        return pd.Series(breadth, dtype=float)

    def _compute_sector_dispersion(self, sector_histories: Dict[str, pd.DataFrame], index: pd.Index) -> pd.Series:
        close_frame = pd.DataFrame(
            {
                sector: frame["close"].sort_index()
                for sector, frame in sector_histories.items()
                if not frame.empty and "close" in frame
            }
        )
        if close_frame.empty:
            return pd.Series(0.04, index=index, dtype=float)
        returns = close_frame.reindex(index, method="ffill").pct_change(7)
        return returns.std(axis=1).fillna(0.04)

    def _compute_momentum_regime(self, price_frames: Dict[str, pd.DataFrame], index: pd.Index) -> pd.Series:
        frames = {
            ticker: frame["close"].sort_index()
            for ticker, frame in price_frames.items()
            if not frame.empty and "close" in frame
        }
        if not frames:
            return pd.Series(0.5, index=index, dtype=float)
        momentum = {}
        for date in index:
            active = 0
            total = 0
            for close in frames.values():
                hist = close.loc[:date]
                if len(hist) < 6:
                    continue
                ret = hist.iloc[-1] / hist.iloc[-6] - 1.0
                active += int(ret > 0.05)
                total += 1
            momentum[date] = active / total if total else 0.5
        return pd.Series(momentum, dtype=float)

    def _compute_liquidity_stress(self, price_frames: Dict[str, pd.DataFrame], index: pd.Index) -> pd.Series:
        components = []
        for frame in price_frames.values():
            if frame.empty or not {"high", "low", "close"}.issubset(frame.columns):
                continue
            proxy = ((frame["high"] - frame["low"]) / frame["close"].replace(0, np.nan)).sort_index()
            components.append(proxy.reindex(index, method="ffill"))
        if not components:
            return pd.Series(0.02, index=index, dtype=float)
        joined = pd.concat(components, axis=1)
        return joined.mean(axis=1).rolling(10, min_periods=1).mean().fillna(0.02)


def position_size_from_regime(
    *,
    regime: str,
    model_confidence: float,
    uncertainty: float,
    regime_historical_win_rate: float,
    vix: float,
) -> float:
    """Dynamic position sizing for the adaptive model output."""

    if regime_historical_win_rate > 0.65:
        base = 0.04
    elif regime_historical_win_rate > 0.55:
        base = 0.025
    else:
        base = 0.01
    confidence_mult = model_confidence / 0.70 if model_confidence > 0 else 0.5
    uncertainty_mult = max(0.5, 1 - (uncertainty * 3))
    vix_mult = max(0.5, 1 - ((vix - 15) / 40))
    final_size = base * confidence_mult * uncertainty_mult * vix_mult
    return float(clamp(final_size, 0.005, 0.05))
