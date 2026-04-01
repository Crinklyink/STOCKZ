"""Technical indicator calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the core indicator set to OHLCV data."""

    if frame.empty:
        return frame.copy()
    data = frame.copy()
    close = data["close"]
    high = data["high"]
    low = data["low"]
    volume = data["volume"]

    data["return_1"] = close.pct_change(1)
    data["return_5"] = close.pct_change(5)
    data["return_20"] = close.pct_change(20)
    data["volume_delta"] = volume.pct_change().replace([np.inf, -np.inf], 0.0)
    data["sma_10"] = close.rolling(10).mean()
    data["sma_20"] = close.rolling(20).mean()
    data["sma_50"] = close.rolling(50).mean()
    data["sma_200"] = close.rolling(200).mean()
    data["ema_12"] = close.ewm(span=12, adjust=False).mean()
    data["ema_26"] = close.ewm(span=26, adjust=False).mean()
    data["macd"] = data["ema_12"] - data["ema_26"]
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_hist"] = data["macd"] - data["macd_signal"]
    data["rsi"] = rsi(close)
    data["atr"] = atr(high, low, close)
    data["obv"] = obv(close, volume)
    data["mfi"] = mfi(high, low, close, volume)
    data["vwap"] = vwap(high, low, close, volume)
    data["vwap_distance"] = (close - data["vwap"]) / data["vwap"].replace(0, np.nan)
    data["bollinger_mid"] = close.rolling(20).mean()
    std = close.rolling(20).std()
    data["bollinger_high"] = data["bollinger_mid"] + 2 * std
    data["bollinger_low"] = data["bollinger_mid"] - 2 * std
    data["adx"] = adx(high, low, close)
    data["roc_10"] = close.pct_change(10)
    data["stoch_k"] = stochastic_k(high, low, close)
    data["stoch_d"] = data["stoch_k"].rolling(3).mean()
    data["inside_bar"] = ((high < high.shift(1)) & (low > low.shift(1))).astype(int)
    data["golden_cross"] = (
        (data["sma_50"] > data["sma_200"]) & (data["sma_50"].shift(1) <= data["sma_200"].shift(1))
    ).astype(int)
    return data.replace([np.inf, -np.inf], np.nan)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    true_range = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period).mean()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume).fillna(0.0).cumsum()


def mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    direction = typical_price.diff().fillna(0.0)
    positive = money_flow.where(direction > 0, 0.0).rolling(period).sum()
    negative = money_flow.where(direction < 0, 0.0).rolling(period).sum().abs()
    ratio = positive / negative.replace(0, np.nan)
    return 100 - (100 / (1 + ratio))


def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    typical_price = (high + low + close) / 3
    return (typical_price * volume).cumsum() / volume.replace(0, np.nan).cumsum()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_series = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).sum() / atr_series.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period).sum() / atr_series.replace(0, np.nan))
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.rolling(period).mean()


def stochastic_k(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    lowest_low = low.rolling(period).min()
    highest_high = high.rolling(period).max()
    return 100 * ((close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan))

