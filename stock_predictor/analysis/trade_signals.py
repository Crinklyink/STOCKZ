"""Actionable breakout and volume signal helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stock_predictor.utils import clamp, coerce_float


@dataclass(slots=True)
class BreakoutResult:
    resistance_level: float
    broke_resistance: bool
    volume_confirmed: bool
    held_breakout: bool
    confirmed: bool
    warning: str
    bonus_points: float
    summary: str


@dataclass(slots=True)
class RelativeVolumeResult:
    ratio: float
    triggered: bool
    bonus_points: float
    summary: str


def confirm_breakout(daily_frame: pd.DataFrame, bonus_points: float) -> BreakoutResult:
    if daily_frame.empty or len(daily_frame) < 60:
        return BreakoutResult(0.0, False, False, False, False, "", 0.0, "Breakout data unavailable.")
    data = daily_frame.copy()
    latest = data.iloc[-1]
    prior = data.iloc[:-2] if len(data) > 2 else data.iloc[:-1]
    resistance_level = max(
        coerce_float(prior["close"].tail(20).max()),
        coerce_float(prior["high"].tail(252 if len(prior) >= 252 else len(prior)).max()),
    )
    broke_resistance = coerce_float(latest["close"]) > resistance_level * 1.002
    average_volume = coerce_float(data["volume"].tail(20).mean(), 1.0)
    volume_confirmed = coerce_float(latest["volume"]) / max(average_volume, 1.0) >= 1.5
    held_breakout = bool((data["close"].tail(2) > resistance_level).all())
    checks = sum([broke_resistance, volume_confirmed, held_breakout])
    confirmed = checks == 3
    warning = ""
    if checks == 2:
        warning = "UNCONFIRMED BREAKOUT"
    summary = "No valid breakout setup detected."
    if confirmed:
        summary = "CONFIRMED BREAKOUT above resistance with institutional volume and follow-through."
    elif warning:
        summary = "UNCONFIRMED BREAKOUT: two of three confirmation checks passed."
    return BreakoutResult(
        resistance_level=round(resistance_level, 2),
        broke_resistance=broke_resistance,
        volume_confirmed=volume_confirmed,
        held_breakout=held_breakout,
        confirmed=confirmed,
        warning=warning,
        bonus_points=bonus_points if confirmed else 0.0,
        summary=summary,
    )


def detect_relative_volume_alert(hourly_frame: pd.DataFrame, bonus_points: float) -> RelativeVolumeResult:
    if hourly_frame.empty or len(hourly_frame) < 80:
        return RelativeVolumeResult(0.0, False, 0.0, "Relative intraday volume unavailable.")
    frame = hourly_frame.copy()
    frame.index = pd.to_datetime(frame.index, utc=True)
    latest_timestamp = frame.index[-1]
    latest_day = latest_timestamp.normalize()
    current_day = frame.loc[frame.index.normalize() == latest_day]
    if current_day.empty:
        return RelativeVolumeResult(0.0, False, 0.0, "Relative intraday volume unavailable.")
    cutoff_time = latest_timestamp.time()
    current_cumulative = coerce_float(current_day.loc[current_day.index.time <= cutoff_time, "volume"].sum())
    history = []
    for day, group in frame.groupby(frame.index.normalize()):
        if day >= latest_day:
            continue
        same_slice = group.loc[group.index.time <= cutoff_time, "volume"]
        if same_slice.empty:
            continue
        history.append(coerce_float(same_slice.sum()))
    typical = sum(history[-10:]) / max(len(history[-10:]), 1)
    ratio = current_cumulative / max(typical, 1.0)
    triggered = ratio >= 2.0
    summary = "UNUSUAL EARLY VOLUME" if triggered else "Intraday volume is within its normal range."
    return RelativeVolumeResult(
        ratio=round(ratio, 2),
        triggered=triggered,
        bonus_points=bonus_points if triggered else 0.0,
        summary=summary,
    )
