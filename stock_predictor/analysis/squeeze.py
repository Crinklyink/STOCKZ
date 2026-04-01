"""Short squeeze probability scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

from stock_predictor.utils import clamp, coerce_float


@dataclass(slots=True)
class SqueezeResult:
    score: float
    qualifying: bool
    summary: str


def calculate_squeeze_probability(info: Dict[str, Any], daily_frame: pd.DataFrame) -> SqueezeResult:
    """Estimate short-squeeze probability from short interest and price action."""

    if daily_frame.empty:
        return SqueezeResult(score=0.0, qualifying=False, summary="No price data")
    close = daily_frame["close"]
    volume = daily_frame["volume"]
    short_pct = coerce_float(info.get("shortPercentOfFloat") or info.get("sharesPercentSharesOut")) * 100
    short_ratio = coerce_float(info.get("shortRatio"))
    borrow_rate = coerce_float(info.get("borrowRate") or info.get("feeRate")) * 100
    recent_return = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 else 0.0
    volume_surge = volume.tail(5).mean() / max(volume.tail(20).mean(), 1) if len(volume) >= 20 else 1.0
    score = (
        0.35 * clamp(short_pct / 30.0, 0.0, 1.0)
        + 0.20 * clamp(short_ratio / 7.0, 0.0, 1.0)
        + 0.10 * clamp(borrow_rate / 15.0, 0.0, 1.0)
        + 0.20 * clamp((recent_return + 5.0) / 15.0, 0.0, 1.0)
        + 0.15 * clamp((volume_surge - 1.0) / 1.5, 0.0, 1.0)
    ) * 100
    qualifying = bool(score > 70 and recent_return > 0 and volume_surge > 1.1)
    summary = (
        f"Short interest {short_pct:.1f}%, days to cover {short_ratio:.2f}, "
        f"borrow {borrow_rate:.1f}%, volume surge {volume_surge:.2f}x"
    )
    return SqueezeResult(score=round(score, 2), qualifying=qualifying, summary=summary)

