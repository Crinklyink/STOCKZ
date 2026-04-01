"""Shared utilities."""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def setup_logging(log_path: Path) -> None:
    """Configure file and stdout logging."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )


def utc_now() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc)


def json_default(value: Any) -> Any:
    """Default serializer for JSON exports."""

    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def save_json(path: Path, payload: Any) -> None:
    """Write JSON to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=json_default),
        encoding="utf-8",
    )


def coerce_float(value: Any, default: float = 0.0) -> float:
    """Best-effort float conversion."""

    try:
        if value is None:
            return default
        coerced = float(value)
        if not math.isfinite(coerced):
            return default
        return coerced
    except (TypeError, ValueError):
        return default


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a float."""

    return max(lower, min(upper, value))


def normalize(value: float, min_val: float, max_val: float) -> float:
    """Clamp and scale a raw value to a 0-100 score."""

    if max_val == min_val:
        return 50.0
    clipped = clamp(value, min_val, max_val)
    return 100.0 * (clipped - min_val) / (max_val - min_val)
