"""Leakage-safe validation helpers for financial labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd
from pandas.tseries.offsets import BDay


@dataclass(frozen=True, slots=True)
class PurgedSplit:
    """Train/test partition plus diagnostic counts."""

    fold: int
    train_index: pd.Index
    test_index: pd.Index
    train_rows: int
    test_rows: int
    purged_rows: int


def purge_train_frame(
    frame: pd.DataFrame,
    *,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp | None = None,
    date_column: str = "date",
    label_end_column: str = "label_end_date",
    embargo_days: int = 5,
) -> pd.DataFrame:
    """Remove rows whose future label window can overlap the test window.

    A row is eligible for training only when its label formation finished before
    the embargo boundary. This is stricter than a plain date split and avoids
    training on information whose outcome overlaps the test period.
    """

    if frame.empty or label_end_column not in frame:
        return frame.copy()
    start = pd.Timestamp(test_start)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    embargo_boundary = start - BDay(max(int(embargo_days), 0))
    label_end = pd.to_datetime(frame[label_end_column], utc=True, errors="coerce")
    date_values = pd.to_datetime(frame[date_column], utc=True, errors="coerce") if date_column in frame else label_end
    mask = label_end < embargo_boundary
    if test_end is not None:
        end = pd.Timestamp(test_end)
        if end.tzinfo is None:
            end = end.tz_localize("UTC")
        else:
            end = end.tz_convert("UTC")
        mask &= ~((date_values >= start) & (date_values <= end))
    return frame.loc[mask].copy()


def monthly_purged_splits(
    frame: pd.DataFrame,
    *,
    min_train_rows: int = 200,
    warmup_months: int = 12,
    embargo_days: int = 5,
) -> Iterator[PurgedSplit]:
    """Yield rolling monthly test folds with purged/embargoed training rows."""

    if frame.empty or "date" not in frame:
        return
    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"], utc=True, errors="coerce")
    work = work.dropna(subset=["date"]).sort_values("date")
    if work.empty:
        return
    work["month"] = work["date"].dt.tz_localize(None).dt.to_period("M").astype(str)
    months = sorted(work["month"].dropna().unique())
    for fold, month in enumerate(months[warmup_months:], start=1):
        test = work.loc[work["month"] == month]
        if test.empty:
            continue
        test_start = pd.to_datetime(test["date"].min(), utc=True)
        test_end = pd.to_datetime(test["date"].max(), utc=True)
        naive_train = work.loc[work["date"] < test_start]
        train = purge_train_frame(
            work,
            test_start=test_start,
            test_end=test_end,
            embargo_days=embargo_days,
        )
        if len(train) < min_train_rows:
            continue
        yield PurgedSplit(
            fold=fold,
            train_index=train.index,
            test_index=test.index,
            train_rows=int(len(train)),
            test_rows=int(len(test)),
            purged_rows=max(0, int(len(naive_train) - len(train))),
        )
