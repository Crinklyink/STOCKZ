"""Lightweight model monitoring summaries for live scan artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def population_stability_index(expected: pd.Series, actual: pd.Series, *, buckets: int = 10) -> float:
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()
    if expected.empty or actual.empty:
        return 0.0
    quantiles = np.linspace(0.0, 1.0, buckets + 1)
    edges = np.unique(expected.quantile(quantiles).to_numpy())
    if len(edges) < 3:
        return 0.0
    expected_counts, _ = np.histogram(expected, bins=edges)
    actual_counts, _ = np.histogram(actual, bins=edges)
    expected_pct = np.clip(expected_counts / max(expected_counts.sum(), 1), 1e-4, None)
    actual_pct = np.clip(actual_counts / max(actual_counts.sum(), 1), 1e-4, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def sector_concentration(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    sectors = [str(item.get("sector") or "Unknown") for item in candidates]
    if not sectors:
        return {"top_sector": "n/a", "top_sector_share": 0.0, "warning": "No active candidates"}
    counts = pd.Series(sectors).value_counts()
    top_sector = str(counts.index[0])
    share = float(counts.iloc[0] / max(len(sectors), 1))
    return {
        "top_sector": top_sector,
        "top_sector_share": round(share * 100.0, 1),
        "warning": "Sector concentration elevated" if share >= 0.45 else "",
    }


def calibration_summary(paper_rows: pd.DataFrame) -> dict[str, Any]:
    if paper_rows.empty or "realized_return" not in paper_rows:
        return {"buckets": [], "warning": "No resolved paper trades yet"}
    frame = paper_rows.copy()
    frame["realized_return"] = pd.to_numeric(frame["realized_return"], errors="coerce")
    frame = frame.dropna(subset=["realized_return"])
    if frame.empty:
        return {"buckets": [], "warning": "No resolved paper trades yet"}
    confidence = pd.to_numeric(frame.get("final_score", pd.Series(50.0, index=frame.index)), errors="coerce").fillna(50.0)
    frame["confidence_bucket"] = pd.cut(confidence, bins=[0, 50, 60, 70, 100], include_lowest=True)
    grouped = frame.groupby("confidence_bucket", observed=False)
    buckets = []
    for bucket, group in grouped:
        if group.empty:
            continue
        buckets.append(
            {
                "bucket": str(bucket),
                "trades": int(len(group)),
                "hit_rate": round(float((group["realized_return"] > 0).mean()) * 100.0, 1),
                "avg_return": round(float(group["realized_return"].mean()) * 100.0, 2),
            }
        )
    return {"buckets": buckets, "warning": ""}


def regime_hit_rates(paper_rows: pd.DataFrame) -> list[dict[str, Any]]:
    if paper_rows.empty or "regime" not in paper_rows or "realized_return" not in paper_rows:
        return []
    frame = paper_rows.copy()
    frame["realized_return"] = pd.to_numeric(frame["realized_return"], errors="coerce")
    frame = frame.dropna(subset=["realized_return"])
    rows = []
    for regime, group in frame.groupby("regime"):
        rows.append(
            {
                "regime": str(regime),
                "trades": int(len(group)),
                "hit_rate": round(float((group["realized_return"] > 0).mean()) * 100.0, 1),
                "avg_return": round(float(group["realized_return"].mean()) * 100.0, 2),
            }
        )
    return rows


def build_model_monitoring_payload(
    *,
    candidates: list[dict[str, Any]],
    paper_rows: pd.DataFrame,
    model_auc: float,
    generated_at: str,
) -> dict[str, Any]:
    concentration = sector_concentration(candidates)
    calibration = calibration_summary(paper_rows)
    regimes = regime_hit_rates(paper_rows)
    score_series = pd.Series([item.get("final_score") for item in candidates], dtype="float64")
    score_health = "healthy"
    if model_auc and model_auc < 0.55:
        score_health = "weak validation"
    if concentration.get("warning"):
        score_health = "watch concentration"
    return {
        "generated_at": generated_at,
        "health": score_health,
        "model_auc": round(float(model_auc or 0.0), 3),
        "candidate_count": int(len(candidates)),
        "sector_concentration": concentration,
        "score_distribution": {
            "mean": round(float(score_series.mean()), 2) if not score_series.dropna().empty else 0.0,
            "std": round(float(score_series.std()), 2) if len(score_series.dropna()) > 1 else 0.0,
        },
        "calibration": calibration,
        "hit_rate_by_regime": regimes,
        "warnings": [
            item
            for item in [
                concentration.get("warning"),
                calibration.get("warning"),
                "Model AUC below 0.55" if model_auc and model_auc < 0.55 else "",
            ]
            if item
        ],
    }
