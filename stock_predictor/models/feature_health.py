"""Rolling feature-health tracking for adaptive weighting."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from stock_predictor.config import AppConfig, get_config


@dataclass(slots=True)
class FeatureHealthRecord:
    feature: str
    hit_rate: float
    sample_count: int
    weeks: int
    status: str
    weight_multiplier: float
    disabled: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class FeatureHealthTracker:
    """Track rolling hit rates for scored features and broad signals."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self.path = self.config.feature_health_path
        self.records: Dict[str, FeatureHealthRecord] = {}
        self._load()

    def refresh(self) -> Dict[str, FeatureHealthRecord]:
        rows = self._load_recent_rows(weeks=8)
        feature_rows: dict[str, list[tuple[bool, str]]] = {}
        for row in rows:
            hit = bool(row.get("resolved_target_hit"))
            week = str(row.get("week") or "")
            for feature in self._extract_fired_features(row):
                feature_rows.setdefault(feature, []).append((hit, week))
        refreshed: Dict[str, FeatureHealthRecord] = {}
        for feature, outcomes in feature_rows.items():
            if not outcomes:
                continue
            hits = [int(result) for result, _ in outcomes]
            weeks = len({week for _, week in outcomes if week})
            hit_rate = sum(hits) / len(hits)
            recent_outcomes = outcomes[-4:]
            recent_hit_rate = sum(int(result) for result, _ in recent_outcomes) / max(len(recent_outcomes), 1)
            disabled = len(recent_outcomes) >= 4 and recent_hit_rate < 0.35
            if disabled:
                status = "DISABLED"
                multiplier = 0.0
            elif hit_rate < 0.45:
                status = "WEAKENING"
                multiplier = 0.5
            elif hit_rate > 0.65:
                status = "ACTIVE"
                multiplier = 1.25
            else:
                status = "ACTIVE"
                multiplier = 1.0
            refreshed[feature] = FeatureHealthRecord(
                feature=feature,
                hit_rate=hit_rate,
                sample_count=len(outcomes),
                weeks=weeks,
                status=status,
                weight_multiplier=multiplier,
                disabled=disabled,
            )
        self.records = dict(sorted(refreshed.items(), key=lambda item: item[1].hit_rate, reverse=True))
        self.save()
        return self.records

    def broad_weight_adjustments(self) -> Dict[str, float]:
        mapping = {
            "ml": ["ml"],
            "technical": ["rsi", "macd", "volume", "price_vs_50ma", "bollinger", "adx", "momentum", "vwap"],
            "volume": ["volume"],
            "pattern": ["pattern"],
            "sentiment": ["sentiment"],
            "options": ["options"],
            "rs": ["rs"],
        }
        adjustments: Dict[str, float] = {}
        for broad, names in mapping.items():
            multipliers = [self.records[name].weight_multiplier for name in names if name in self.records]
            adjustments[broad] = float(sum(multipliers) / len(multipliers)) if multipliers else 1.0
        return adjustments

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": pd.Timestamp.utcnow().isoformat(),
            "records": {name: record.to_dict() for name, record in self.records.items()},
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            raw_records = payload.get("records", {})
            self.records = {
                str(name): FeatureHealthRecord(
                    feature=str(name),
                    hit_rate=float(record.get("hit_rate", 0.0)),
                    sample_count=int(record.get("sample_count", 0)),
                    weeks=int(record.get("weeks", 0)),
                    status=str(record.get("status", "ACTIVE")),
                    weight_multiplier=float(record.get("weight_multiplier", 1.0)),
                    disabled=bool(record.get("disabled", False)),
                )
                for name, record in raw_records.items()
            }
        except Exception:
            self.records = {}

    def _load_recent_rows(self, *, weeks: int) -> list[dict[str, object]]:
        db_path = self.config.paper_trade_db
        if not db_path.exists():
            return []
        query = """
            SELECT p.created_at, p.payload_json, e.realized_return, e.resolved_target_hit, e.hit_target
            FROM paper_predictions p
            JOIN paper_evaluations e ON p.run_id = e.run_id AND p.ticker = e.ticker
            WHERE e.realized_return IS NOT NULL
            ORDER BY p.created_at ASC
        """
        with sqlite3.connect(db_path) as conn:
            frame = pd.read_sql_query(query, conn)
        if frame.empty:
            return []
        frame["created_ts"] = pd.to_datetime(frame["created_at"], utc=True, errors="coerce")
        frame["week"] = frame["created_ts"].dt.tz_localize(None).dt.to_period("W").astype(str)
        unique_weeks = frame["week"].dropna().drop_duplicates().tolist()[-weeks:]
        frame = frame.loc[frame["week"].isin(unique_weeks)]
        rows: list[dict[str, object]] = []
        for row in frame.to_dict(orient="records"):
            try:
                payload = json.loads(row.get("payload_json") or "{}")
            except Exception:
                payload = {}
            rows.append(
                {
                    "week": row.get("week"),
                    "payload": payload,
                    "resolved_target_hit": row.get("resolved_target_hit")
                    if pd.notna(row.get("resolved_target_hit"))
                    else bool(row.get("hit_target")),
                }
            )
        return rows

    def _extract_fired_features(self, row: dict[str, object]) -> list[str]:
        payload = row.get("payload")
        if not isinstance(payload, dict):
            return []
        fired: list[str] = []
        diagnostics = payload.get("diagnostics", {})
        if isinstance(diagnostics, dict):
            subscores = diagnostics.get("subscores", {})
            if isinstance(subscores, dict):
                tech = subscores.get("technical_signals", {})
                if isinstance(tech, dict):
                    for feature, value in tech.items():
                        try:
                            if float(value) >= 60.0:
                                fired.append(str(feature))
                        except Exception:
                            continue
        broad_map = {
            "ml": payload.get("ml_score"),
            "pattern": payload.get("pattern_score"),
            "volume": payload.get("volume_momentum_score"),
            "options": payload.get("options_score"),
            "sentiment": payload.get("sentiment_score"),
            "rs": payload.get("rs_score"),
        }
        for feature, value in broad_map.items():
            try:
                if float(value) >= 60.0:
                    fired.append(feature)
            except Exception:
                continue
        return sorted(set(fired))
