"""Prediction tracking, adaptive weighting, and weekly performance history."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from stock_predictor.scoring.adaptive import evolve_weights, normalize_weights
from stock_predictor.scoring.composite import CandidateScore


def _week_labels(series: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(series, utc=True, errors="coerce")
    naive = timestamps.dt.tz_localize(None)
    return naive.dt.to_period("W").astype(str)


def _coerce_bool_series(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(False, index=index, dtype=bool)
    return series.fillna(0).astype(bool)


def resolve_target_hit_outcome(
    *,
    target_price: float | None,
    latest_price: float | None,
    window_high_price: float | None = None,
    stored_hit_target: bool | None = None,
) -> tuple[bool, str]:
    """Return the canonical target-hit outcome and how it was resolved."""

    if pd.notna(window_high_price) and pd.notna(target_price):
        return bool(float(window_high_price) >= float(target_price)), "window_high"
    if pd.notna(latest_price) and pd.notna(target_price):
        return bool(float(latest_price) >= float(target_price)), "latest_price_fallback"
    return bool(stored_hit_target), "legacy_hit_target_fallback"


def compute_window_high_price(
    price_frame: pd.DataFrame | None,
    *,
    created_at: str | datetime,
    window_days: int = 7,
) -> float | None:
    """Compute the best price reached inside the intended evaluation window."""

    if price_frame is None or price_frame.empty or "high" not in price_frame:
        return None
    window_start = pd.Timestamp(created_at)
    if pd.isna(window_start):
        return None
    if window_start.tzinfo is None:
        window_start = window_start.tz_localize("UTC")
    else:
        window_start = window_start.tz_convert("UTC")
    window_start = window_start.floor("D")
    window_end = window_start + pd.Timedelta(days=window_days)
    window_slice = price_frame.loc[(price_frame.index >= window_start) & (price_frame.index <= window_end)]
    if window_slice.empty:
        return None
    high_values = pd.to_numeric(window_slice["high"], errors="coerce").dropna()
    if high_values.empty:
        return None
    return float(high_values.max())


def with_resolved_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    """Resolve reporting outcomes from persisted path-aware fields when available."""

    if frame.empty:
        return frame.copy()

    resolved = frame.copy()
    target_price = pd.to_numeric(resolved["target_price"], errors="coerce") if "target_price" in resolved else None
    latest_price = None
    for candidate_column in ("current_price", "latest_price"):
        if candidate_column in resolved:
            latest_price = pd.to_numeric(resolved[candidate_column], errors="coerce")
            break
    window_high_price = pd.to_numeric(resolved["window_high_price"], errors="coerce") if "window_high_price" in resolved else None
    stored_hit_target = _coerce_bool_series(resolved["hit_target"] if "hit_target" in resolved else None, resolved.index)
    persisted_resolved = _coerce_bool_series(
        resolved["resolved_target_hit"] if "resolved_target_hit" in resolved else None,
        resolved.index,
    )
    persisted_methods = (
        resolved["resolution_method"]
        if "resolution_method" in resolved
        else pd.Series("", index=resolved.index, dtype=object)
    )
    persisted_available = (
        resolved["resolved_target_hit"].notna()
        if "resolved_target_hit" in resolved
        else pd.Series(False, index=resolved.index, dtype=bool)
    )
    resolved_values = []
    resolution_methods = []
    for index in resolved.index:
        if bool(persisted_available.loc[index]):
            resolved_values.append(bool(persisted_resolved.loc[index]))
            persisted_method = persisted_methods.loc[index]
            resolution_methods.append(str(persisted_method) if pd.notna(persisted_method) and str(persisted_method).strip() else "persisted")
            continue
        outcome, method = resolve_target_hit_outcome(
            target_price=None if target_price is None else target_price.loc[index],
            latest_price=None if latest_price is None else latest_price.loc[index],
            window_high_price=None if window_high_price is None else window_high_price.loc[index],
            stored_hit_target=bool(stored_hit_target.loc[index]),
        )
        resolved_values.append(outcome)
        resolution_methods.append(method)
    resolved["resolved_target_hit"] = pd.Series(resolved_values, index=resolved.index, dtype=bool)
    resolved["resolution_method"] = resolution_methods
    if "realized_return" in resolved:
        realized = pd.to_numeric(resolved["realized_return"], errors="coerce")
        resolved["positive_return"] = (realized > 0).fillna(False)
    else:
        resolved["positive_return"] = False
    return resolved


_with_resolved_outcomes = with_resolved_outcomes


class BacktestTracker:
    """Persist picks, evaluate outcomes, and evolve scoring weights."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    run_id TEXT,
                    ticker TEXT,
                    created_at TEXT,
                    entry_price REAL,
                    target_price REAL,
                    final_score REAL,
                    sector TEXT,
                    confidence_label TEXT,
                    score_low REAL,
                    score_high REAL,
                    payload_json TEXT,
                    PRIMARY KEY (run_id, ticker)
                )
                """
            )
            self._ensure_columns(
                conn,
                "predictions",
                {
                    "created_at": "TEXT",
                    "entry_price": "REAL",
                    "target_price": "REAL",
                    "final_score": "REAL",
                    "sector": "TEXT",
                    "confidence_label": "TEXT",
                    "score_low": "REAL",
                    "score_high": "REAL",
                    "payload_json": "TEXT",
                },
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS evaluations (
                    run_id TEXT,
                    ticker TEXT,
                    evaluated_at TEXT,
                    latest_price REAL,
                    realized_return REAL,
                    hit_target INTEGER,
                    PRIMARY KEY (run_id, ticker)
                )
                """
            )
            self._ensure_columns(
                conn,
                "evaluations",
                {
                    "evaluated_at": "TEXT",
                    "latest_price": "REAL",
                    "realized_return": "REAL",
                    "hit_target": "INTEGER",
                    "window_high_price": "REAL",
                    "resolved_target_hit": "INTEGER",
                    "resolution_method": "TEXT",
                },
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_signals (
                    run_id TEXT,
                    ticker TEXT,
                    week TEXT,
                    signal_name TEXT,
                    signal_score REAL,
                    PRIMARY KEY (run_id, ticker, signal_name)
                )
                """
            )
            self._ensure_columns(
                conn,
                "prediction_signals",
                {
                    "week": "TEXT",
                    "signal_name": "TEXT",
                    "signal_score": "REAL",
                },
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS weight_history (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT,
                    weights_json TEXT,
                    rolling_accuracy_json TEXT,
                    wrong_streaks_json TEXT
                )
                """
            )
            self._ensure_columns(
                conn,
                "weight_history",
                {
                    "created_at": "TEXT",
                    "weights_json": "TEXT",
                    "rolling_accuracy_json": "TEXT",
                    "wrong_streaks_json": "TEXT",
                },
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_predictions (
                    run_id TEXT,
                    ticker TEXT,
                    created_at TEXT,
                    entry_price REAL,
                    target_price REAL,
                    final_score REAL,
                    payload_json TEXT,
                    PRIMARY KEY (run_id, ticker)
                )
                """
            )
            self._ensure_columns(
                conn,
                "paper_predictions",
                {
                    "created_at": "TEXT",
                    "entry_price": "REAL",
                    "target_price": "REAL",
                    "final_score": "REAL",
                    "payload_json": "TEXT",
                },
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_evaluations (
                    run_id TEXT,
                    ticker TEXT,
                    evaluated_at TEXT,
                    latest_price REAL,
                    realized_return REAL,
                    hit_target INTEGER,
                    PRIMARY KEY (run_id, ticker)
                )
                """
            )
            self._ensure_columns(
                conn,
                "paper_evaluations",
                {
                    "evaluated_at": "TEXT",
                    "latest_price": "REAL",
                    "realized_return": "REAL",
                    "hit_target": "INTEGER",
                    "window_high_price": "REAL",
                    "resolved_target_hit": "INTEGER",
                    "resolution_method": "TEXT",
                },
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS regime_contexts (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT,
                    vix_bucket TEXT,
                    spy_trend TEXT,
                    sector_leaders TEXT
                )
                """
            )
            self._ensure_columns(
                conn,
                "regime_contexts",
                {
                    "created_at": "TEXT",
                    "vix_bucket": "TEXT",
                    "spy_trend": "TEXT",
                    "sector_leaders": "TEXT",
                },
            )

    def _ensure_columns(self, conn: sqlite3.Connection, table: str, columns: Dict[str, str]) -> None:
        existing = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        for column, definition in columns.items():
            if column in existing:
                continue
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def record_predictions(self, run_id: str, candidates: Iterable[CandidateScore]) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        week = datetime.now(timezone.utc).strftime("%Y-W%W")
        with self._connect() as conn:
            for candidate in candidates:
                payload_json = json.dumps(candidate.to_dict(), default=str)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO predictions
                    (run_id, ticker, created_at, entry_price, target_price, final_score, sector,
                     confidence_label, score_low, score_high, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        candidate.ticker,
                        created_at,
                        candidate.current_price,
                        candidate.targets["tp2"],
                        candidate.final_score,
                        candidate.sector,
                        candidate.confidence_label,
                        candidate.score_low,
                        candidate.score_high,
                        payload_json,
                    ),
                )
                signal_scores = {
                    "ml": candidate.ml_score,
                    "technical": candidate.technical_score,
                    "pattern": candidate.pattern_score,
                    "volume": candidate.volume_momentum_score,
                    "options": candidate.options_score,
                    "sentiment": candidate.sentiment_score,
                    "rs": candidate.rs_score,
                }
                for signal_name, signal_score in signal_scores.items():
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO prediction_signals
                        (run_id, ticker, week, signal_name, signal_score)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (run_id, candidate.ticker, week, signal_name, signal_score),
                    )

    def record_paper_predictions(self, run_id: str, candidates: Iterable[CandidateScore]) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            for candidate in candidates:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO paper_predictions
                    (run_id, ticker, created_at, entry_price, target_price, final_score, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        candidate.ticker,
                        created_at,
                        candidate.current_price,
                        candidate.targets["tp2"],
                        candidate.final_score,
                        json.dumps(candidate.to_dict(), default=str),
                    ),
                )

    def record_regime_context(
        self,
        run_id: str,
        *,
        vix: float,
        spy_week_return: float,
        sector_leaders: Iterable[str],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO regime_contexts
                (run_id, created_at, vix_bucket, spy_trend, sector_leaders)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    datetime.now(timezone.utc).isoformat(),
                    _vix_bucket(vix),
                    _spy_trend(spy_week_return),
                    ",".join(list(sector_leaders)[:3]),
                ),
            )

    def evaluate_due_predictions(
        self,
        latest_prices: Dict[str, float],
        *,
        price_paths: Dict[str, pd.DataFrame] | None = None,
    ) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT p.*, e.evaluated_at, e.latest_price, e.realized_return, e.hit_target,
                       e.window_high_price, e.resolved_target_hit, e.resolution_method
                FROM predictions p
                LEFT JOIN evaluations e ON p.run_id = e.run_id AND p.ticker = e.ticker
                WHERE e.run_id IS NULL OR e.resolved_target_hit IS NULL
                """
            ).fetchall()
            for row in rows:
                created_at = datetime.fromisoformat(row["created_at"])
                if row["evaluated_at"] is None and created_at > cutoff:
                    continue
                latest_price = row["latest_price"] if row["latest_price"] is not None else latest_prices.get(row["ticker"])
                if latest_price is None:
                    continue
                realized_return = (
                    float(row["realized_return"])
                    if row["realized_return"] is not None
                    else latest_price / row["entry_price"] - 1
                )
                window_high_price = (
                    float(row["window_high_price"])
                    if row["window_high_price"] is not None
                    else compute_window_high_price(
                        None if not price_paths else price_paths.get(row["ticker"]),
                        created_at=created_at,
                    )
                )
                resolved_target_hit, resolution_method = resolve_target_hit_outcome(
                    target_price=row["target_price"],
                    latest_price=latest_price,
                    window_high_price=window_high_price,
                    stored_hit_target=bool(row["hit_target"]) if row["hit_target"] is not None else None,
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO evaluations
                    (run_id, ticker, evaluated_at, latest_price, realized_return, hit_target,
                     window_high_price, resolved_target_hit, resolution_method)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["run_id"],
                        row["ticker"],
                        row["evaluated_at"] or datetime.now(timezone.utc).isoformat(),
                        latest_price,
                        realized_return,
                        int(realized_return >= 0.04),
                        window_high_price,
                        int(resolved_target_hit),
                        resolution_method,
                    ),
                )

    def evaluate_due_paper_predictions(
        self,
        latest_prices: Dict[str, float],
        *,
        price_paths: Dict[str, pd.DataFrame] | None = None,
    ) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT p.*, e.evaluated_at, e.latest_price, e.realized_return, e.hit_target,
                       e.window_high_price, e.resolved_target_hit, e.resolution_method
                FROM paper_predictions p
                LEFT JOIN paper_evaluations e ON p.run_id = e.run_id AND p.ticker = e.ticker
                WHERE e.run_id IS NULL OR e.resolved_target_hit IS NULL
                """
            ).fetchall()
            for row in rows:
                created_at = datetime.fromisoformat(row["created_at"])
                if row["evaluated_at"] is None and created_at > cutoff:
                    continue
                latest_price = row["latest_price"] if row["latest_price"] is not None else latest_prices.get(row["ticker"])
                if latest_price is None:
                    continue
                realized_return = (
                    float(row["realized_return"])
                    if row["realized_return"] is not None
                    else latest_price / row["entry_price"] - 1
                )
                window_high_price = (
                    float(row["window_high_price"])
                    if row["window_high_price"] is not None
                    else compute_window_high_price(
                        None if not price_paths else price_paths.get(row["ticker"]),
                        created_at=created_at,
                    )
                )
                resolved_target_hit, resolution_method = resolve_target_hit_outcome(
                    target_price=row["target_price"],
                    latest_price=latest_price,
                    window_high_price=window_high_price,
                    stored_hit_target=bool(row["hit_target"]) if row["hit_target"] is not None else None,
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO paper_evaluations
                    (run_id, ticker, evaluated_at, latest_price, realized_return, hit_target,
                     window_high_price, resolved_target_hit, resolution_method)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["run_id"],
                        row["ticker"],
                        row["evaluated_at"] or datetime.now(timezone.utc).isoformat(),
                        latest_price,
                        realized_return,
                        int(realized_return >= 0.04),
                        window_high_price,
                        int(resolved_target_hit),
                        resolution_method,
                    ),
                )

    def latest_weights(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT weights_json
                FROM weight_history
                ORDER BY created_at DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return normalize_weights(base_weights)
        return normalize_weights(json.loads(row["weights_json"]))

    def refresh_adaptive_weights(self, run_id: str, base_weights: Dict[str, float]) -> Dict[str, float]:
        rows = self.recent_signal_rows()
        latest = self.latest_weights(base_weights)
        result = evolve_weights(base_weights, rows, current_weights=latest)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO weight_history
                (run_id, created_at, weights_json, rolling_accuracy_json, wrong_streaks_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(result.weights),
                    json.dumps(result.rolling_accuracy),
                    json.dumps(result.wrong_streaks),
                ),
            )
        return result.weights

    def recent_signal_rows(self, days: int = 35) -> list[dict]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT s.signal_name, s.signal_score, s.week,
                       COALESCE(e.resolved_target_hit, e.hit_target) AS hit_target
                FROM prediction_signals s
                JOIN predictions p ON s.run_id = p.run_id AND s.ticker = p.ticker
                JOIN evaluations e ON p.run_id = e.run_id AND p.ticker = e.ticker
                WHERE p.created_at >= ?
                """,
                (cutoff,),
            ).fetchall()
        return [dict(row) for row in rows]

    def performance_frame(self) -> pd.DataFrame:
        with self._connect() as conn:
            frame = pd.read_sql_query(
                """
                SELECT p.run_id, p.created_at, p.ticker, p.entry_price, p.target_price,
                       p.final_score, p.confidence_label, p.score_low, p.score_high,
                       e.latest_price, e.realized_return, e.hit_target,
                       e.window_high_price, e.resolved_target_hit, e.resolution_method
                FROM predictions p
                LEFT JOIN evaluations e ON p.run_id = e.run_id AND p.ticker = e.ticker
                ORDER BY p.created_at DESC
                """,
                conn,
            )
        return frame

    def weight_history_frame(self) -> pd.DataFrame:
        with self._connect() as conn:
            frame = pd.read_sql_query(
                """
                SELECT *
                FROM weight_history
                ORDER BY created_at DESC
                """,
                conn,
            )
        return frame

    def latest_run_frame(self) -> pd.DataFrame:
        with self._connect() as conn:
            frame = pd.read_sql_query(
                """
                SELECT *
                FROM predictions
                WHERE run_id = (SELECT run_id FROM predictions ORDER BY created_at DESC LIMIT 1)
                ORDER BY final_score DESC
                """,
                conn,
            )
        return frame

    def weekly_hit_rate(self) -> pd.DataFrame:
        frame = _with_resolved_outcomes(self.performance_frame()).dropna(subset=["realized_return"])
        if frame.empty:
            return frame
        frame["week"] = _week_labels(frame["created_at"])
        grouped = frame.groupby("week", as_index=False)["resolved_target_hit"].mean()
        return grouped.rename(columns={"resolved_target_hit": "hit_target"})

    def confidence_trend(self) -> pd.DataFrame:
        frame = _with_resolved_outcomes(self.performance_frame()).dropna(subset=["realized_return"]).copy()
        if frame.empty:
            return frame
        frame["week"] = _week_labels(frame["created_at"])
        frame["interval_width"] = frame["score_high"] - frame["score_low"]
        frame["high_confidence"] = (frame["confidence_label"] == "high").astype(int)
        grouped = frame.groupby("week", as_index=False).agg(
            hit_rate=("resolved_target_hit", "mean"),
            avg_interval_width=("interval_width", "mean"),
            high_confidence_share=("high_confidence", "mean"),
        )
        return grouped

    def paper_trade_summary(self, weeks: int = 8) -> Dict[str, float | str | int]:
        with self._connect() as conn:
            frame = pd.read_sql_query(
                """
                SELECT p.created_at, p.ticker, p.entry_price, p.target_price,
                       e.latest_price, e.realized_return, e.hit_target,
                       e.window_high_price, e.resolved_target_hit, e.resolution_method
                FROM paper_predictions p
                LEFT JOIN paper_evaluations e ON p.run_id = e.run_id AND p.ticker = e.ticker
                ORDER BY p.created_at DESC
                """,
                conn,
            )
        frame = _with_resolved_outcomes(frame).dropna(subset=["realized_return"]).copy()
        if frame.empty:
            return {}
        frame["week"] = _week_labels(frame["created_at"])
        unique_weeks = frame["week"].drop_duplicates().tolist()[:weeks]
        frame = frame[frame["week"].isin(unique_weeks)]
        best = frame.sort_values("realized_return", ascending=False).iloc[0]
        worst = frame.sort_values("realized_return", ascending=True).iloc[0]
        avg_win = float(frame.loc[frame["realized_return"] > 0, "realized_return"].mean() * 100.0) if (frame["realized_return"] > 0).any() else 0.0
        avg_loss = float(frame.loc[frame["realized_return"] < 0, "realized_return"].mean() * 100.0) if (frame["realized_return"] < 0).any() else 0.0
        return {
            "weeks": weeks,
            "hit_rate": float(frame["resolved_target_hit"].mean() * 100.0),
            "target_hit_rate": float(frame["resolved_target_hit"].mean() * 100.0),
            "positive_return_rate": float(frame["positive_return"].mean() * 100.0),
            "average_return": float(frame["realized_return"].mean() * 100.0),
            "average_win": avg_win,
            "average_loss": avg_loss,
            "best_pick": str(best["ticker"]),
            "best_return": float(best["realized_return"] * 100.0),
            "worst_pick": str(worst["ticker"]),
            "worst_return": float(worst["realized_return"] * 100.0),
        }

    def paper_results_frame(self) -> pd.DataFrame:
        with self._connect() as conn:
            frame = pd.read_sql_query(
                """
                SELECT p.run_id, p.created_at, p.ticker, p.entry_price, p.target_price, p.final_score,
                       e.latest_price AS current_price, e.realized_return, e.hit_target,
                       e.window_high_price, e.resolved_target_hit, e.resolution_method
                FROM paper_predictions p
                LEFT JOIN paper_evaluations e ON p.run_id = e.run_id AND p.ticker = e.ticker
                ORDER BY p.created_at DESC, p.final_score DESC
                """,
                conn,
            )
        return frame

    def paper_trade_results_summary(self) -> Dict[str, float | str | int]:
        frame = _with_resolved_outcomes(self.paper_results_frame()).dropna(subset=["realized_return"]).copy()
        if frame.empty:
            return {}
        frame["week"] = _week_labels(frame["created_at"])
        weekly = frame.groupby("week", as_index=False).agg(
            avg_return=("realized_return", "mean"),
            win_rate=("resolved_target_hit", "mean"),
            positive_return_rate=("positive_return", "mean"),
        )
        latest_streak = 0
        direction = None
        for _, row in weekly.sort_values("week", ascending=False).iterrows():
            hit = row["avg_return"] > 0
            if direction is None:
                direction = hit
            if hit != direction:
                break
            latest_streak += 1
        best_week = weekly.sort_values("avg_return", ascending=False).iloc[0]
        worst_week = weekly.sort_values("avg_return", ascending=True).iloc[0]
        return {
            "win_rate": float(frame["resolved_target_hit"].mean() * 100.0),
            "target_hit_rate": float(frame["resolved_target_hit"].mean() * 100.0),
            "positive_return_rate": float(frame["positive_return"].mean() * 100.0),
            "average_return": float(frame["realized_return"].mean() * 100.0),
            "best_week": str(best_week["week"]),
            "best_week_return": float(best_week["avg_return"] * 100.0),
            "worst_week": str(worst_week["week"]),
            "worst_week_return": float(worst_week["avg_return"] * 100.0),
            "streak_weeks": int(latest_streak),
            "streak_direction": "winning" if direction else "losing",
        }

    def weeks_tracked(self) -> int:
        frame = self.paper_results_frame().dropna(subset=["realized_return"]).copy()
        if frame.empty:
            return 0
        frame["week"] = _week_labels(frame["created_at"])
        return int(frame["week"].nunique())

    def threshold_recommendation(self, *, min_weeks: int = 4) -> Dict[str, float | int] | None:
        frame = _with_resolved_outcomes(self.paper_results_frame()).dropna(subset=["realized_return"]).copy()
        if frame.empty:
            return None
        frame["week"] = _week_labels(frame["created_at"])
        tracked_weeks = int(frame["week"].nunique())
        if tracked_weeks < min_weeks:
            return None

        best = None
        observed_min = max(int(frame["final_score"].min() // 1), 45)
        observed_max = min(int(frame["final_score"].max() // 1) + 1, 90)
        for threshold in range(observed_min, observed_max + 1, 2):
            subset = frame.loc[frame["final_score"] >= threshold]
            if len(subset) < min_weeks or subset["week"].nunique() < min_weeks:
                continue
            win_rate = float(subset["resolved_target_hit"].mean())
            avg_return = float(subset["realized_return"].mean())
            candidate = {
                "threshold": float(threshold),
                "win_rate": win_rate * 100.0,
                "average_return": avg_return * 100.0,
                "sample_count": int(len(subset)),
                "weeks": int(subset["week"].nunique()),
            }
            if best is None or (candidate["win_rate"], candidate["average_return"]) > (
                best["win_rate"],
                best["average_return"],
            ):
                best = candidate
        return best

    def regime_memory_summary(
        self,
        *,
        vix: float,
        spy_week_return: float,
        sector_leaders: Iterable[str],
    ) -> Dict[str, float | str | int]:
        current_bucket = _vix_bucket(vix)
        current_trend = _spy_trend(spy_week_return)
        current_leaders = {leader for leader in sector_leaders if leader}
        with self._connect() as conn:
            frame = pd.read_sql_query(
                """
                SELECT rc.run_id, rc.vix_bucket, rc.spy_trend, rc.sector_leaders,
                       p.target_price, e.latest_price, e.realized_return, e.hit_target,
                       e.window_high_price, e.resolved_target_hit, e.resolution_method
                FROM regime_contexts rc
                JOIN paper_predictions p ON rc.run_id = p.run_id
                JOIN paper_evaluations e ON p.run_id = e.run_id AND p.ticker = e.ticker
                """,
                conn,
            )
        frame = _with_resolved_outcomes(frame)
        if frame.empty:
            return {}
        frame = frame.loc[
            (frame["vix_bucket"] == current_bucket)
            & (frame["spy_trend"] == current_trend)
        ].copy()
        if frame.empty:
            return {}
        if current_leaders:
            frame["leader_overlap"] = frame["sector_leaders"].fillna("").map(
                lambda value: len(current_leaders.intersection({item for item in str(value).split(",") if item}))
            )
            frame = frame.loc[frame["leader_overlap"] > 0]
            if frame.empty:
                return {}
        grouped = frame.groupby("run_id", as_index=False).agg(
            win_rate=("resolved_target_hit", "mean"),
            avg_return=("realized_return", "mean"),
        )
        return {
            "sample_runs": int(len(grouped)),
            "vix_bucket": current_bucket,
            "spy_trend": current_trend,
            "win_rate": float(grouped["win_rate"].mean() * 100.0),
            "target_hit_rate": float(grouped["win_rate"].mean() * 100.0),
            "average_return": float(grouped["avg_return"].mean() * 100.0),
        }


class SignalAttributionTracker:
    """Track which signals were active for weekly paper picks and how they performed."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS signal_outcomes (
                    run_id TEXT,
                    ticker TEXT,
                    created_at TEXT,
                    entry_price REAL,
                    target_price REAL,
                    signal_name TEXT,
                    signal_score REAL,
                    signal_on INTEGER,
                    evaluated_at TEXT,
                    latest_price REAL,
                    realized_return REAL,
                    hit_target INTEGER,
                    window_high_price REAL,
                    resolved_target_hit INTEGER,
                    resolution_method TEXT,
                    PRIMARY KEY (run_id, ticker, signal_name)
                )
                """
            )
            self._ensure_columns(
                conn,
                "signal_outcomes",
                {
                    "window_high_price": "REAL",
                    "resolved_target_hit": "INTEGER",
                    "resolution_method": "TEXT",
                },
            )

    def _ensure_columns(self, conn: sqlite3.Connection, table: str, columns: Dict[str, str]) -> None:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for column, definition in columns.items():
            if column in existing:
                continue
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def record_predictions(self, run_id: str, candidates: Iterable[CandidateScore]) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            for candidate in candidates:
                signal_scores = {
                    "ml": candidate.ml_score,
                    "technical": candidate.technical_score,
                    "volume": candidate.volume_momentum_score,
                    "options": candidate.options_score,
                    "sentiment": candidate.sentiment_score,
                    "rs": candidate.rs_score,
                    "pattern": candidate.pattern_score,
                }
                for signal_name, signal_score in signal_scores.items():
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO signal_outcomes
                        (run_id, ticker, created_at, entry_price, target_price, signal_name,
                         signal_score, signal_on, evaluated_at, latest_price, realized_return, hit_target,
                         window_high_price, resolved_target_hit, resolution_method)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, NULL)
                        """,
                        (
                            run_id,
                            candidate.ticker,
                            created_at,
                            candidate.current_price,
                            candidate.targets["tp2"],
                            signal_name,
                            float(signal_score),
                            int(signal_score >= 60.0),
                        ),
                    )

    def evaluate_due_predictions(
        self,
        latest_prices: Dict[str, float],
        *,
        price_paths: Dict[str, pd.DataFrame] | None = None,
    ) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM signal_outcomes
                WHERE evaluated_at IS NULL OR resolved_target_hit IS NULL
                """
            ).fetchall()
            for row in rows:
                created_at = datetime.fromisoformat(row["created_at"])
                if row["evaluated_at"] is None and created_at > cutoff:
                    continue
                latest_price = row["latest_price"] if row["latest_price"] is not None else latest_prices.get(row["ticker"])
                if latest_price is None:
                    continue
                realized_return = (
                    float(row["realized_return"])
                    if row["realized_return"] is not None
                    else latest_price / row["entry_price"] - 1.0
                )
                window_high_price = (
                    float(row["window_high_price"])
                    if row["window_high_price"] is not None
                    else compute_window_high_price(
                        None if not price_paths else price_paths.get(row["ticker"]),
                        created_at=created_at,
                    )
                )
                resolved_target_hit, resolution_method = resolve_target_hit_outcome(
                    target_price=row["target_price"],
                    latest_price=latest_price,
                    window_high_price=window_high_price,
                    stored_hit_target=bool(row["hit_target"]) if row["hit_target"] is not None else None,
                )
                conn.execute(
                    """
                    UPDATE signal_outcomes
                    SET evaluated_at = ?, latest_price = ?, realized_return = ?, hit_target = ?,
                        window_high_price = ?, resolved_target_hit = ?, resolution_method = ?
                    WHERE run_id = ? AND ticker = ? AND signal_name = ?
                    """,
                    (
                        row["evaluated_at"] or datetime.now(timezone.utc).isoformat(),
                        latest_price,
                        realized_return,
                        int(realized_return >= 0.04),
                        window_high_price,
                        int(resolved_target_hit),
                        resolution_method,
                        row["run_id"],
                        row["ticker"],
                        row["signal_name"],
                    ),
                )

    def signal_summary(self, *, weeks: int = 4) -> Dict[str, object]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=7 * weeks)).isoformat()
        with self._connect() as conn:
            frame = pd.read_sql_query(
                """
                SELECT created_at, signal_name, signal_score, signal_on, target_price, latest_price,
                       realized_return, hit_target, window_high_price, resolved_target_hit, resolution_method
                FROM signal_outcomes
                WHERE evaluated_at IS NOT NULL AND created_at >= ?
                """,
                conn,
                params=(cutoff,),
            )
        frame = _with_resolved_outcomes(frame)
        if frame.empty:
            return {}
        active = frame.loc[frame["signal_on"] == 1].copy()
        if active.empty:
            return {}
        grouped = (
            active.groupby("signal_name", as_index=False)
            .agg(
                wins=("resolved_target_hit", "sum"),
                total=("resolved_target_hit", "count"),
                win_rate=("resolved_target_hit", "mean"),
                avg_return=("realized_return", "mean"),
            )
            .sort_values(["win_rate", "avg_return"], ascending=False)
        )
        if grouped.empty:
            return {}
        best = grouped.iloc[0]
        return {
            "best_signal": str(best["signal_name"]),
            "best_signal_win_rate": float(best["win_rate"] * 100.0),
            "best_signal_average_return": float(best["avg_return"] * 100.0),
            "rows": grouped.to_dict(orient="records"),
        }


def _vix_bucket(vix: float) -> str:
    if vix >= 30.0:
        return "30+"
    if vix >= 25.0:
        return "25-30"
    if vix >= 20.0:
        return "20-25"
    return "<20"


def _spy_trend(spy_week_return: float) -> str:
    if spy_week_return <= -0.02:
        return "falling"
    if spy_week_return >= 0.02:
        return "rising"
    return "flat"
