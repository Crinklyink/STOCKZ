"""Main window for the native Stock Predictor application."""

from __future__ import annotations

import json
import os
import sqlite3
import smtplib
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import pandas as pd
from PySide6.QtCore import QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QFont, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from stock_predictor.config import PROJECT_ROOT, get_config
from stock_predictor.output.backtest import with_resolved_outcomes
from ui.analyze_view import AnalyzeView
from ui.dashboard import DashboardView
from ui.history_view import HistoryView, PerformanceView
from ui.picks_view import PicksView
from ui.settings_view import SettingsView
from ui.widgets import COLORS, ScanProgressDialog, ScanWorker, SidebarButton, apple_font, apply_card_shadow, css_color, is_dark_mode, style_primary_button, theme_colors


APP_VERSION = "3.0"


@dataclass(slots=True)
class AppState:
    payload: dict[str, Any]
    single_analysis: dict[str, Any]
    single_analysis_history: list[dict[str, Any]]
    model_metadata: dict[str, Any]
    rolling_summary: dict[str, Any]
    weekly_rows: list[dict[str, Any]]
    history_rows: list[dict[str, Any]]
    cumulative_rows: list[dict[str, Any]]
    history_weeks: list[dict[str, Any]]
    history_details: list[dict[str, Any]]
    breakdowns: dict[str, list[dict[str, Any]]]
    settings: dict[str, Any]
    last_scan_text: str
    top_pick_text: str


class AppDataService:
    """Read live state from the existing predictor artifacts and databases."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.config = get_config()
        self.env_path = project_root / ".env"

    def load_state(self) -> AppState:
        payload = self._load_json(self.config.latest_scan_path)
        single_analysis = self._load_json(self.config.latest_single_analysis_path)
        single_analysis_history = self._load_analysis_history()
        model_metadata = self._load_json(self.config.xgb_metadata_path)
        history_details = self._load_history_details()
        weekly_rows = self._build_weekly_rows(history_details)
        history_rows = self._build_history_rows(history_details)
        cumulative_rows = self._build_cumulative_rows(weekly_rows)
        breakdowns = self._build_breakdowns(history_details)
        rolling_summary = self._build_rolling_summary(history_details, weekly_rows)
        last_scan_text = self._format_last_scan(payload)
        display = payload.get("selected", []) or payload.get("display_candidates", [])
        top_pick_text = "--"
        if display:
            top_pick_text = f"{display[0].get('ticker', '--')} {float(display[0].get('final_score', 0.0)):.2f}"
        return AppState(
            payload=payload,
            single_analysis=single_analysis,
            single_analysis_history=single_analysis_history,
            model_metadata=model_metadata,
            rolling_summary=rolling_summary,
            weekly_rows=weekly_rows,
            history_rows=history_rows,
            cumulative_rows=cumulative_rows,
            history_weeks=history_rows,
            history_details=history_details,
            breakdowns=breakdowns,
            settings=self._load_settings(),
            last_scan_text=last_scan_text,
            top_pick_text=top_pick_text,
        )

    def save_settings(self, values: dict[str, Any]) -> None:
        env_map = self._read_env_map()
        updates = {
            "ALERT_EMAIL": values.get("alert_email", ""),
            "SMTP_USER": values.get("smtp_user", ""),
            "SMTP_PASS": values.get("smtp_pass", ""),
            "SUNDAY_SCAN_TIME": values.get("sunday_time", "18:00"),
            "WEDNESDAY_CHECK_TIME": values.get("wednesday_time", "18:00"),
            "AUTO_OPEN_APP_AFTER_SCAN": "1" if values.get("auto_open_app", True) else "0",
            "RETRAIN_MODEL_WEEKLY": "1" if values.get("retrain_weekly", True) else "0",
            "APP_VIX_CUTOFF": str(values.get("vix_cutoff", 30)),
            "APP_MIN_SCORE_THRESHOLD": str(values.get("min_score_threshold", 54)),
            "APP_MIN_RR_RATIO": f"{float(values.get('min_rr_ratio', 1.0)):.1f}",
            "APP_MAX_PICKS": str(values.get("max_picks", 5)),
            "DEFAULT_UNIVERSE": str(values.get("universe", "full")),
        }
        env_map.update(updates)
        lines = [
            "# Stock Predictor desktop app settings",
            *[f"{key}={value}" for key, value in sorted(env_map.items()) if str(value).strip()],
            "",
        ]
        self.env_path.write_text("\n".join(lines), encoding="utf-8")

    def test_email(self, values: dict[str, Any]) -> None:
        config = get_config()
        config.alert_email = str(values.get("alert_email", "")).strip()
        config.smtp_username = str(values.get("smtp_user", "")).strip()
        config.smtp_password = str(values.get("smtp_pass", ""))
        config.smtp_from_email = config.smtp_username
        if not all([config.alert_email, config.smtp_username, config.smtp_password]):
            raise ValueError("Email address, SMTP user, and app password are all required.")
        message = EmailMessage()
        message["From"] = config.smtp_from_email or config.smtp_username
        message["To"] = config.alert_email
        message["Subject"] = "Stock Predictor test email"
        message.set_content("Your Stock Predictor desktop app is connected and ready.")
        with smtplib.SMTP(config.smtp_host or "smtp.gmail.com", config.smtp_port) as server:
            server.starttls()
            server.login(config.smtp_username, config.smtp_password)
            server.send_message(message)

    def _load_settings(self) -> dict[str, Any]:
        env_map = self._read_env_map()
        return {
            "email_enabled": bool(env_map.get("ALERT_EMAIL") and env_map.get("SMTP_USER") and env_map.get("SMTP_PASS")),
            "alert_email": env_map.get("ALERT_EMAIL", ""),
            "smtp_user": env_map.get("SMTP_USER", ""),
            "smtp_pass": env_map.get("SMTP_PASS", ""),
            "sunday_time": env_map.get("SUNDAY_SCAN_TIME", "18:00"),
            "wednesday_time": env_map.get("WEDNESDAY_CHECK_TIME", "18:00"),
            "auto_open_app": env_map.get("AUTO_OPEN_APP_AFTER_SCAN", "1") != "0",
            "retrain_weekly": env_map.get("RETRAIN_MODEL_WEEKLY", "1") != "0",
            "vix_cutoff": float(env_map.get("APP_VIX_CUTOFF", "30")),
            "min_score_threshold": float(env_map.get("APP_MIN_SCORE_THRESHOLD", "54")),
            "min_rr_ratio": float(env_map.get("APP_MIN_RR_RATIO", "1.0")),
            "max_picks": int(float(env_map.get("APP_MAX_PICKS", "5"))),
            "universe": env_map.get("DEFAULT_UNIVERSE", "full"),
        }

    def _read_env_map(self) -> dict[str, str]:
        if not self.env_path.exists():
            return {}
        result: dict[str, str] = {}
        for raw_line in self.env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
        return result

    def _load_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _load_analysis_history(self) -> list[dict[str, Any]]:
        path = self.config.single_analysis_history_path
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        items = payload.get("items", payload)
        if not isinstance(items, list):
            return []
        return [item for item in items if isinstance(item, dict)]

    def _load_history_details(self) -> list[dict[str, Any]]:
        db_path = self.config.paper_trade_db
        if not db_path.exists():
            return []
        query = """
            SELECT p.run_id, p.created_at, p.ticker, p.entry_price, p.target_price, p.final_score, p.payload_json,
                   e.latest_price, e.realized_return, e.hit_target,
                   e.window_high_price, e.resolved_target_hit, e.resolution_method,
                   rc.vix_bucket
            FROM paper_predictions p
            LEFT JOIN paper_evaluations e ON p.run_id = e.run_id AND p.ticker = e.ticker
            LEFT JOIN regime_contexts rc ON p.run_id = rc.run_id
            ORDER BY p.created_at DESC, p.final_score DESC
        """
        try:
            with sqlite3.connect(db_path) as conn:
                frame = pd.read_sql_query(query, conn)
        except Exception:
            return []
        if frame.empty:
            return []
        frame = with_resolved_outcomes(frame)
        frame["created_ts"] = pd.to_datetime(frame["created_at"], utc=True, errors="coerce", format="mixed")
        frame["week_key"] = frame["created_ts"].dt.tz_localize(None).dt.to_period("W").map(lambda period: period.start_time)
        frame["week_label"] = frame["created_ts"].dt.strftime("%b %d")
        frame["run_label"] = frame["created_ts"].dt.strftime("%b %d · %I:%M%p").str.replace(" 0", " ", regex=False)
        details: list[dict[str, Any]] = []
        for row in frame.to_dict(orient="records"):
            payload = {}
            try:
                payload = json.loads(row.get("payload_json") or "{}")
            except Exception:
                payload = {}
            realized_return = row.get("realized_return")
            current_price = float(row.get("latest_price") or row.get("entry_price") or 0.0)
            target_price = float(row.get("target_price") or 0.0)
            details.append(
                {
                    "run_id": row.get("run_id"),
                    "run_label": row.get("run_label"),
                    "created_ts": row.get("created_ts"),
                    "week_key": str(row.get("week_key")),
                    "week_label": row.get("week_label"),
                    "ticker": row.get("ticker"),
                    "entry_price": float(row.get("entry_price") or 0.0),
                    "current_price": current_price,
                    "target_price": target_price,
                    "window_high_price": float(row.get("window_high_price") or 0.0) if pd.notna(row.get("window_high_price")) else None,
                    "realized_return_pct": float(realized_return * 100.0) if pd.notna(realized_return) else None,
                    "hit_target": bool(row.get("resolved_target_hit")) if pd.notna(row.get("resolved_target_hit")) else False,
                    "resolution_method": row.get("resolution_method") or "legacy_hit_target_fallback",
                    "final_score": float(row.get("final_score") or 0.0),
                    "sector": payload.get("sector", "Unknown"),
                    "tier_label": payload.get("tier_label", ""),
                    "company_name": payload.get("company_name", ""),
                    "vix_bucket": row.get("vix_bucket") or "unknown",
                }
            )
        return details

    def _build_history_rows(self, details: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not details:
            return []
        frame = pd.DataFrame(details)
        if frame.empty or "run_id" not in frame.columns:
            return []
        frame["created_ts"] = pd.to_datetime(frame["created_ts"], utc=True, errors="coerce", format="mixed")
        history_rows: list[dict[str, Any]] = []
        for run_id, group in frame.groupby("run_id", sort=False):
            ordered_group = group.sort_values("final_score", ascending=False, na_position="last")
            completed = ordered_group.dropna(subset=["realized_return_pct"])
            best_pick = "-"
            if not completed.empty:
                best_row = completed.sort_values("realized_return_pct", ascending=False).iloc[0]
                best_pick = f"{best_row['ticker']} {best_row['realized_return_pct']:+.1f}%"
            target_hits = int(completed["hit_target"].fillna(False).sum()) if not completed.empty else "-"
            target_hit_rate = float(completed["hit_target"].fillna(False).mean() * 100.0) if not completed.empty else None
            avg_return = float(completed["realized_return_pct"].mean()) if not completed.empty else None
            created_ts = pd.to_datetime(ordered_group["created_ts"].iloc[0], utc=True, errors="coerce", format="mixed")
            history_rows.append(
                {
                    "run_id": str(run_id),
                    "run_label": str(ordered_group["run_label"].iloc[0]),
                    "created_at": created_ts.isoformat() if pd.notna(created_ts) else "",
                    "picks": int(len(ordered_group)),
                    "target_hits": target_hits,
                    "target_hit_rate": target_hit_rate,
                    "target_hit_rate_label": f"{target_hit_rate:.0f}%" if target_hit_rate is not None else "pending",
                    "avg_return": avg_return,
                    "avg_return_label": f"{avg_return:+.1f}%" if avg_return is not None else "pending",
                    "best_pick": best_pick,
                }
            )
        history_rows.sort(key=lambda row: row.get("created_at", ""), reverse=True)
        return history_rows

    def _build_weekly_rows(self, details: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not details:
            return []
        frame = pd.DataFrame(details)
        if {"run_id", "created_ts", "week_key"}.issubset(frame.columns):
            run_meta = (
                frame.assign(created_ts=pd.to_datetime(frame["created_ts"], utc=True, errors="coerce", format="mixed"))
                .groupby(["week_key", "run_id"], as_index=False)["created_ts"]
                .max()
                .sort_values("created_ts")
            )
            latest_run_ids = set(run_meta.groupby("week_key").tail(1)["run_id"].tolist())
            if latest_run_ids:
                frame = frame[frame["run_id"].isin(latest_run_ids)].copy()
        weekly_rows: list[dict[str, Any]] = []
        for week_key, group in frame.groupby("week_key", sort=False):
            completed = group.dropna(subset=["realized_return_pct"])
            best_pick = "-"
            if not completed.empty:
                best_row = completed.sort_values("realized_return_pct", ascending=False).iloc[0]
                best_pick = f"{best_row['ticker']} {best_row['realized_return_pct']:+.1f}%"
            target_hits = int(completed["hit_target"].fillna(False).sum()) if not completed.empty else "-"
            target_hit_rate = (
                float(completed["hit_target"].fillna(False).mean() * 100.0)
                if not completed.empty
                else None
            )
            positive_return_rate = (
                float((completed["realized_return_pct"] > 0).mean() * 100.0)
                if not completed.empty
                else None
            )
            weekly_rows.append(
                {
                    "week_key": str(week_key),
                    "week_label": str(group["week_label"].iloc[0]),
                    "picks": int(len(group)),
                    "target_hits": target_hits,
                    "target_hit_rate": target_hit_rate,
                    "target_hit_rate_label": f"{target_hit_rate:.0f}%" if target_hit_rate is not None else "pending",
                    "positive_return_rate": positive_return_rate,
                    "positive_return_rate_label": f"{positive_return_rate:.0f}%" if positive_return_rate is not None else "pending",
                    # Backward-compatible aliases for older consumers; both now mean target hits.
                    "winners": target_hits,
                    "hit_rate": target_hit_rate,
                    "hit_rate_label": f"{target_hit_rate:.0f}%" if target_hit_rate is not None else "pending",
                    "avg_return": float(completed["realized_return_pct"].mean()) if not completed.empty else None,
                    "avg_return_label": f"{float(completed['realized_return_pct'].mean()):+.1f}%" if not completed.empty else "pending",
                    "best_pick": best_pick,
                }
            )
        weekly_rows.sort(key=lambda row: row["week_key"], reverse=True)
        return weekly_rows

    def _build_cumulative_rows(self, weekly_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cumulative = 0.0
        rows: list[dict[str, Any]] = []
        for row in reversed(weekly_rows):
            if row.get("avg_return") is None:
                continue
            cumulative += float(row["avg_return"])
            rows.append(
                {
                    "week": row["week_key"],
                    "week_label": row["week_label"],
                    "avg_return": float(row["avg_return"]),
                    "cumulative_return": cumulative,
                }
            )
        return rows

    def _build_breakdowns(self, details: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        if not details:
            return {"sector": [], "vix_bucket": [], "tier": []}
        frame = pd.DataFrame(details).dropna(subset=["realized_return_pct"]).copy()
        if frame.empty:
            return {"sector": [], "vix_bucket": [], "tier": []}
        return {
            "sector": self._aggregate_breakdown(frame, "sector"),
            "vix_bucket": self._aggregate_breakdown(frame, "vix_bucket"),
            "tier": self._aggregate_breakdown(frame, "tier_label"),
        }

    def _aggregate_breakdown(self, frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
        grouped = (
            frame.groupby(column, as_index=False)
            .agg(avg_return=("realized_return_pct", "mean"), count=("ticker", "count"))
            .sort_values("avg_return", ascending=False)
        )
        rows = []
        for row in grouped.to_dict(orient="records"):
            rows.append(
                {
                    "label": str(row.get(column) or "Unknown"),
                    "avg_return": float(row.get("avg_return") or 0.0),
                    "count": int(row.get("count") or 0),
                }
            )
        return rows

    def _build_rolling_summary(self, details: list[dict[str, Any]], weekly_rows: list[dict[str, Any]]) -> dict[str, Any]:
        completed = [detail for detail in details if detail.get("realized_return_pct") is not None]
        if not completed:
            return {}
        returns = [float(detail["realized_return_pct"]) for detail in completed]
        target_hit_rate = (
            sum(bool(detail.get("hit_target")) for detail in completed) / len(completed)
        ) * 100.0
        positive_return_rate = (sum(value > 0 for value in returns) / len(returns)) * 100.0
        best_week = next((row for row in weekly_rows if row.get("avg_return") is not None), None)
        worst_week = next((row for row in reversed(weekly_rows) if row.get("avg_return") is not None), None)
        streak_weeks = 0
        streak_direction = None
        for row in weekly_rows:
            avg_return = row.get("avg_return")
            if avg_return is None:
                continue
            is_winning = avg_return > 0
            if streak_direction is None:
                streak_direction = "winning" if is_winning else "losing"
            if (streak_direction == "winning" and not is_winning) or (streak_direction == "losing" and is_winning):
                break
            streak_weeks += 1
        best = max((row for row in weekly_rows if row.get("avg_return") is not None), key=lambda item: float(item["avg_return"]), default=None)
        worst = min((row for row in weekly_rows if row.get("avg_return") is not None), key=lambda item: float(item["avg_return"]), default=None)
        return {
            "target_hit_rate": target_hit_rate,
            "positive_return_rate": positive_return_rate,
            # Backward-compatible alias for older consumers; now means target hit rate.
            "win_rate": target_hit_rate,
            "average_return": sum(returns) / len(returns),
            "best_week": best["week_label"] if best else "-",
            "best_week_return": float(best["avg_return"]) if best else None,
            "worst_week": worst["week_label"] if worst else "-",
            "worst_week_return": float(worst["avg_return"]) if worst else None,
            "streak_weeks": streak_weeks,
            "streak_direction": streak_direction or "none",
            "streak_basis": "positive average return",
        }

    def _format_last_scan(self, payload: dict[str, Any]) -> str:
        generated_at = payload.get("generated_at")
        if generated_at:
            try:
                timestamp = pd.to_datetime(generated_at, utc=True).tz_convert("America/New_York")
                return timestamp.strftime("%a %b %d %-I:%M%p")
            except Exception:
                pass
        if self.config.latest_scan_path.exists():
            return datetime.fromtimestamp(self.config.latest_scan_path.stat().st_mtime).strftime("%a %b %d %I:%M%p")
        return "Never"


class StockPredictorWindow(QMainWindow):
    """Professional native macOS desktop shell for the stock predictor."""

    refresh_requested = Signal()

    def __init__(self, *, enable_tray: bool = True) -> None:
        super().__init__()
        self.project_root = PROJECT_ROOT
        self.config = get_config()
        self.data_service = AppDataService(self.project_root)
        self.state = self.data_service.load_state()
        self.scan_worker: ScanWorker | None = None
        self.scan_dialog: ScanProgressDialog | None = None
        self.tray: QSystemTrayIcon | None = None
        self.refresh_requested.connect(self.refresh_views)

        self.setWindowTitle(f"Stock Predictor v{APP_VERSION}")
        self.setMinimumSize(1200, 800)
        self.resize(1280, 860)
        self.setUnifiedTitleAndToolBarOnMac(True)

        icon_path = self.project_root / "assets" / "icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        app = QApplication.instance()
        if app is not None and hasattr(app, "paletteChanged"):
            app.paletteChanged.connect(lambda *_: self._handle_palette_change())

        self._build_ui()
        if enable_tray:
            self._build_tray()
        self.apply_theme()
        self.refresh_views()

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setCentralWidget(central)

        self.sidebar = QFrame()
        self.sidebar.setFixedWidth(220)
        side_layout = QVBoxLayout(self.sidebar)
        side_layout.setContentsMargins(18, 20, 18, 20)
        side_layout.setSpacing(12)

        self.logo = QLabel("Stock Predictor")
        self.logo.setFont(apple_font("display", 20, QFont.Weight.DemiBold))
        side_layout.addWidget(self.logo)
        side_layout.addSpacing(10)

        self.nav_buttons: dict[str, SidebarButton] = {}
        nav_items = [
            ("dashboard", "⊞  Dashboard"),
            ("picks", "◎  This Week's Picks"),
            ("analyze", "◌  Analyze Stock"),
            ("performance", "↗  Performance"),
            ("history", "☰  History"),
            ("settings", "⚙  Settings"),
        ]
        for key, label in nav_items:
            button = SidebarButton(label)
            button.clicked.connect(lambda checked, view_key=key: self._set_view(view_key))
            self.nav_buttons[key] = button
            side_layout.addWidget(button)
        side_layout.addStretch(1)

        self.last_scan_label = QLabel("Last scan\n--")
        self.auc_label = QLabel("AUC --")
        self.last_scan_label.setFont(apple_font("text", 12, QFont.Weight.Normal))
        self.auc_label.setFont(apple_font("text", 12, QFont.Weight.Normal))
        self.run_button = QPushButton("Run Scan Now")
        style_primary_button(self.run_button)
        self.run_button.clicked.connect(self.run_scan_now)
        side_layout.addWidget(self.last_scan_label)
        side_layout.addWidget(self.auc_label)
        side_layout.addSpacing(6)
        side_layout.addWidget(self.run_button)
        layout.addWidget(self.sidebar)

        self.stack = QStackedWidget()
        self.dashboard_view = DashboardView()
        self.picks_view = PicksView()
        self.analyze_view = AnalyzeView(self.project_root)
        self.performance_view = PerformanceView()
        self.history_view = HistoryView()
        self.settings_view = SettingsView()
        self.settings_view.settings_saved.connect(self.save_settings)
        self.settings_view.test_email_requested.connect(self.test_email_settings)
        self.settings_view.retrain_requested.connect(self.retrain_model)
        self.settings_view.clear_cache_requested.connect(self.clear_cache)
        self.settings_view.view_logs_requested.connect(self.view_logs)

        self.view_order = {
            "dashboard": self.dashboard_view,
            "picks": self.picks_view,
            "analyze": self.analyze_view,
            "performance": self.performance_view,
            "history": self.history_view,
            "settings": self.settings_view,
        }
        for widget in self.view_order.values():
            self.stack.addWidget(widget)
        layout.addWidget(self.stack, 1)
        self._set_view("dashboard")

    def apply_theme(self) -> None:
        colors = theme_colors(self)
        self.setStyleSheet(f"background: {css_color(colors['window'])}; color: {css_color(colors['text'])};")
        self.sidebar.setStyleSheet(f"background: {css_color(colors['window'])}; border: 0;")
        self.logo.setStyleSheet(f"color: {css_color(colors['text'])};")
        subtle = css_color(colors["text_tertiary"])
        self.last_scan_label.setStyleSheet(f"color: {subtle};")
        self.auc_label.setStyleSheet(f"color: {subtle};")
        style_primary_button(self.run_button)
        apply_card_shadow(self.run_button, enabled=False)
        self.stack.setStyleSheet(f"background: {css_color(colors['window'])};")
        for button in self.nav_buttons.values():
            button.apply_theme()
        for view in self.view_order.values():
            if hasattr(view, "apply_theme"):
                view.apply_theme()

    def _handle_palette_change(self) -> None:
        self.apply_theme()
        self.refresh_views()

    def _build_tray(self) -> None:
        self.tray = QSystemTrayIcon(self)
        if not self.windowIcon().isNull():
            self.tray.setIcon(self.windowIcon())
        menu = QMenu(self)
        open_action = QAction("Open Stock Predictor", self)
        open_action.triggered.connect(self.showNormal)
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        menu.addAction(open_action)
        menu.addSeparator()
        menu.addAction(quit_action)
        self.tray.setContextMenu(menu)
        self.tray.show()

    def _set_view(self, key: str) -> None:
        for button_key, button in self.nav_buttons.items():
            button.setChecked(button_key == key)
        self.stack.setCurrentWidget(self.view_order[key])

    def refresh_views(self) -> None:
        self.state = self.data_service.load_state()
        state_dict = self._state_to_dict(self.state)
        self.dashboard_view.update_data(state_dict)
        self.picks_view.update_data(state_dict)
        self.analyze_view.update_data(state_dict)
        self.performance_view.update_data(state_dict)
        self.history_view.update_data(state_dict)
        self.settings_view.update_data(state_dict)

        self.last_scan_label.setText(f"Last scan: {self.state.last_scan_text}")
        self.auc_label.setText(f"Model AUC: {float(self.state.model_metadata.get('auc', 0.0)):.3f}")
        self.setWindowTitle(f"Stock Predictor v{APP_VERSION} — {self.state.top_pick_text}")

    def run_scan_now(self) -> None:
        if self.scan_worker is not None and self.scan_worker.isRunning():
            return
        universe = self.state.settings.get("universe", "full")
        top_n = int(self.state.settings.get("max_picks", 10))
        self.scan_dialog = ScanProgressDialog(self)
        self.scan_worker = ScanWorker(self.project_root, universe_mode=universe, top_n=top_n)
        self.scan_worker.progress_changed.connect(self.scan_dialog.update_progress)
        self.scan_worker.scan_finished.connect(self._handle_scan_finished)
        self.scan_worker.scan_failed.connect(self._handle_scan_failed)
        self.scan_worker.start()
        self.scan_dialog.exec()

    def _handle_scan_finished(self, payload: dict[str, Any]) -> None:
        if self.scan_dialog:
            selected = payload.get("selected", [])
            summary = f"{len(selected)} official picks found." if selected else "No official picks this week."
            self.scan_dialog.set_completed(summary)
            QTimer.singleShot(900, self.scan_dialog.accept)
        self.refresh_views()
        selected_count = len(payload.get("selected", []))
        self._show_notification(
            "📈 Stock Predictor",
            f"{selected_count} picks found this week" if selected_count else "⛔ No picks (VIX too high)",
        )
        self._set_dock_badge(selected_count)
        self.scan_worker = None

    def _handle_scan_failed(self, message: str) -> None:
        if self.scan_dialog:
            self.scan_dialog.set_failed(message)
        QMessageBox.warning(self, "Scan failed", message)
        self.scan_worker = None

    def save_settings(self, values: dict[str, Any]) -> None:
        try:
            self.data_service.save_settings(values)
            subprocess.run([os.sys.executable, "setup_schedule.py"], cwd=self.project_root, check=False, capture_output=True)
            self.refresh_views()
            QMessageBox.information(self, "Settings saved", "Settings were saved and the schedules were refreshed.")
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", str(exc))

    def test_email_settings(self, values: dict[str, Any]) -> None:
        try:
            self.data_service.test_email(values)
            QMessageBox.information(self, "Email sent", "Test email sent successfully.")
        except Exception as exc:
            QMessageBox.warning(self, "Email failed", str(exc))

    def retrain_model(self) -> None:
        QMessageBox.information(self, "Retraining started", "Model retraining is running in the background.")

        def _work() -> None:
            subprocess.run([os.sys.executable, "main.py", "--train", "--universe", "full"], cwd=self.project_root, check=False)
            self.refresh_requested.emit()

        threading.Thread(target=_work, daemon=True).start()

    def clear_cache(self) -> None:
        cache_root = self.project_root / "stock_predictor" / "artifacts"
        removed = []
        for suffix in ("cache.sqlite3", "cache.sqlite3-shm", "cache.sqlite3-wal"):
            path = cache_root / suffix
            if path.exists():
                path.unlink()
                removed.append(path.name)
        QMessageBox.information(self, "Cache cleared", "Removed: " + (", ".join(removed) or "nothing to clear"))

    def view_logs(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str((self.project_root / "logs").resolve())))

    def _show_notification(self, title: str, message: str) -> None:
        if self.tray is not None and self.tray.isVisible():
            self.tray.showMessage(title, message)

    def _set_dock_badge(self, count: int) -> None:
        try:
            from AppKit import NSApplication  # type: ignore

            app = NSApplication.sharedApplication()
            app.dockTile().setBadgeLabel_(str(count) if count else "")
        except Exception:
            pass

    @staticmethod
    def _state_to_dict(state: AppState) -> dict[str, Any]:
        return {
            "payload": state.payload,
            "single_analysis": state.single_analysis,
            "single_analysis_history": state.single_analysis_history,
            "model_metadata": state.model_metadata,
            "rolling_summary": state.rolling_summary,
            "weekly_rows": state.weekly_rows,
            "history_rows": state.history_rows,
            "cumulative_rows": state.cumulative_rows,
            "history_weeks": state.history_weeks,
            "history_details": state.history_details,
            "breakdowns": state.breakdowns,
            "settings": state.settings,
            "last_scan_text": state.last_scan_text,
            "top_pick_text": state.top_pick_text,
        }
