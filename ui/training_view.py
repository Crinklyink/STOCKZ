"""Model training workspace for the native Stock Predictor app."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.widgets import (
    COLORS,
    InfoPill,
    SectionLabel,
    StatCard,
    apple_font,
    apply_card_shadow,
    css_color,
    is_dark_mode,
    style_primary_button,
    style_secondary_button,
    theme_colors,
)


PROGRESS_RE = re.compile(r"^\[(training|backtest)\]\s+(.*)$")


class TrainingJobWorker(QThread):
    """Run model training or backtesting while streaming CLI output."""

    progress_changed = Signal(dict)
    line_received = Signal(str)
    job_finished = Signal(dict)
    job_failed = Signal(str)

    def __init__(
        self,
        project_root: Path,
        *,
        job: str,
        universe_mode: str,
        fresh: bool,
    ) -> None:
        super().__init__()
        self.project_root = project_root
        self.job = job
        self.universe_mode = universe_mode
        self.fresh = fresh
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:  # type: ignore[override]
        command = [sys.executable, "main.py", "--universe", self.universe_mode]
        command.append("--train" if self.job == "train" else "--backtest-adaptive")
        if self.fresh:
            command.append("--fresh")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        label = "Training model" if self.job == "train" else "Running adaptive backtest"
        self.progress_changed.emit({"percent": 4, "stage": label, "detail": "Starting subprocess"})

        try:
            process = subprocess.Popen(
                command,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
        except Exception as exc:
            self.job_failed.emit(f"Could not start job: {exc}")
            return

        percent = 8
        started = time.perf_counter()
        stage_weights = [
            ("Resolving", 12),
            ("Fetching historical", 26),
            ("Fetching sector", 34),
            ("Fetching benchmark", 42),
            ("Calculating", 52),
            ("Training", 68),
            ("Running", 76),
            ("Saving", 92),
            ("complete", 100),
        ]

        output: list[str] = []
        try:
            stream = process.stdout
            if stream is not None:
                while True:
                    if self._cancelled and process.poll() is None:
                        process.terminate()
                        self.job_failed.emit("Job canceled.")
                        return
                    line = stream.readline()
                    if line == "" and process.poll() is not None:
                        break
                    if not line:
                        elapsed = int(time.perf_counter() - started)
                        if percent < 88 and elapsed > 0:
                            percent = min(88, percent + 1)
                            self.progress_changed.emit(
                                {"percent": percent, "stage": label, "detail": "Working through data pipeline"}
                            )
                        self.msleep(350)
                        continue
                    clean = line.rstrip()
                    output.append(clean)
                    self.line_received.emit(clean)
                    match = PROGRESS_RE.match(clean)
                    if match:
                        _, detail = match.groups()
                        lowered = detail.lower()
                        for needle, value in stage_weights:
                            if needle.lower() in lowered:
                                percent = max(percent, value)
                                break
                        self.progress_changed.emit({"percent": percent, "stage": label, "detail": detail})
        finally:
            returncode = process.wait()

        if returncode != 0:
            self.job_failed.emit(f"Job failed with exit code {returncode}.")
            return
        self.progress_changed.emit({"percent": 100, "stage": "Complete", "detail": "Refreshing model metrics"})
        self.job_finished.emit({"job": self.job, "output": "\n".join(output)})


class TrainingView(QWidget):
    """Dedicated tab for model training, rolling backtests, and automation controls."""

    auto_training_changed = Signal(bool)
    logs_requested = Signal()
    job_completed = Signal()

    def __init__(self, project_root: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project_root = project_root
        self.worker: TrainingJobWorker | None = None
        self._started_at = 0.0

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 28, 28, 28)
        root.setSpacing(16)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        self.title = QLabel("Training")
        self.title.setFont(apple_font("display", 28, QFont.Weight.Bold))
        self.subtitle = QLabel("Train the model, run rolling walk-forward backtests, and keep the weekly auto-trainer visible.")
        self.subtitle.setFont(apple_font("text", 14))
        self.subtitle.setWordWrap(True)
        title_box.addWidget(self.title)
        title_box.addWidget(self.subtitle)
        header.addLayout(title_box, 1)
        self.status_pill = InfoPill("Idle", "neutral")
        header.addWidget(self.status_pill, alignment=Qt.AlignmentFlag.AlignTop)
        root.addLayout(header)

        self.stat_cards = {
            "stack": StatCard("Stack", "--", "active model"),
            "auc": StatCard("AUC", "--", "walk-forward"),
            "samples": StatCard("Samples", "--", "training rows"),
            "trained": StatCard("Trained", "--", "last run"),
        }
        stats = QGridLayout()
        stats.setHorizontalSpacing(12)
        stats.setVerticalSpacing(12)
        for index, card in enumerate(self.stat_cards.values()):
            stats.addWidget(card, index // 4, index % 4)
        root.addLayout(stats)

        self.control_panel = QFrame()
        controls = QVBoxLayout(self.control_panel)
        controls.setContentsMargins(20, 20, 20, 20)
        controls.setSpacing(14)
        controls.addWidget(SectionLabel("Run"))
        run_row = QHBoxLayout()
        self.universe_combo = QComboBox()
        self.universe_combo.addItem("Mini - fast", "mini")
        self.universe_combo.addItem("Full - recommended", "full")
        self.universe_combo.addItem("US Market - large", "us_market")
        self.fresh_check = QCheckBox("Fresh data")
        self.auto_check = QCheckBox("Auto-train weekly")
        self.train_button = QPushButton("Train Model")
        self.backtest_button = QPushButton("Run Rolling Backtest")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        style_primary_button(self.train_button)
        style_secondary_button(self.backtest_button)
        style_secondary_button(self.cancel_button)
        self.train_button.clicked.connect(lambda: self._start_job("train"))
        self.backtest_button.clicked.connect(lambda: self._start_job("backtest"))
        self.cancel_button.clicked.connect(self._cancel_job)
        self.auto_check.toggled.connect(self.auto_training_changed.emit)
        for widget in (self.universe_combo, self.fresh_check, self.auto_check, self.train_button, self.backtest_button, self.cancel_button):
            run_row.addWidget(widget)
        run_row.addStretch(1)
        controls.addLayout(run_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(False)
        self.stage_label = QLabel("Ready")
        self.detail_label = QLabel("Choose a universe and start training or a rolling backtest.")
        self.elapsed_label = QLabel("Elapsed 0s")
        self.stage_label.setFont(apple_font("text", 14, QFont.Weight.DemiBold))
        self.detail_label.setFont(apple_font("text", 13))
        self.elapsed_label.setFont(apple_font("mono", 12))
        controls.addWidget(self.progress_bar)
        controls.addWidget(self.stage_label)
        controls.addWidget(self.detail_label)
        controls.addWidget(self.elapsed_label)
        root.addWidget(self.control_panel)

        self.log_panel = QFrame()
        log_layout = QVBoxLayout(self.log_panel)
        log_layout.setContentsMargins(20, 20, 20, 20)
        log_layout.setSpacing(10)
        log_header = QHBoxLayout()
        log_header.addWidget(SectionLabel("Live Output"))
        log_header.addStretch(1)
        self.logs_button = QPushButton("Open Logs")
        style_secondary_button(self.logs_button)
        self.logs_button.clicked.connect(self.logs_requested.emit)
        log_header.addWidget(self.logs_button)
        log_layout.addLayout(log_header)
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMaximumBlockCount(800)
        log_layout.addWidget(self.output, 1)
        root.addWidget(self.log_panel, 1)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.apply_theme()

    def apply_theme(self) -> None:
        colors = theme_colors(self)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(f"{self.__class__.__name__} {{ background-color: {css_color(colors['window'])}; }}")
        self.title.setStyleSheet(f"color: {css_color(colors['text'])};")
        self.subtitle.setStyleSheet(f"color: {css_color(colors['text_secondary'])};")
        panel_style = (
            f"QFrame {{ background: {css_color(colors['base'])}; border: 1px solid {css_color(colors['border'])}; border-radius: 12px; }}"
            " QLabel, QWidget { background: transparent; border: none; }"
        )
        for panel in (self.control_panel, self.log_panel):
            panel.setStyleSheet(panel_style)
            apply_card_shadow(panel, enabled=not is_dark_mode(panel))
        for label in (self.detail_label, self.elapsed_label):
            label.setStyleSheet(f"color: {css_color(colors['text_secondary'])};")
        self.stage_label.setStyleSheet(f"color: {css_color(colors['text'])};")
        self.output.setFont(apple_font("mono", 12))
        self.output.setStyleSheet(
            f"""
            QPlainTextEdit {{
                background: {css_color(colors['alt_base'])};
                color: {css_color(colors['text'])};
                border: 1px solid {css_color(colors['border'])};
                border-radius: 10px;
                padding: 10px;
            }}
            """
        )
        self.progress_bar.setStyleSheet(
            f"""
            QProgressBar {{
                background: {css_color(colors['alt_base'])};
                border: 0;
                border-radius: 4px;
                height: 8px;
            }}
            QProgressBar::chunk {{
                background: {COLORS['blue']};
                border-radius: 4px;
            }}
            """
        )
        self.universe_combo.setStyleSheet(
            f"background: {css_color(colors['alt_base'])}; color: {css_color(colors['text'])}; "
            f"border: 1px solid {css_color(colors['border'])}; border-radius: 8px; padding: 8px;"
        )
        for check in (self.fresh_check, self.auto_check):
            check.setFont(apple_font("text", 13))
            check.setStyleSheet(f"color: {css_color(colors['text'])};")

    def update_data(self, state: dict[str, Any]) -> None:
        settings = state.get("settings", {})
        universe = str(settings.get("universe", "full"))
        index = self.universe_combo.findData(universe)
        if index >= 0:
            self.universe_combo.setCurrentIndex(index)
        self.auto_check.blockSignals(True)
        self.auto_check.setChecked(bool(settings.get("retrain_weekly", True)))
        self.auto_check.blockSignals(False)

        meta = state.get("model_metadata", {})
        stack = str(meta.get("model_stack", "Adaptive Ensemble"))
        profile = str(meta.get("profile_label", "--"))
        auc = float(meta.get("auc", 0.0) or 0.0)
        samples = int(meta.get("training_samples", 0) or 0)
        trained_at = str(meta.get("trained_at", "--"))[:16].replace("T", " ")
        self.stat_cards["stack"].set_data("Stack", stack, profile, "neutral")
        self.stat_cards["auc"].set_data("AUC", f"{auc:.3f}", "walk-forward", "good" if auc >= 0.6 else "warning")
        self.stat_cards["samples"].set_data("Samples", f"{samples:,}", "training rows", "good" if samples else "warning")
        self.stat_cards["trained"].set_data("Trained", trained_at or "--", "last artifact", "neutral")

    def _start_job(self, job: str) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        self.output.clear()
        self.progress_bar.setValue(0)
        self._started_at = time.perf_counter()
        self.timer.start(500)
        self.status_pill.set_pill("Running", "blue")
        self.train_button.setEnabled(False)
        self.backtest_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.worker = TrainingJobWorker(
            self.project_root,
            job=job,
            universe_mode=str(self.universe_combo.currentData()),
            fresh=self.fresh_check.isChecked(),
        )
        self.worker.progress_changed.connect(self._handle_progress)
        self.worker.line_received.connect(self._append_line)
        self.worker.job_finished.connect(self._handle_finished)
        self.worker.job_failed.connect(self._handle_failed)
        self.worker.start()

    def _cancel_job(self) -> None:
        if self.worker is not None:
            self.worker.cancel()

    def _handle_progress(self, payload: dict[str, Any]) -> None:
        self.progress_bar.setValue(int(payload.get("percent", 0)))
        self.stage_label.setText(str(payload.get("stage", "Working")))
        self.detail_label.setText(str(payload.get("detail", "")))

    def _append_line(self, line: str) -> None:
        if line.strip():
            self.output.appendPlainText(line)

    def _handle_finished(self, payload: dict[str, Any]) -> None:
        self._finish_ui("Complete", "good")
        self.progress_bar.setValue(100)
        self.stage_label.setText("Complete")
        self.detail_label.setText("Model artifacts and backtest data refreshed.")
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
        self.job_completed.emit()

    def _handle_failed(self, message: str) -> None:
        self._finish_ui("Failed", "bad")
        self.detail_label.setText(message)
        self.output.appendPlainText(message)
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None

    def _finish_ui(self, text: str, tone: str) -> None:
        self.timer.stop()
        self.status_pill.set_pill(text, tone)
        self.train_button.setEnabled(True)
        self.backtest_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

    def _tick(self) -> None:
        elapsed = int(time.perf_counter() - self._started_at)
        self.elapsed_label.setText(f"Elapsed {elapsed}s")

