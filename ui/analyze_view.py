"""Single-stock analysis screen for the native Stock Predictor app."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ui.widgets import (
    AnalyzeTickerWorker,
    InfoPill,
    PickCard,
    SectionLabel,
    apple_font,
    apply_card_shadow,
    css_color,
    is_dark_mode,
    style_primary_button,
    style_secondary_button,
    theme_colors,
)


class AnalyzeView(QWidget):
    """Interactive single-ticker analysis view."""

    def __init__(self, project_root: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project_root = project_root
        self.worker: AnalyzeTickerWorker | None = None
        self._universe_mode = "full"
        self._last_payload: dict[str, Any] = {}
        self._history_payloads: list[dict[str, Any]] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        header = QVBoxLayout()
        header.setSpacing(4)
        self.title = QLabel("Analyze a Stock")
        self.title.setFont(apple_font("display", 28, QFont.Weight.DemiBold))
        self.subtitle = QLabel("Run the same scoring stack on one ticker and see whether it would qualify this week.")
        self.subtitle.setFont(apple_font("text", 14, QFont.Weight.Normal))
        self.subtitle.setWordWrap(True)
        header.addWidget(self.title)
        header.addWidget(self.subtitle)
        root.addLayout(header)

        self.controls_card = QFrame()
        controls_layout = QVBoxLayout(self.controls_card)
        controls_layout.setContentsMargins(20, 20, 20, 20)
        controls_layout.setSpacing(12)
        controls_layout.addWidget(SectionLabel("Analyze Ticker"))

        input_row = QHBoxLayout()
        input_row.setSpacing(10)
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("Enter a ticker, e.g. AAPL")
        self.ticker_input.setFont(apple_font("text", 15, QFont.Weight.Medium))
        self.ticker_input.setFixedWidth(260)
        self.ticker_input.returnPressed.connect(self.run_analysis)
        self.analyze_button = QPushButton("Analyze")
        style_primary_button(self.analyze_button)
        self.analyze_button.setFixedWidth(120)
        self.analyze_button.clicked.connect(self.run_analysis)
        input_row.addWidget(self.ticker_input)
        input_row.addWidget(self.analyze_button)
        input_row.addStretch(1)
        controls_layout.addLayout(input_row)

        self.status_label = QLabel("Type a ticker to run a full single-stock analysis.")
        self.status_label.setFont(apple_font("text", 13, QFont.Weight.Normal))
        self.status_label.setWordWrap(True)
        controls_layout.addWidget(self.status_label)

        pills_row = QHBoxLayout()
        pills_row.setSpacing(8)
        self.eligibility_pill = InfoPill("Awaiting analysis", "neutral")
        self.regime_pill = InfoPill("Regime --", "neutral")
        self.threshold_pill = InfoPill("Threshold --", "neutral")
        self.model_pill = InfoPill("Model --", "neutral")
        self.universe_pill = InfoPill("Universe full", "neutral")
        pills_row.addWidget(self.eligibility_pill)
        pills_row.addWidget(self.regime_pill)
        pills_row.addWidget(self.threshold_pill)
        pills_row.addWidget(self.model_pill)
        pills_row.addWidget(self.universe_pill)
        pills_row.addStretch(1)
        controls_layout.addLayout(pills_row)
        root.addWidget(self.controls_card)

        self.history_card = QFrame()
        history_layout = QVBoxLayout(self.history_card)
        history_layout.setContentsMargins(20, 20, 20, 20)
        history_layout.setSpacing(10)
        history_layout.addWidget(SectionLabel("Recent Analyses"))
        self.history_grid = QGridLayout()
        self.history_grid.setHorizontalSpacing(8)
        self.history_grid.setVerticalSpacing(8)
        history_layout.addLayout(self.history_grid)
        root.addWidget(self.history_card)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setStyleSheet("background: transparent;")
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(12)
        self.empty_card = QFrame()
        empty_layout = QVBoxLayout(self.empty_card)
        empty_layout.setContentsMargins(20, 20, 20, 20)
        empty_layout.setSpacing(10)
        empty_layout.addWidget(SectionLabel("Latest Analysis"))
        self.empty_text = QLabel("No single-stock analysis has been run yet.")
        self.empty_text.setFont(apple_font("text", 14, QFont.Weight.Normal))
        self.empty_text.setWordWrap(True)
        empty_layout.addWidget(self.empty_text)
        self.content_layout.addWidget(self.empty_card)
        self.content_layout.addStretch(1)
        self.scroll_area.setWidget(self.content)
        root.addWidget(self.scroll_area, 1)

        self.apply_theme()

    def apply_theme(self) -> None:
        colors = theme_colors(self)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(f"{self.__class__.__name__} {{ background-color: {css_color(colors['window'])}; }}")
        self.title.setStyleSheet(f"color: {css_color(colors['text'])};")
        self.subtitle.setStyleSheet(f"color: {css_color(colors['text_secondary'])};")
        base_bg = css_color(colors["base"])
        border_c = css_color(colors["border"])
        card_style = (
            f"QFrame {{ background: {base_bg}; border: 1px solid {border_c}; border-radius: 8px; }}\n"
            f"QLabel {{ background: transparent; border: none; }}\n"
            f"QWidget {{ background: transparent; border: none; }}"
        )
        for panel in (self.controls_card, self.history_card, self.empty_card):
            panel.setStyleSheet(card_style)
            apply_card_shadow(panel, enabled=not is_dark_mode(panel))
        self.ticker_input.setStyleSheet(
            f"""
            QLineEdit {{
                background: {css_color(colors['base'])};
                color: {css_color(colors['text'])};
                border: 1px solid rgba(255, 255, 255, 0.07);
                border-radius: 8px;
                padding: 8px 10px;
                min-height: 34px;
            }}
            QLineEdit:focus {{
                border: 1px solid #007AFF;
            }}
            """
        )
        self.status_label.setStyleSheet(f"color: {css_color(colors['text_secondary'])};")
        self.empty_text.setStyleSheet(f"color: {css_color(colors['text_secondary'])};")
        style_primary_button(self.analyze_button)

    def update_data(self, state: dict[str, Any]) -> None:
        self._universe_mode = str(state.get("settings", {}).get("universe", "full"))
        self.universe_pill.set_pill(f"Universe {self._universe_mode}", "neutral")
        model_meta = state.get("model_metadata", {})
        stack = str(model_meta.get("model_stack", "XGBoost"))
        auc = float(model_meta.get("auc", 0.0) or 0.0)
        profile = str(model_meta.get("selected_profile", "")).strip()
        model_text = f"{stack} · AUC {auc:.3f}"
        if profile:
            model_text = f"{model_text} · {profile}"
        self.model_pill.set_pill(model_text, "neutral")
        self._history_payloads = list(state.get("single_analysis_history", []))
        self._render_history_buttons()
        payload = state.get("single_analysis", {})
        if payload:
            self._render_payload(payload)

    def run_analysis(self) -> None:
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            self.status_label.setText("Enter a valid ticker symbol first.")
            return
        if self.worker is not None and self.worker.isRunning():
            return
        self.ticker_input.setText(ticker)
        self.analyze_button.setEnabled(False)
        self.status_label.setText(f"Analyzing {ticker} with the live scoring stack…")
        self.worker = AnalyzeTickerWorker(self.project_root, ticker, universe_mode=self._universe_mode)
        self.worker.setParent(None)  # Prevent app crash if view closes while running
        self.worker.progress_changed.connect(self._handle_progress)
        self.worker.analysis_finished.connect(self._handle_finished)
        self.worker.analysis_failed.connect(self._handle_failed)
        self.worker.start()

    def _render_history_buttons(self) -> None:
        while self.history_grid.count():
            item = self.history_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if not self._history_payloads:
            empty = QLabel("Your recent ticker analyses will appear here.")
            empty.setFont(apple_font("text", 13, QFont.Weight.Normal))
            empty.setWordWrap(True)
            empty.setStyleSheet(f"color: {css_color(theme_colors(self.history_card)['text_secondary'])};")
            self.history_grid.addWidget(empty, 0, 0, 1, 2)
            return
        for index, payload in enumerate(self._history_payloads[:8]):
            ticker = str(payload.get("ticker", "--"))
            candidate = payload.get("candidate") or {}
            score = float(candidate.get("final_score", 0.0) or 0.0)
            button = QPushButton(f"{ticker}   {score:.1f}")
            style_secondary_button(button)
            button.setMinimumHeight(44)
            button.clicked.connect(lambda checked=False, p=payload: self._render_payload(p))
            self.history_grid.addWidget(button, index // 4, index % 4)

    def _handle_progress(self, message: str) -> None:
        if message:
            self.status_label.setText(message)

    def _handle_finished(self, payload: dict[str, Any]) -> None:
        self.analyze_button.setEnabled(True)
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
        self._history_payloads = [payload, *[item for item in self._history_payloads if item.get("ticker") != payload.get("ticker")]]
        self._render_history_buttons()
        self._render_payload(payload)

    def _handle_failed(self, message: str) -> None:
        self.analyze_button.setEnabled(True)
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
        self.status_label.setText(message)

    def _render_payload(self, payload: dict[str, Any]) -> None:
        self._last_payload = payload
        while self.content_layout.count() > 1:
            item = self.content_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        candidate = payload.get("candidate") or {}
        eligible = bool(payload.get("officially_eligible"))
        reasons = [str(reason) for reason in payload.get("eligibility_reasons", []) if str(reason).strip()]
        scan_summary = payload.get("scan_summary", {})
        regime = str(payload.get("regime_label") or scan_summary.get("regime") or "neutral").upper()
        threshold_used = float(payload.get("threshold_used", 0.0) or 0.0)
        self.status_label.setText(payload.get("macro_summary") or "Analysis complete.")
        self.eligibility_pill.set_pill(
            "Official Pick This Week" if eligible else "Watchlist / Not Eligible",
            "good" if eligible else "warning",
        )
        self.regime_pill.set_pill(f"Regime {regime}", "neutral")
        self.threshold_pill.set_pill(f"Threshold {threshold_used:.1f}", "neutral")

        if candidate:
            card = PickCard(candidate)
            self.content_layout.insertWidget(0, card)
            notes_card = QFrame()
            notes_layout = QVBoxLayout(notes_card)
            notes_layout.setContentsMargins(20, 20, 20, 20)
            notes_layout.setSpacing(8)
            notes_layout.addWidget(SectionLabel("Eligibility"))
            body = QLabel(
                "This ticker would qualify for this week's official picks."
                if eligible
                else "This ticker does not currently qualify for the official list."
            )
            body.setWordWrap(True)
            body.setFont(apple_font("text", 14, QFont.Weight.Normal))
            notes_layout.addWidget(body)
            if reasons:
                for reason in reasons:
                    note = QLabel(f"• {reason}")
                    note.setWordWrap(True)
                    note.setFont(apple_font("text", 13, QFont.Weight.Normal))
                    notes_layout.addWidget(note)
            self.content_layout.insertWidget(1, notes_card)
            colors = theme_colors(notes_card)
            notes_card.setStyleSheet(
                f"background: {css_color(colors['base'])}; "
                f"border: 1px solid rgba(255, 255, 255, 0.07); "
                "border-radius: 8px;"
            )
            apply_card_shadow(notes_card, enabled=not is_dark_mode(notes_card))
            body.setStyleSheet(f"color: {css_color(colors['text'])};")
            for index in range(2, notes_layout.count()):
                item = notes_layout.itemAt(index)
                widget = item.widget()
                if widget is not None:
                    widget.setStyleSheet(f"color: {css_color(colors['text_secondary'])};")
        else:
            self.empty_text.setText("Analysis ran, but no candidate payload was returned.")
