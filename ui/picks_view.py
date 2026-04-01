"""This week's picks screen for the native Stock Predictor app."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QFrame, QHBoxLayout, QScrollArea, QVBoxLayout, QWidget, QPushButton

from ui.widgets import InfoPill, PickCard, SectionLabel, apply_card_shadow, css_color, is_dark_mode, style_secondary_button, theme_colors


class PicksView(QWidget):
    """Scrollable list of weekly pick cards."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        self.header_card = QFrame()
        header_layout = QHBoxLayout(self.header_card)
        header_layout.setContentsMargins(16, 12, 16, 12)
        header_layout.setSpacing(8)
        self.regime_badge = InfoPill("RISK OFF", "bad")
        self.vix_badge = InfoPill("VIX --", "neutral")
        self.spy_badge = InfoPill("SPY --", "neutral")
        self.header_title = SectionLabel("This Week's Picks")
        header_layout.addWidget(self.header_title)
        header_layout.addStretch(1)
        header_layout.addWidget(self.regime_badge)
        header_layout.addWidget(self.vix_badge)
        header_layout.addWidget(self.spy_badge)
        root.addWidget(self.header_card)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setStyleSheet("background: transparent;")
        self.content = QWidget()
        self.cards_layout = QVBoxLayout(self.content)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(12)
        self.cards_layout.addStretch(1)
        self.scroll_area.setWidget(self.content)
        root.addWidget(self.scroll_area, 1)

        footer = QHBoxLayout()
        footer.addStretch(1)
        self.copy_button = QPushButton("Copy picks to clipboard")
        style_secondary_button(self.copy_button)
        self.copy_button.clicked.connect(self._copy_cards)
        footer.addWidget(self.copy_button)
        root.addLayout(footer)

        self._current_cards: list[dict[str, Any]] = []
        self.apply_theme()

    def apply_theme(self) -> None:
        colors = theme_colors(self)
        self.setStyleSheet(f"background: {css_color(colors['window'])};")
        self.header_card.setStyleSheet(
            f"background: {css_color(colors['base'])}; border: 1px solid {css_color(colors['border'])}; border-radius: 12px;"
        )
        apply_card_shadow(self.header_card, enabled=not is_dark_mode(self.header_card))

    def changeEvent(self, event) -> None:  # type: ignore[override]
        super().changeEvent(event)

    def update_data(self, state: dict[str, Any]) -> None:
        payload = state.get("payload", {})
        official = list(payload.get("selected", []))
        candidates = official or list(payload.get("display_candidates", []))
        self._current_cards = candidates

        vix = float(payload.get("scan_summary", {}).get("vix", 0.0))
        spy = float(payload.get("scan_summary", {}).get("spy_week_return", 0.0))
        regime = str(payload.get("regime_label") or payload.get("scan_summary", {}).get("regime", "Neutral")).upper()
        tone = "good" if vix < 20 else "warning" if vix < 25 else "bad"
        self.regime_badge.set_pill(f"● {regime}", tone)
        self.vix_badge.set_pill(f"VIX {vix:.1f}", tone)
        self.spy_badge.set_pill(f"SPY {spy:+.1f}%", "neutral")
        self.header_title.setText("THIS WEEK'S PICKS" if official else "WATCHLIST")

        while self.cards_layout.count() > 1:
            item = self.cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for candidate in candidates:
            card = PickCard(candidate)
            self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)

    def _copy_cards(self) -> None:
        if not self._current_cards:
            return
        lines = []
        for candidate in self._current_cards:
            lines.extend(
                [
                    (
                        f"{candidate.get('ticker')} | "
                        f"{candidate.get('company_name')} | "
                        f"Entry {candidate.get('current_price', 0):.2f} | "
                        f"Target {candidate.get('targets', {}).get('tp2', 0):.2f} | "
                        f"Stop {candidate.get('stop_loss', 0):.2f} | "
                        f"Score {candidate.get('final_score', 0):.1f}"
                    ),
                    f"Why: {candidate.get('ai_explanation', '')}",
                    "",
                ]
            )
        QGuiApplication.clipboard().setText("\n".join(lines).strip())
