"""Reusable widgets for the native Stock Predictor app."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from PySide6.QtCore import Property, QEasingCurve, QEvent, QPointF, QPropertyAnimation, QRectF, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase, QGuiApplication, QLinearGradient, QPainter, QPainterPath, QPalette, QPen
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QToolTip,
    QVBoxLayout,
    QWidget,
)


COLORS = {
    "green": "#34C759",
    "red": "#FF3B30",
    "orange": "#FF9500",
    "blue": "#007AFF",
    "indigo": "#5856D6",
    "teal": "#5AC8FA",
    "success": "#34C759",
    "danger": "#FF3B30",
    "warning": "#FF9500",
    "accent": "#007AFF",
    "accent_soft": "#5AC8FA",
    "good": "#34C759",
    "bad": "#FF3B30",
}

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
TQDM_PROGRESS_RE = re.compile(
    r"(Scoring tickers|Final scoring):\s*(\d+)%.*?(\d+)/(\d+).*?(?:,\s*([A-Z0-9\-\.^]+))?\]?$"
)
ANALYSIS_NOISE_PATTERNS = (
    "RequestsDependencyWarning",
    "urllib3 (",
    "HTTP Request:",
    "Warning: You are sending unauthenticated requests to the HF Hub.",
    "BertForSequenceClassification LOAD REPORT",
    "Loading weights:",
    "Key                          | Status",
    "bert.embeddings.position_ids",
    "Notes:",
    "- UNEXPECTED",
)


def apply_shadow(widget: QWidget, color: str = COLORS["accent"], blur: int = 30, alpha: int = 60) -> None:
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(blur)
    shadow.setOffset(0, 8)
    qcolor = QColor(color)
    qcolor.setAlpha(alpha)
    shadow.setColor(qcolor)
    widget.setGraphicsEffect(shadow)


def format_money(value: float) -> str:
    return f"${value:,.2f}"


def format_percent(value: float) -> str:
    return f"{value:+.1f}%"


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def tone_color(value: float, *, good_cutoff: float, neutral_cutoff: float) -> str:
    if value >= good_cutoff:
        return COLORS["success"]
    if value >= neutral_cutoff:
        return COLORS["warning"]
    return COLORS["danger"]


def with_alpha(color: QColor | str, alpha: float) -> QColor:
    qcolor = QColor(color)
    qcolor.setAlphaF(max(0.0, min(1.0, alpha)))
    return qcolor


def css_color(color: QColor | str, alpha: float | None = None) -> str:
    qcolor = QColor(color)
    if alpha is not None:
        qcolor.setAlphaF(max(0.0, min(1.0, alpha)))
    return qcolor.name(QColor.NameFormat.HexArgb)


def should_ignore_analysis_output_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    return any(pattern in stripped for pattern in ANALYSIS_NOISE_PATTERNS)


def _available_families() -> set[str]:
    try:
        return set(QFontDatabase.families())
    except Exception:
        return set()


def apple_font(role: str, pixel_size: int, weight: int | QFont.Weight = QFont.Weight.Normal, italic: bool = False) -> QFont:
    available = _available_families()
    choices = {
        "display": ["SF Pro Display", ".AppleSystemUIFont"],
        "text": ["SF Pro Text", ".AppleSystemUIFont"],
        "rounded": ["SF Pro Rounded", "SF Pro Display", ".AppleSystemUIFont"],
        "mono": ["SF Mono", ".AppleSystemUIFont"],
    }.get(role, [".AppleSystemUIFont"])
    family = next((item for item in choices if item in available), ".AppleSystemUIFont")
    font = QFont(family)
    font.setPixelSize(pixel_size)
    font.setWeight(weight)
    font.setItalic(italic)
    return font


def apply_letter_spacing(font: QFont, spacing: float = 0.8) -> QFont:
    font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    return font


def theme_colors(widget: QWidget | None = None) -> dict[str, QColor]:
    app = QApplication.instance()
    palette = widget.palette() if widget is not None else (app.palette() if app is not None else QPalette())
    window = palette.color(QPalette.ColorRole.Window)
    base = palette.color(QPalette.ColorRole.Base)
    alt_base = palette.color(QPalette.ColorRole.AlternateBase)
    text = palette.color(QPalette.ColorRole.WindowText)
    dark = window.lightness() < 128
    return {
        "is_dark": QColor(0, 0, 0, 255 if dark else 0),
        "window": window,
        "base": base,
        "alt_base": alt_base,
        "text": text,
        "text_secondary": with_alpha(text, 0.6),
        "text_tertiary": with_alpha(text, 0.4),
        "border": QColor(255, 255, 255, 20) if dark else QColor(0, 0, 0, 15),
        "separator": QColor(255, 255, 255, 20) if dark else QColor(0, 0, 0, 20),
        "hover": with_alpha(text, 0.06 if dark else 0.04),
        "selection": with_alpha(COLORS["blue"], 0.15),
        "button_hover": with_alpha(COLORS["blue"], 0.10),
        "badge_neutral": with_alpha(text, 0.08 if dark else 0.05),
        "sheet": with_alpha(base, 0.96),
        "shadow": QColor(0, 0, 0, 20),
    }


def is_dark_mode(widget: QWidget | None = None) -> bool:
    return theme_colors(widget)["window"].lightness() < 128


def apply_card_shadow(widget: QWidget, *, enabled: bool) -> None:
    if not enabled:
        widget.setGraphicsEffect(None)
        return
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(20)
    shadow.setOffset(0, 2)
    shadow.setColor(QColor(0, 0, 0, 20))
    widget.setGraphicsEffect(shadow)


def secondary_text_css(widget: QWidget, opacity: float = 0.6) -> str:
    return css_color(theme_colors(widget)["text"], opacity)


def style_secondary_button(button: QPushButton) -> None:
    colors = theme_colors(button)
    button.setFont(apple_font("text", 14, QFont.Weight.DemiBold))
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.setStyleSheet(
        f"""
        QPushButton {{
            background: transparent;
            color: {COLORS["blue"]};
            border: 1px solid {css_color(colors["border"])};
            border-radius: 10px;
            padding: 8px 14px;
        }}
        QPushButton:hover {{
            background: {css_color(colors["button_hover"])};
        }}
        """
    )


def style_primary_button(button: QPushButton) -> None:
    button.setFont(apple_font("text", 15, QFont.Weight.DemiBold))
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.setStyleSheet(
        f"""
        QPushButton {{
            background: {COLORS["blue"]};
            color: white;
            border: 0;
            border-radius: 10px;
            padding: 8px 16px;
        }}
        QPushButton:hover {{
            background: {css_color(COLORS["blue"], 0.88)};
        }}
        """
    )


class SidebarButton(QPushButton):
    """Navigation button used in the app sidebar."""

    def __init__(self, label: str, parent: QWidget | None = None) -> None:
        super().__init__(label, parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(36)
        self.setFont(apple_font("text", 14, QFont.Weight.DemiBold))
        self.apply_theme()

    def apply_theme(self) -> None:
        colors = theme_colors(self)
        self.setStyleSheet(
            f"""
            QPushButton {{
                border: 0;
                border-radius: 10px;
                color: {css_color(colors["text_secondary"])};
                background: transparent;
                text-align: left;
                padding: 0 12px;
            }}
            QPushButton:hover {{
                color: {css_color(colors["text"])};
                background: {css_color(colors["hover"])};
            }}
            QPushButton:checked {{
                color: {COLORS["blue"]};
                background: {css_color(colors["selection"])};
            }}
            """
        )

    def changeEvent(self, event) -> None:  # type: ignore[override]
        super().changeEvent(event)


class StatCard(QFrame):
    """Top-level stat card widget."""

    def __init__(self, title: str, value: str = "--", subtitle: str = "", tone: str = "neutral", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tone = tone
        self._title = QLabel(title.upper())
        self._value = QLabel(value)
        self._subtitle = QLabel(subtitle)

        self._title.setFont(apply_letter_spacing(apple_font("text", 11, QFont.Weight.DemiBold), 0.5))
        self._value.setFont(apple_font("rounded", 34, QFont.Weight.Light))
        self._subtitle.setFont(apple_font("text", 12, QFont.Weight.Normal))
        self._subtitle.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(8)
        layout.addWidget(self._title)
        layout.addWidget(self._value)
        layout.addWidget(self._subtitle)
        layout.addStretch(1)
        self.setObjectName("statCard")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(124)
        self.set_tone(tone)

    def set_data(self, title: str, value: str, subtitle: str, tone: str = "neutral") -> None:
        self._tone = tone
        self._title.setText(title.upper())
        self._value.setText(value)
        self._subtitle.setText(subtitle)
        self.set_tone(tone)

    def set_tone(self, tone: str) -> None:
        colors = theme_colors(self)
        accent = {
            "good": COLORS["green"],
            "warning": COLORS["orange"],
            "bad": COLORS["red"],
            "neutral": css_color(colors["text"]),
        }.get(tone, css_color(colors["text"]))
        self._title.setStyleSheet(f"color: {css_color(colors['text_secondary'])};")
        self._value.setStyleSheet(f"color: {accent};")
        self._subtitle.setStyleSheet(f"color: {css_color(colors['text_secondary'])};")
        self.setStyleSheet(
            f"""
            QFrame#statCard {{
                background: {css_color(colors["base"])};
                border: 1px solid {css_color(colors["border"])};
                border-radius: 12px;
            }}
            """
        )
        apply_card_shadow(self, enabled=not is_dark_mode(self))

    def changeEvent(self, event) -> None:  # type: ignore[override]
        super().changeEvent(event)


class SignalMeter(QWidget):
    """A compact meter used inside pick cards."""

    def __init__(self, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.label = QLabel(label)
        self.value_label = QLabel("0")
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(3)
        self.label.setFont(apply_letter_spacing(apple_font("text", 11, QFont.Weight.DemiBold), 0.4))
        self.value_label.setFont(apple_font("rounded", 12, QFont.Weight.Medium))

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self.label)
        top.addStretch(1)
        top.addWidget(self.value_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(top)
        layout.addWidget(self.bar)
        self._value = 0.0
        self.apply_theme()

    def set_value(self, value: float, color: str | None = None) -> None:
        self._value = float(value)
        self.bar.setValue(int(clamp(value, 0, 100)))
        self.value_label.setText(f"{value:.0f}")
        self.apply_theme(color=color)

    def apply_theme(self, color: str | None = None) -> None:
        colors = theme_colors(self)
        chunk_color = color or tone_color(self._value, good_cutoff=70, neutral_cutoff=50)
        self.label.setStyleSheet(f"color: {css_color(colors['text_secondary'])};")
        self.value_label.setStyleSheet(f"color: {css_color(colors['text'])};")
        self.bar.setStyleSheet(
            f"""
            QProgressBar {{
                background: {css_color(colors["border"], 0.4)};
                border: 0;
                border-radius: 2px;
            }}
            QProgressBar::chunk {{
                background: {chunk_color};
                border-radius: 2px;
            }}
            """
        )

    def changeEvent(self, event) -> None:  # type: ignore[override]
        super().changeEvent(event)


class SignalDot(QLabel):
    """Rounded signal status dot."""

    def __init__(self, text: str, state: str = "neutral", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(28)
        self.set_state(state)

    def set_state(self, state: str) -> None:
        colors = theme_colors(self)
        color = {
            "good": COLORS["green"],
            "warning": COLORS["orange"],
            "bad": COLORS["red"],
            "neutral": css_color(colors["text_secondary"]),
        }.get(state, css_color(colors["text_secondary"]))
        background = QColor(color)
        background.setAlpha(45)
        self.setFont(apple_font("text", 12, QFont.Weight.Medium))
        self.setStyleSheet(
            f"""
            QLabel {{
                border-radius: 6px;
                padding: 4px 8px;
                color: {color};
                background: {background.name(QColor.NameFormat.HexArgb)};
            }}
            """
        )


class SectionLabel(QLabel):
    """Uppercase Apple-style section label."""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text.upper(), parent)
        self.setFont(apply_letter_spacing(apple_font("text", 11, QFont.Weight.DemiBold), 0.8))
        self.apply_theme()

    def apply_theme(self) -> None:
        self.setStyleSheet(f"color: {secondary_text_css(self, 0.5)};")

    def changeEvent(self, event) -> None:  # type: ignore[override]
        super().changeEvent(event)


class ScoreBadge(QLabel):
    """Rounded score badge with semantic color."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("0", parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumWidth(46)
        self.setFont(apple_font("rounded", 12, QFont.Weight.Medium))
        self._tone = "neutral"
        self.apply_theme()

    def set_score(self, score: float) -> None:
        self.setText(f"{score:.0f}")
        if score >= 70:
            self._tone = "good"
        elif score >= 54:
            self._tone = "warning"
        else:
            self._tone = "bad"
        self.apply_theme()

    def apply_theme(self) -> None:
        colors = theme_colors(self)
        tone_color_map = {
            "good": COLORS["green"],
            "warning": COLORS["orange"],
            "bad": COLORS["red"],
            "neutral": css_color(colors["text_secondary"]),
        }
        tone = QColor(tone_color_map[self._tone])
        bg = QColor(tone)
        bg.setAlpha(28)
        self.setStyleSheet(
            f"""
            QLabel {{
                min-height: 24px;
                padding: 0 8px;
                border-radius: 6px;
                color: {tone.name()};
                background: {bg.name(QColor.NameFormat.HexArgb)};
            }}
            """
        )

    def changeEvent(self, event) -> None:  # type: ignore[override]
        super().changeEvent(event)


class SignalDotsWidget(QWidget):
    """Five-dot confluence indicator."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._labels = [QLabel("●") for _ in range(5)]
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        for label in self._labels:
            label.setFont(apple_font("rounded", 10, QFont.Weight.Medium))
            layout.addWidget(label)
        self._count = 0
        self._tone = "neutral"
        self.apply_theme()

    def set_state(self, count: int, tone: str) -> None:
        self._count = max(0, min(5, count))
        self._tone = tone
        self.apply_theme()

    def apply_theme(self) -> None:
        colors = theme_colors(self)
        active = {
            "good": COLORS["green"],
            "warning": COLORS["orange"],
            "bad": COLORS["red"],
            "neutral": css_color(colors["text_secondary"]),
        }.get(self._tone, css_color(colors["text_secondary"]))
        inactive = css_color(colors["text_tertiary"])
        for index, label in enumerate(self._labels):
            label.setStyleSheet(f"color: {active if index < self._count else inactive};")

    def changeEvent(self, event) -> None:  # type: ignore[override]
        super().changeEvent(event)


class InfoPill(QLabel):
    """Small rounded information pill."""

    def __init__(self, text: str = "", tone: str = "neutral", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._tone = tone
        self.setFont(apple_font("text", 12, QFont.Weight.Medium))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.apply_theme()

    def set_pill(self, text: str, tone: str = "neutral") -> None:
        self.setText(text)
        self._tone = tone
        self.apply_theme()

    def apply_theme(self) -> None:
        colors = theme_colors(self)
        tone_map = {
            "good": QColor(COLORS["green"]),
            "warning": QColor(COLORS["orange"]),
            "bad": QColor(COLORS["red"]),
            "blue": QColor(COLORS["blue"]),
            "neutral": QColor(colors["text"]),
        }
        tone = tone_map.get(self._tone, QColor(colors["text"]))
        background = QColor(tone)
        background.setAlpha(26)
        self.setStyleSheet(
            f"""
            QLabel {{
                min-height: 26px;
                padding: 0 10px;
                border-radius: 6px;
                color: {tone.name()};
                background: {background.name(QColor.NameFormat.HexArgb)};
            }}
            """
        )

    def changeEvent(self, event) -> None:  # type: ignore[override]
        super().changeEvent(event)


class ToggleSwitch(QCheckBox):
    """Apple-style toggle switch."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(44, 26)

    def sizeHint(self) -> QSize:
        return QSize(44, 26)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        colors = theme_colors(self)
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            track_rect = QRectF(0, 0, self.width(), self.height())
            track_color = QColor(COLORS["green"]) if self.isChecked() else QColor(colors["border"])
            if not self.isChecked():
                track_color = QColor(colors["text_tertiary"])
                track_color.setAlpha(70 if is_dark_mode(self) else 55)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(track_color)
            painter.drawRoundedRect(track_rect, 13, 13)

            knob_diameter = 22
            knob_x = self.width() - knob_diameter - 2 if self.isChecked() else 2
            knob_rect = QRectF(knob_x, 2, knob_diameter, knob_diameter)
            painter.setBrush(QColor("white"))
            painter.drawEllipse(knob_rect)
        finally:
            painter.end()


class WatchlistRow(QFrame):
    """Compact Apple-style row used on the dashboard."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("watchlistRow")
        self.setMinimumHeight(44)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._hover_value = 0.0
        self._hover_animation = QPropertyAnimation(self, b"hoverValue", self)
        self._hover_animation.setDuration(150)
        self._hover_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)

        self.ticker_label = QLabel("--")
        self.ticker_label.setFont(apple_font("display", 16, QFont.Weight.DemiBold))
        self.score_badge = ScoreBadge()
        self.price_label = QLabel("--")
        self.target_label = QLabel("--")
        self.return_label = QLabel("--")
        self.dots = SignalDotsWidget()

        for label in (self.price_label, self.target_label, self.return_label):
            label.setFont(apple_font("rounded", 14, QFont.Weight.Medium))

        layout.addWidget(self.ticker_label)
        layout.addWidget(self.score_badge)
        layout.addStretch(1)
        layout.addWidget(self.price_label)
        layout.addWidget(self.target_label)
        layout.addWidget(self.return_label)
        layout.addWidget(self.dots)
        self.apply_theme()

    def set_candidate(self, candidate: dict) -> None:
        current = float(candidate.get("current_price", 0.0))
        target = float(candidate.get("targets", {}).get("tp2", current))
        pct = ((target / current) - 1.0) * 100.0 if current else 0.0
        final_score = float(candidate.get("final_score", 0.0))
        tone = "good" if final_score >= 70 else "warning" if final_score >= 54 else "bad"
        self.ticker_label.setText(str(candidate.get("ticker", "--")))
        self.score_badge.set_score(final_score)
        self.price_label.setText(format_money(current))
        self.target_label.setText(format_money(target))
        self.return_label.setText(format_percent(pct))
        self.dots.set_state(int(candidate.get("confluence_count", 0)), tone)
        self.return_label.setStyleSheet(f"color: {tone_color(pct, good_cutoff=4.0, neutral_cutoff=0.0)};")
        self.apply_theme()

    def apply_theme(self) -> None:
        colors = theme_colors(self)
        self.ticker_label.setStyleSheet(f"color: {css_color(colors['text'])};")
        self.price_label.setStyleSheet(f"color: {css_color(colors['text'])};")
        self.target_label.setStyleSheet(f"color: {COLORS['green']};")
        self.setStyleSheet("background: transparent; border: 0;")
        self.update()

    @Property(float)
    def hoverValue(self) -> float:  # noqa: N802 - Qt property naming
        return self._hover_value

    @hoverValue.setter
    def hoverValue(self, value: float) -> None:  # noqa: N802 - Qt property naming
        self._hover_value = value
        self.update()

    def enterEvent(self, event) -> None:  # type: ignore[override]
        self._hover_animation.stop()
        self._hover_animation.setStartValue(self._hover_value)
        self._hover_animation.setEndValue(1.0)
        self._hover_animation.start()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        self._hover_animation.stop()
        self._hover_animation.setStartValue(self._hover_value)
        self._hover_animation.setEndValue(0.0)
        self._hover_animation.start()
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        colors = theme_colors(self)
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            if self._hover_value > 0:
                fill = with_alpha(COLORS["blue"], 0.05 * self._hover_value)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(fill)
                painter.drawRoundedRect(self.rect().adjusted(0, 1, 0, -1), 8, 8)
            painter.setPen(QPen(colors["separator"], 1))
            painter.drawLine(self.rect().bottomLeft(), self.rect().bottomRight())
        finally:
            painter.end()

    def changeEvent(self, event) -> None:  # type: ignore[override]
        super().changeEvent(event)


class TargetRangeBar(QWidget):
    """Thin stop-to-target track with a movable current marker."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._progress = 0.5
        self.setMinimumHeight(18)

    def set_progress(self, progress: float) -> None:
        self._progress = clamp(progress, 0.0, 1.0)
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        colors = theme_colors(self)
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            track_rect = QRectF(0, self.height() / 2 - 2, self.width(), 4)
            left_rect = QRectF(track_rect.left(), track_rect.top(), track_rect.width() * 0.5, track_rect.height())
            right_rect = QRectF(track_rect.center().x(), track_rect.top(), track_rect.width() * 0.5, track_rect.height())
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(COLORS["red"]))
            painter.drawRoundedRect(left_rect, 2, 2)
            painter.setBrush(QColor(COLORS["green"]))
            painter.drawRoundedRect(right_rect, 2, 2)
            painter.setBrush(QColor(colors["border"]))
            painter.drawRoundedRect(track_rect, 2, 2)
            painter.setBrush(QColor(COLORS["red"]))
            painter.drawRoundedRect(left_rect, 2, 2)
            painter.setBrush(QColor(COLORS["green"]))
            painter.drawRoundedRect(right_rect, 2, 2)
            marker_x = track_rect.left() + track_rect.width() * self._progress
            painter.setBrush(QColor(COLORS["blue"]))
            painter.drawEllipse(QPointF(marker_x, track_rect.center().y()), 5, 5)
        finally:
            painter.end()


class PickCard(QFrame):
    """Large Apple-style card for the weekly picks screen."""

    def __init__(self, candidate: dict | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("pickCard")
        self.ticker_label = QLabel("TICK")
        self.company_label = QLabel("Company")
        self.tier_badge = InfoPill("Tier 1", "blue")
        self.sector_badge = InfoPill("Sector", "warning")
        self.price_label = QLabel("$0.00")
        self.target_label = QLabel("$0.00  +0.0%")
        self.range_bar = TargetRangeBar()
        self.current_label = QLabel("Current $0.00")
        self.stop_label = QLabel("Stop $0.00")
        self.rr_label = QLabel("R:R 0.0")
        self.size_label = QLabel("Size 0%")
        self.signals_label = SectionLabel("Signals")
        self.explanation_label = QLabel("")
        self.signal_meters = {
            "Tech": SignalMeter("Tech"),
            "RS": SignalMeter("RS"),
            "Vol": SignalMeter("Vol"),
            "ML": SignalMeter("ML"),
            "Pat": SignalMeter("Pat"),
        }

        self.ticker_label.setFont(apple_font("display", 28, QFont.Weight.Bold))
        self.company_label.setFont(apple_font("text", 13, QFont.Weight.Normal))
        self.price_label.setFont(apple_font("rounded", 24, QFont.Weight.Light))
        self.target_label.setFont(apple_font("rounded", 18, QFont.Weight.DemiBold))
        for label in (self.current_label, self.stop_label, self.rr_label, self.size_label):
            label.setFont(apple_font("text", 12, QFont.Weight.Normal))
        self.explanation_label.setFont(apple_font("text", 13, QFont.Weight.Normal, italic=True))
        self.explanation_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(2)
        left.addWidget(self.ticker_label)
        left.addWidget(self.company_label)
        header.addLayout(left)
        header.addStretch(1)
        header.addWidget(self.tier_badge)
        header.addWidget(self.sector_badge)
        layout.addLayout(header)

        price_row = QHBoxLayout()
        price_row.addWidget(self.price_label)
        price_row.addStretch(1)
        price_row.addWidget(self.target_label)
        layout.addLayout(price_row)

        layout.addWidget(self.range_bar)

        meta_row = QHBoxLayout()
        for label in (self.current_label, self.stop_label, self.rr_label, self.size_label):
            meta_row.addWidget(label)
        meta_row.addStretch(1)
        layout.addLayout(meta_row)

        layout.addWidget(self.signals_label)

        rows = [QHBoxLayout(), QHBoxLayout(), QHBoxLayout()]
        for row in rows:
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(16)
        rows[0].addWidget(self.signal_meters["Tech"])
        rows[0].addWidget(self.signal_meters["RS"])
        rows[1].addWidget(self.signal_meters["Vol"])
        rows[1].addWidget(self.signal_meters["ML"])
        rows[2].addWidget(self.signal_meters["Pat"])
        rows[2].addStretch(1)
        for row in rows:
            layout.addLayout(row)
        layout.addWidget(self.explanation_label)

        self.apply_theme()
        if candidate:
            self.set_candidate(candidate)

    def apply_theme(self) -> None:
        colors = theme_colors(self)
        self.setStyleSheet(
            f"""
            QFrame#pickCard {{
                background: {css_color(colors["base"])};
                border: 1px solid {css_color(colors["border"])};
                border-radius: 12px;
            }}
            QFrame#pickCard:hover {{
                background: {css_color(with_alpha(colors["base"], 0.98))};
            }}
            """
        )
        apply_card_shadow(self, enabled=not is_dark_mode(self))
        self.ticker_label.setStyleSheet(f"color: {css_color(colors['text'])};")
        self.company_label.setStyleSheet(f"color: {css_color(colors['text_secondary'])};")
        self.price_label.setStyleSheet(f"color: {css_color(colors['text'])};")
        self.target_label.setStyleSheet(f"color: {COLORS['green']};")
        for label in (self.current_label, self.stop_label, self.rr_label, self.size_label):
            label.setStyleSheet(f"color: {css_color(colors['text_secondary'])};")
        self.explanation_label.setStyleSheet(f"color: {css_color(colors['text_secondary'])};")

    def changeEvent(self, event) -> None:  # type: ignore[override]
        super().changeEvent(event)

    def set_candidate(self, candidate: dict) -> None:
        current = float(candidate.get("current_price", 0.0))
        target = float(candidate.get("targets", {}).get("tp2", current))
        stop = float(candidate.get("stop_loss", current))
        upside = ((target / current) - 1.0) * 100.0 if current else 0.0
        progress_value = ((current - stop) / max(target - stop, 0.01)) if current and target > stop else 0.5
        final_score = float(candidate.get("final_score", 0.0))
        tier = str(candidate.get("tier_label", "Tier"))
        self.ticker_label.setText(str(candidate.get("ticker", "TICK")))
        self.company_label.setText(str(candidate.get("company_name", "Company")))
        self.tier_badge.set_pill(tier, "blue")
        self.sector_badge.set_pill(
            f"{candidate.get('sector', 'Unknown')} {candidate.get('sector_temperature_tag', '')}".strip(),
            "warning",
        )
        self.price_label.setText(format_money(current))
        self.target_label.setText(f"{format_money(target)}  {format_percent(upside)}")
        self.range_bar.set_progress(progress_value)
        self.current_label.setText(f"Current {format_money(current)}")
        self.stop_label.setText(f"Stop {format_money(stop)}")
        self.rr_label.setText(f"R:R {float(candidate.get('risk_reward', 0.0)):.1f}")
        self.size_label.setText(f"Size {float(candidate.get('kelly_size_pct', 0.0)):.1f}%")
        self.signal_meters["Tech"].set_value(float(candidate.get("technical_score", 0.0)))
        self.signal_meters["RS"].set_value(float(candidate.get("rs_score", 0.0)))
        self.signal_meters["Vol"].set_value(float(candidate.get("volume_momentum_score", 0.0)))
        self.signal_meters["ML"].set_value(float(candidate.get("ml_score", 0.0)), color=COLORS["indigo"])
        self.signal_meters["Pat"].set_value(float(candidate.get("pattern_score", 0.0)), color=COLORS["teal"])
        explanation = str(candidate.get("ai_explanation", "No explanation available."))
        self.explanation_label.setText(explanation)
        self.apply_theme()


class VixGauge(QWidget):
    """Apple-style linear market regime indicator."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._value = 18.0
        self._display_value = 18.0
        self._spy_week = 0.0
        self._breadth: float | None = 0.0
        self._regime = "Neutral"
        self.setMinimumSize(QSize(320, 220))
        self.animation = QPropertyAnimation(self, b"displayValue", self)
        self.animation.setDuration(700)
        self.animation.setEasingCurve(QEasingCurve.Type.OutCubic)

    def sizeHint(self) -> QSize:
        return QSize(320, 240)

    @Property(float)
    def displayValue(self) -> float:  # noqa: N802 - Qt property naming
        return self._display_value

    @displayValue.setter
    def displayValue(self, value: float) -> None:  # noqa: N802 - Qt property naming
        self._display_value = value
        self.update()

    def set_value(self, value: float) -> None:
        self._value = value
        self.animation.stop()
        self.animation.setStartValue(self._display_value)
        self.animation.setEndValue(value)
        self.animation.start()

    def set_context(self, *, spy_week: float, breadth_percent: float | None, regime: str) -> None:
        self._spy_week = spy_week
        self._breadth = breadth_percent
        self._regime = regime
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        colors = theme_colors(self)
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.fillRect(self.rect(), Qt.GlobalColor.transparent)
            header_font = apply_letter_spacing(apple_font("text", 11, QFont.Weight.DemiBold), 0.8)
            value_font = apple_font("rounded", 48, QFont.Weight.ExtraLight)
            label_font = apple_font("text", 13, QFont.Weight.Normal)
            row_font = apple_font("text", 14, QFont.Weight.Normal)

            painter.setPen(QColor(colors["text_secondary"]))
            painter.setFont(header_font)
            painter.drawText(QRectF(0, 0, self.width(), 20), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "MARKET REGIME")

            painter.setPen(QColor(colors["text"]))
            painter.setFont(label_font)
            painter.drawText(QRectF(0, 32, 120, 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "VIX")
            painter.setFont(value_font)
            painter.drawText(QRectF(self.width() - 120, 18, 120, 44), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{self._display_value:.1f}")

            bar_rect = QRectF(0, 74, self.width(), 10)
            gradient = QLinearGradient(bar_rect.left(), 0, bar_rect.right(), 0)
            gradient.setColorAt(0.0, QColor(COLORS["green"]))
            gradient.setColorAt(0.6, QColor(COLORS["orange"]))
            gradient.setColorAt(1.0, QColor(COLORS["red"]))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(gradient)
            painter.drawRoundedRect(bar_rect, 5, 5)

            pip_x = bar_rect.left() + (clamp(self._display_value, 0.0, 40.0) / 40.0) * bar_rect.width()
            pip_color = QColor(colors["base"]) if is_dark_mode(self) else QColor(colors["text"])
            painter.setBrush(pip_color)
            painter.drawEllipse(QPointF(pip_x, bar_rect.center().y()), 7, 7)

            painter.setFont(apple_font("text", 11, QFont.Weight.Normal))
            painter.setPen(QColor(colors["text_secondary"]))
            painter.drawText(QRectF(0, 92, self.width() / 3, 18), Qt.AlignmentFlag.AlignLeft, "CALM")
            painter.drawText(QRectF(self.width() / 3, 92, self.width() / 3, 18), Qt.AlignmentFlag.AlignHCenter, "CAUTION")
            painter.drawText(QRectF(2 * self.width() / 3, 92, self.width() / 3, 18), Qt.AlignmentFlag.AlignRight, "RISK")

            painter.setFont(row_font)
            rows = [
                ("SPY this week", f"{self._spy_week:+.1f}%"),
                ("Market breadth", f"{self._breadth:.0f}%" if self._breadth is not None else "Unavailable"),
                ("Regime", self._regime),
            ]
            start_y = 124
            for index, (label, value) in enumerate(rows):
                y = start_y + (index * 28)
                painter.setPen(QColor(colors["text_secondary"]))
                painter.drawText(QRectF(0, y, self.width() / 2, 22), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label)
                painter.setPen(QColor(colors["text"]))
                painter.drawText(QRectF(self.width() / 2, y, self.width() / 2, 22), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, value)
        finally:
            painter.end()


class PerformanceBarsWidget(QWidget):
    """Simple interactive weekly performance bar chart."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: list[dict] = []
        self.setMouseTracking(True)
        self.setMinimumHeight(220)

    def set_rows(self, rows: list[dict]) -> None:
        self._rows = rows[-8:]
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            colors = theme_colors(self)
            painter.fillRect(self.rect(), Qt.GlobalColor.transparent)

            if not self._rows:
                painter.setPen(QColor(colors["text_secondary"]))
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No completed results yet")
                return

            margin_left = 54
            margin_bottom = 44
            chart_rect = self.rect().adjusted(margin_left, 20, -20, -margin_bottom)
            values = [float(row.get("avg_return") or 0.0) for row in self._rows]
            max_abs = max(2.0, max(abs(value) for value in values))
            zero_y = chart_rect.center().y()
            completed_count = sum(row.get("avg_return") is not None for row in self._rows)

            painter.setPen(QPen(QColor(colors["separator"]), 1))
            painter.drawLine(chart_rect.left(), int(zero_y), chart_rect.right(), int(zero_y))

            bar_width = 24
            gap = (chart_rect.width() - bar_width * len(self._rows)) / max(len(self._rows) - 1, 1)

            if completed_count == 0:
                painter.setPen(QColor(colors["text_secondary"]))
                painter.setFont(apple_font("text", 13, QFont.Weight.Normal))
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No completed weekly results yet")
                return

            for index, row in enumerate(self._rows):
                value = float(row.get("avg_return") or 0.0)
                x = chart_rect.left() + index * (bar_width + gap)
                height = (abs(value) / max_abs) * (chart_rect.height() / 2 - 8)
                if row.get("avg_return") is None:
                    bar_rect = QRectF(x, zero_y - 10, bar_width, 20)
                    color = with_alpha(colors["text"], 0.22)
                elif value >= 0:
                    bar_rect = QRectF(x, zero_y - height, bar_width, height)
                    color = QColor(COLORS["green"])
                else:
                    bar_rect = QRectF(x, zero_y, bar_width, height)
                    color = QColor(COLORS["red"])
                painter.setBrush(color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(bar_rect, 6, 6)
                painter.setPen(QColor(colors["text_secondary"]))
                painter.setFont(apple_font("text", 11, QFont.Weight.Normal))
                label_rect = QRectF(x - 8, chart_rect.bottom() + 8, bar_width + 16, 18)
                painter.drawText(label_rect, Qt.AlignmentFlag.AlignHCenter, str(row.get("week_label", "")))
        finally:
            painter.end()



    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        def safe_float(val, default=0.0):
            try:
                return float(val) if val is not None else default
            except (TypeError, ValueError):
                return default

        if not self._rows:
            return
        margin_left = 54
        margin_bottom = 44
        chart_rect = self.rect().adjusted(margin_left, 20, -20, -margin_bottom)
        bar_width = max(22, int(chart_rect.width() / max(len(self._rows) * 1.8, 1)))
        gap = (chart_rect.width() - bar_width * len(self._rows)) / max(len(self._rows) - 1, 1)
        for index, row in enumerate(self._rows):
            x = chart_rect.left() + index * (bar_width + gap)
            rect = QRectF(x, chart_rect.top(), bar_width, chart_rect.height())
            if rect.contains(event.position()):
                QToolTip.showText(
                    event.globalPosition().toPoint(),
                    (
                        f"{row.get('week_label', '')}\n"
                        f"Picks: {row.get('picks', '-')}\n"
                        f"Target hit rate: {safe_float(row.get('target_hit_rate') or row.get('hit_rate')):.0f}%\n"
                        f"Positive-return rate: {safe_float(row.get('positive_return_rate')):.0f}%\n"
                        f"Avg return: {safe_float(row.get('avg_return')):+.1f}%"
                    ),
                    self,
                )
                return
        QToolTip.hideText()


class CumulativeReturnChart(QWidget):
    """Lightweight cumulative return line chart without QtCharts."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: list[dict] = []
        self.setMinimumHeight(320)

    def set_rows(self, rows: list[dict]) -> None:
        self._rows = rows
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            colors = theme_colors(self)
            painter.fillRect(self.rect(), Qt.GlobalColor.transparent)

            if not self._rows:
                painter.setPen(QColor(colors["text_secondary"]))
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No completed results yet")
                return

            margin_left = 58
            margin_right = 26
            margin_top = 26
            margin_bottom = 46
            chart_rect = self.rect().adjusted(margin_left, margin_top, -margin_right, -margin_bottom)
            if chart_rect.width() <= 0 or chart_rect.height() <= 0:
                return

            values = [float(row.get("cumulative_return", 0.0)) for row in self._rows]
            min_value = min(min(values), 0.0)
            max_value = max(max(values), 0.0)
            if math.isclose(max_value, min_value):
                max_value += 1.0
                min_value -= 1.0

            def map_y(value: float) -> float:
                scale = (value - min_value) / max(max_value - min_value, 0.001)
                return chart_rect.bottom() - (scale * chart_rect.height())

            zero_y = map_y(0.0)
            painter.setPen(QPen(QColor(colors["separator"]), 1, Qt.PenStyle.DashLine))
            painter.drawLine(chart_rect.left(), int(zero_y), chart_rect.right(), int(zero_y))

            point_count = len(values)
            if point_count == 1:
                points = [QPointF(chart_rect.center().x(), map_y(values[0]))]
            else:
                step_x = chart_rect.width() / max(point_count - 1, 1)
                points = [QPointF(chart_rect.left() + (index * step_x), map_y(value)) for index, value in enumerate(values)]

            line_path = QPainterPath(points[0])
            if len(points) > 1:
                for index in range(1, len(points)):
                    prev = points[index - 1]
                    point = points[index]
                    midpoint_x = (prev.x() + point.x()) / 2
                    line_path.cubicTo(QPointF(midpoint_x, prev.y()), QPointF(midpoint_x, point.y()), point)

            trend_color = COLORS["blue"] if values[-1] >= 0 else COLORS["red"]
            painter.setPen(QPen(QColor(trend_color), 2))
            painter.drawPath(line_path)

            painter.setPen(QColor(colors["text_secondary"]))
            for value, text in ((max_value, f"{max_value:+.1f}%"), (0.0, "0.0%"), (min_value, f"{min_value:+.1f}%")):
                y = map_y(value)
                painter.drawText(QRectF(8, y - 10, margin_left - 12, 20), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, text)

            painter.setFont(apple_font("text", 11, QFont.Weight.Normal))
            sample_indices = sorted(set([0, len(self._rows) // 2, len(self._rows) - 1]))
            for index in sample_indices:
                point = points[index if point_count > 1 else 0]
                label = str(self._rows[index].get("week_label", ""))
                painter.drawText(
                    QRectF(point.x() - 40, chart_rect.bottom() + 10, 80, 18),
                    Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                    label,
                )
        finally:
            painter.end()


class FeatureImportanceChart(QWidget):
    """Simple horizontal bar chart for feature importance."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._items: list[tuple[str, float]] = []
        self.setMinimumHeight(260)

    def set_items(self, importance: dict[str, float]) -> None:
        self._items = sorted(importance.items(), key=lambda item: item[1], reverse=True)[:8]
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            colors = theme_colors(self)
            painter.fillRect(self.rect(), Qt.GlobalColor.transparent)

            if not self._items:
                painter.setPen(QColor(colors["text_secondary"]))
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Feature importance will appear after training.")
                return

            margin_left = 160
            margin_right = 44
            margin_top = 26
            margin_bottom = 22
            chart_rect = self.rect().adjusted(margin_left, margin_top, -margin_right, -margin_bottom)
            max_value = max((weight for _, weight in self._items), default=0.01)
            row_height = chart_rect.height() / max(len(self._items), 1)

            for index, (name, weight) in enumerate(self._items):
                y = chart_rect.top() + (index * row_height)
                bar_y = y + 8
                bar_height = max(12.0, row_height - 16.0)
                width = (weight / max(max_value, 0.001)) * chart_rect.width()
                track_rect = QRectF(chart_rect.left(), bar_y, chart_rect.width(), bar_height)
                value_rect = QRectF(chart_rect.left(), bar_y, width, bar_height)
                label_rect = QRectF(16, y, margin_left - 24, row_height)
                pct_rect = QRectF(chart_rect.right() + 8, y, margin_right - 12, row_height)

                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(colors["border"]))
                painter.drawRoundedRect(track_rect, 7, 7)
                painter.setBrush(QColor(COLORS["blue"]))
                painter.drawRoundedRect(value_rect, 7, 7)

                painter.setFont(apple_font("text", 13, QFont.Weight.Normal))
                painter.setPen(QColor(colors["text"]))
                painter.drawText(label_rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, name.replace("_", " "))
                painter.setPen(QColor(colors["text_secondary"]))
                painter.drawText(pct_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, f"{weight * 100.0:.1f}%")
        finally:
            painter.end()


class ScanProgressDialog(QDialog):
    """Modal progress dialog shown while the weekly scan is running."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Scanning...")
        self.setModal(True)
        self.setMinimumWidth(460)
        colors = theme_colors(self)
        self.setStyleSheet(f"background: {css_color(with_alpha(colors['window'], 0.72))};")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(0)

        self.sheet = QFrame()
        self.sheet.setObjectName("scanSheet")
        self.sheet.setStyleSheet(
            f"""
            QFrame#scanSheet {{
                background: {css_color(colors["sheet"])};
                border: 1px solid {css_color(colors["border"])};
                border-radius: 16px;
            }}
            """
        )
        apply_card_shadow(self.sheet, enabled=not is_dark_mode(self))
        sheet_layout = QVBoxLayout(self.sheet)
        sheet_layout.setContentsMargins(24, 24, 24, 24)
        sheet_layout.setSpacing(12)

        title = QLabel("Scanning...")
        title.setObjectName("title")
        title.setFont(apple_font("display", 28, QFont.Weight.DemiBold))
        title.setStyleSheet(f"color: {css_color(colors['text'])};")
        self.subtitle = QLabel("Analyzing stocks")
        self.subtitle.setFont(apple_font("text", 14, QFont.Weight.Normal))
        self.subtitle.setStyleSheet(f"color: {css_color(colors['text_secondary'])};")

        self.stage1_label = QLabel("Stage 1: Filtering tickers")
        self.stage1_label.setFont(apple_font("text", 13, QFont.Weight.Medium))
        self.stage1_label.setStyleSheet(f"color: {{css_color(colors['text'])}};")
        self.stage1_bar = QProgressBar()
        self.stage1_bar.setRange(0, 100)
        self.stage2_label = QLabel("Stage 2: Scoring survivors")
        self.stage2_label.setFont(apple_font("text", 13, QFont.Weight.Medium))
        self.stage2_label.setStyleSheet(f"color: {css_color(colors['text'])};")
        self.stage2_bar = QProgressBar()
        self.stage2_bar.setRange(0, 100)
        self.current_label = QLabel("Current: warming up")
        self.current_label.setFont(apple_font("mono", 13, QFont.Weight.Normal))
        self.current_label.setStyleSheet(f"color: {css_color(colors['text_secondary'])};")
        self.elapsed_label = QLabel("Elapsed: 0s")
        self.elapsed_label.setFont(apple_font("text", 12, QFont.Weight.Normal))
        self.elapsed_label.setStyleSheet(f"color: {css_color(colors['text_secondary'])};")
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.setFont(apple_font("text", 13, QFont.Weight.Normal))
        self.summary_label.setStyleSheet(f"color: {css_color(colors['text'])};")
        for bar in (self.stage1_bar, self.stage2_bar):
            bar.setTextVisible(False)
            bar.setFixedHeight(4)
            bar.setStyleSheet(
                f"""
                QProgressBar {{
                    background: {css_color(colors["border"], 0.5)};
                    border: 0;
                    border-radius: 2px;
                }}
                QProgressBar::chunk {{
                    background: {COLORS["blue"]};
                    border-radius: 2px;
                }}
                """
            )

        sheet_layout.addWidget(title)
        sheet_layout.addWidget(self.subtitle)
        sheet_layout.addSpacing(6)
        sheet_layout.addWidget(self.stage1_label)
        sheet_layout.addWidget(self.stage1_bar)
        sheet_layout.addWidget(self.stage2_label)
        sheet_layout.addWidget(self.stage2_bar)
        sheet_layout.addSpacing(8)
        sheet_layout.addWidget(self.current_label)
        sheet_layout.addWidget(self.elapsed_label, alignment=Qt.AlignmentFlag.AlignRight)
        sheet_layout.addWidget(self.summary_label)
        layout.addWidget(self.sheet)

        self._started_at = time.perf_counter()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick_elapsed)
        self._timer.start(250)

    def _tick_elapsed(self) -> None:
        elapsed = int(time.perf_counter() - self._started_at)
        self.elapsed_label.setText(f"Elapsed: {elapsed}s")

    def update_progress(self, payload: dict) -> None:
        self.stage1_bar.setValue(int(payload.get("stage1", 0)))
        self.stage2_bar.setValue(int(payload.get("stage2", 0)))
        self.current_label.setText(f"Current: {payload.get('current', 'working...')}")
        if payload.get("subtitle"):
            self.subtitle.setText(str(payload["subtitle"]))
        if payload.get("stage1_text"):
            self.stage1_label.setText(str(payload["stage1_text"]))
        if payload.get("stage2_text"):
            self.stage2_label.setText(str(payload["stage2_text"]))

    def set_completed(self, text: str) -> None:
        self.stage1_bar.setValue(100)
        self.stage2_bar.setValue(100)
        self.summary_label.setText(text)
        self._timer.stop()

    def set_failed(self, text: str) -> None:
        self.summary_label.setText(text)
        self.summary_label.setStyleSheet(f"color: {COLORS['red']};")
        self._timer.stop()


class ScanWorker(QThread):
    """Run the scan subprocess and emit UI-friendly progress updates."""

    progress_changed = Signal(dict)
    scan_finished = Signal(dict)
    scan_failed = Signal(str)
    log_line = Signal(str)

    def __init__(self, project_root: Path, universe_mode: str = "full", top_n: int = 10) -> None:
        super().__init__()
        self.project_root = project_root
        self.universe_mode = universe_mode
        self.top_n = top_n
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:  # type: ignore[override]
        payload_path = self.project_root / "stock_predictor" / "artifacts" / "latest_scan.json"
        env = os.environ.copy()
        env.update({"PYTHONUNBUFFERED": "1"})
        command = [
            sys.executable,
            "main.py",
            "--top-n",
            str(self.top_n),
            "--universe",
            self.universe_mode,
        ]
        try:
            process = subprocess.Popen(
                command,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=0,
                env=env,
            )
        except Exception as exc:
            self.scan_failed.emit(f"Failed to start scan: {exc}")
            return

        stage1 = 8
        stage2 = 0
        buffer = ""
        started = time.perf_counter()
        stage1_initial = f"Stage 1: Filtering {'~10,000' if self.universe_mode == 'us_market' else '565'} tickers"
        subtitle_text = f"Analyzing {'~10,000' if self.universe_mode == 'us_market' else '565'} stocks"
        self.progress_changed.emit(
            {
                "subtitle": subtitle_text,
                "stage1": stage1,
                "stage2": stage2,
                "stage1_text": stage1_initial,
                "stage2_text": "Stage 2: Scoring survivors",
                "current": "Loading cached market data",
            }
        )

        def handle_line(raw_line: str) -> None:
            nonlocal stage1, stage2
            line = ANSI_RE.sub("", raw_line).strip()
            if not line:
                return
            self.log_line.emit(line)
            match = TQDM_PROGRESS_RE.search(line)
            if match:
                phase, percent, done, total, current = match.groups()
                if phase == "Scoring tickers":
                    stage1 = 100
                    stage2 = int(clamp((int(done) / max(int(total), 1)) * 55.0, 0.0, 55.0))
                else:
                    stage1 = 100
                    stage2 = 55 + int(clamp((int(done) / max(int(total), 1)) * 45.0, 0.0, 45.0))
                self.progress_changed.emit(
                    {
                        "stage1": stage1,
                        "stage2": stage2,
                        "stage1_text": f"Stage 1: filtered {total} tickers" if phase != "Scoring tickers" else "Stage 1: filtered tickers",
                        "stage2_text": f"Stage 2: scoring {total} survivors",
                        "current": current or phase,
                    }
                )
                return
            if "WEEKLY STOCK SCAN" in line:
                stage1 = 100
                stage2 = 100
                self.progress_changed.emit(
                    {
                        "stage1": stage1,
                        "stage2": stage2,
                        "current": "Rendering results",
                    }
                )

        stream = process.stdout
        if stream is None:
            self.scan_failed.emit("Scan process did not provide output.")
            return

        while True:
            if self._cancelled and process.poll() is None:
                process.terminate()
                self.scan_failed.emit("Scan canceled.")
                return
            char = stream.read(1)
            if char == "":
                if process.poll() is not None:
                    if buffer.strip():
                        handle_line(buffer)
                    break
                elapsed = time.perf_counter() - started
                if stage1 < 95:
                    stage1 = min(95, int(20 + elapsed * 18))
                    self.progress_changed.emit(
                        {
                            "stage1": stage1,
                            "stage2": stage2,
                            "current": "Preparing market snapshot",
                        }
                    )
                self.msleep(50)
                continue
            if char in "\r\n":
                if buffer.strip():
                    handle_line(buffer)
                buffer = ""
            else:
                buffer += char

        returncode = process.wait()
        if returncode != 0:
            self.scan_failed.emit(f"Scan failed with exit code {returncode}.")
            return
        if not payload_path.exists():
            self.scan_failed.emit("Scan completed but no latest_scan.json was produced.")
            return
        try:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.scan_failed.emit(f"Scan completed but payload could not be read: {exc}")
            return
        self.scan_finished.emit(payload)


class AnalyzeTickerWorker(QThread):
    """Run a single-ticker analysis subprocess and emit the parsed payload."""

    progress_changed = Signal(str)
    analysis_finished = Signal(dict)
    analysis_failed = Signal(str)

    def __init__(self, project_root: Path, ticker: str, universe_mode: str = "full") -> None:
        super().__init__()
        self.project_root = project_root
        self.ticker = ticker.upper().strip()
        self.universe_mode = universe_mode

    def run(self) -> None:  # type: ignore[override]
        payload_path = self.project_root / "stock_predictor" / "artifacts" / "latest_single_analysis.json"
        env = os.environ.copy()
        env.update({"PYTHONUNBUFFERED": "1"})
        command = [
            sys.executable,
            "main.py",
            "--analyze",
            self.ticker,
            "--universe",
            self.universe_mode,
        ]
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
            self.analysis_failed.emit(f"Failed to start analysis: {exc}")
            return

        self.progress_changed.emit(f"Analyzing {self.ticker}…")
        try:
            if process.stdout is not None:
                for raw_line in process.stdout:
                    line = ANSI_RE.sub("", raw_line).strip()
                    if line and not should_ignore_analysis_output_line(line):
                        self.progress_changed.emit(line)
        finally:
            returncode = process.wait()

        if returncode != 0:
            self.analysis_failed.emit(f"Analysis failed with exit code {returncode}.")
            return
        if not payload_path.exists():
            self.analysis_failed.emit("Analysis completed but no result file was produced.")
            return
        try:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.analysis_failed.emit(f"Analysis payload could not be read: {exc}")
            return
        self.analysis_finished.emit(payload)
