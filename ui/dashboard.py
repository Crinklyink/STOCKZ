"""Dashboard screen for the native Stock Predictor app."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from ui.widgets import (
    PerformanceBarsWidget,
    SectionLabel,
    StatCard,
    VixGauge,
    WatchlistRow,
    apple_font,
    apply_card_shadow,
    css_color,
    is_dark_mode,
    theme_colors,
)


class DashboardView(QWidget):
    """Default dashboard screen with high-level stats and market context."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 28, 28, 28)
        root.setSpacing(14)

        self.page_title = QLabel("Performance Snapshot")
        self.page_title.setFont(apple_font("display", 22, QFont.Weight.Bold))
        root.addWidget(self.page_title)

        self.cards = {
            "week": StatCard("This Week", "--", "7-day performance"),
            "win_rate": StatCard("Hit Rate", "--", "hit rate"),
            "avg_return": StatCard("Avg Return", "--", "8 week return"),
            "vix": StatCard("VIX", "--", "trend"),
        }
        card_row = QHBoxLayout()
        card_row.setSpacing(12)
        for card in self.cards.values():
            card_row.addWidget(card, 1)
        root.addLayout(card_row)

        self.banner = QFrame()
        banner_layout = QHBoxLayout(self.banner)
        banner_layout.setContentsMargins(14, 10, 14, 10)
        banner_layout.setSpacing(10)
        self.banner_dot = QLabel("●")
        self.banner_dot.setFont(apple_font("rounded", 14))
        self.banner_label = QLabel("Loading regime...")
        self.banner_label.setFont(apple_font("text", 12))
        self.banner_label.setWordWrap(True)
        self.banner_label.setMinimumHeight(36)
        self.banner_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        self.banner_meta = QLabel("--")
        self.banner_meta.setFont(apple_font("text", 12))
        self.banner_meta.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        banner_layout.addWidget(self.banner_dot)
        banner_layout.addWidget(self.banner_label, 1)
        banner_layout.addWidget(self.banner_meta)
        root.addWidget(self.banner)

        middle_row = QHBoxLayout()
        middle_row.setSpacing(12)

        self.picks_panel = QFrame()
        picks_layout = QVBoxLayout(self.picks_panel)
        picks_layout.setContentsMargins(20, 20, 20, 20)
        picks_layout.setSpacing(8)
        self.picks_title = SectionLabel("This Week")
        picks_layout.addWidget(self.picks_title)
        self.rows_container = QVBoxLayout()
        self.rows_container.setContentsMargins(0, 0, 0, 0)
        self.rows_container.setSpacing(0)
        self.pick_rows = [WatchlistRow() for _ in range(6)]
        for row in self.pick_rows:
            self.rows_container.addWidget(row)
            row.hide()
        picks_layout.addLayout(self.rows_container)
        picks_layout.addStretch(1)
        middle_row.addWidget(self.picks_panel, 1)

        self.regime_panel = QFrame()
        regime_layout = QVBoxLayout(self.regime_panel)
        regime_layout.setContentsMargins(20, 20, 20, 20)
        regime_layout.setSpacing(12)
        self.regime_title = SectionLabel("Market Regime")
        self.vix_gauge = VixGauge()
        regime_layout.addWidget(self.regime_title)
        regime_layout.addWidget(self.vix_gauge, 1)
        middle_row.addWidget(self.regime_panel, 1)
        root.addLayout(middle_row, 1)

        self.performance_panel = QFrame()
        performance_layout = QVBoxLayout(self.performance_panel)
        performance_layout.setContentsMargins(20, 20, 20, 20)
        performance_layout.setSpacing(12)
        self.performance_title = SectionLabel("8-Week Performance")
        self.performance_chart = PerformanceBarsWidget()
        performance_layout.addWidget(self.performance_title)
        performance_layout.addWidget(self.performance_chart)
        root.addWidget(self.performance_panel, 1)

        self.apply_theme()

    def apply_theme(self) -> None:
        colors = theme_colors(self)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(f"{self.__class__.__name__} {{ background-color: {css_color(colors['window'])}; }}")
        self.page_title.setStyleSheet(
            f"color: {css_color(colors['text'])}; background: transparent; border: none;"
        )
        base_bg = css_color(colors["base"])
        border_c = css_color(colors["border"])
        # Scoped stylesheet — border only targets the QFrame itself, not children
        scoped_panel = (
            f"QFrame {{ background: {base_bg}; border: 1px solid {border_c}; border-radius: 8px; }}"
            f" QLabel {{ background: transparent; border: none; }}"
            f" QWidget {{ background: transparent; border: none; }}"
        )
        for panel in (self.banner, self.picks_panel, self.regime_panel, self.performance_panel):
            panel.setStyleSheet(scoped_panel)
            apply_card_shadow(panel, enabled=not is_dark_mode(panel))
        self.banner_label.setStyleSheet(
            f"color: {css_color(colors['text'])}; background: transparent; border: none;"
        )
        self.banner_meta.setStyleSheet(
            f"color: {css_color(colors['text_secondary'])}; background: transparent; border: none;"
        )
    def changeEvent(self, event) -> None:  # type: ignore[override]
        super().changeEvent(event)

    def update_data(self, state: dict[str, Any]) -> None:
        import logging
        logger = logging.getLogger("ui.dashboard")

        payload = state.get("payload", {})
        summary = payload.get("scan_summary", {})
        rolling = state.get("rolling_summary", {})
        
        if not rolling:
            logger.info("rolling_summary unavailable or empty.")

        picks = payload.get("selected", []) or payload.get("display_candidates", [])
        official_count = len(payload.get("selected", []))
        display_count = len(payload.get("display_candidates", []))

        week_tone = "good" if official_count else "warning"
        week_subtitle = f"{official_count} official picks" if official_count else f"{display_count} watchlist names"
        self.cards["week"].set_data("This Week", f"{display_count} picks", week_subtitle, week_tone)

        target_hit_rate = rolling.get("target_hit_rate")
        if target_hit_rate is None:
            target_hit_rate = rolling.get("win_rate")

        if target_hit_rate is not None:
            target_hit_rate = float(target_hit_rate)
            pos_return_val = rolling.get("positive_return_rate")
            pos_str = f"8 weeks · positive {float(pos_return_val):.0f}%" if pos_return_val is not None else "8 weeks"
            self.cards["win_rate"].set_data(
                "Target Hit Rate",
                f"{target_hit_rate:.0f}%",
                pos_str,
                "good" if target_hit_rate >= 50 else "bad",
            )
        else:
            logger.info("target_hit_rate missing from rolling_summary")
            self.cards["win_rate"].set_data(
                "Target Hit Rate",
                "Not yet tracked",
                "No completed results yet",
                "warning",
            )

        avg_return = rolling.get("average_return")
        if avg_return is not None:
            avg_return = float(avg_return)
            self.cards["avg_return"].set_data(
                "Avg Return",
                f"{avg_return:+.2f}%/week",
                "last 8 weeks",
                "good" if avg_return >= 0 else "bad",
            )
        else:
            logger.info("average_return missing from rolling_summary")
            self.cards["avg_return"].set_data(
                "Avg Return",
                "Not yet tracked",
                "No completed results yet",
                "warning",
            )

        vix = float(summary.get("vix", 0.0))
        self.cards["vix"].set_data(
            "VIX",
            f"{vix:.1f}",
            "high" if vix > 25 else "caution" if vix > 20 else "calm",
            "bad" if vix > 25 else "warning" if vix > 20 else "good",
        )

        regime = str(payload.get("regime_label") or summary.get("regime", "Neutral")).upper()
        self.banner_label.setText(self._format_banner_text(str(summary.get("selection_warning", regime))))
        
        spy_val = summary.get("spy_week_return")
        spy = float(spy_val) if spy_val is not None else 0.0
        
        macro = payload.get("macro", {})
        breadth_val = macro.get("breadth_percentile")
        if breadth_val is None:
            logger.info("Market breadth missing from source.")
            breadth_str = "Unavailable"
            breadth = None
        elif breadth_val == 0.5:
            logger.info("Market breadth defaulted to 0.5. Treating as missing.")
            breadth_str = "Unavailable"
            breadth = None
        else:
            breadth = float(breadth_val) * 100.0
            breadth_str = f"{breadth:.0f}%"
            
        dxy_val = macro.get("dxy_5d_return")
        dxy_str = f"{float(dxy_val) * 100.0:+.1f}%" if dxy_val is not None else "Unavailable"
        
        self.banner_meta.setText(f"SPY {spy:+.1f}% · Breadth {breadth_str} · DXY {dxy_str}")
        dot_color = "green" if vix < 20 else "orange" if vix < 25 else "red"
        self.banner_dot.setStyleSheet(f"color: {css_color(dot_color)};")

        self.picks_title.setText("THIS WEEK" if official_count else "WATCHLIST")
        for index, row in enumerate(self.pick_rows):
            if index < len(picks[:6]):
                row.set_candidate(picks[index])
                row.show()
            else:
                row.hide()

        self.vix_gauge.set_value(vix)
        self.vix_gauge.set_context(spy_week=spy, breadth_percent=breadth, regime=regime.title())
        
        weekly_rows = state.get("weekly_rows", [])
        if not weekly_rows or sum(1 for row in weekly_rows if row.get("avg_return") is not None) == 0:
            logger.info("No completed weekly rows available.")
        self.performance_chart.set_rows(weekly_rows)

    @staticmethod
    def _format_banner_text(text: str) -> str:
        cleaned = " ".join(text.split())
        if len(cleaned) <= 120:
            return cleaned
        separators = [" — ", ". ", " | ", "; "]
        for separator in separators:
            if separator.strip() in cleaned:
                left, right = cleaned.split(separator.strip(), 1)
                if left and right:
                    return f"{left.strip()}{separator.strip()}\n{right.strip()}"
        midpoint = len(cleaned) // 2
        split_at = cleaned.rfind(" ", 0, midpoint + 20)
        if split_at == -1:
            split_at = midpoint
        return f"{cleaned[:split_at].strip()}\n{cleaned[split_at:].strip()}"
