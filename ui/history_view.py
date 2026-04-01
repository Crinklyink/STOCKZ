"""Performance and history screens for the native Stock Predictor app."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from ui.widgets import (
    COLORS,
    CumulativeReturnChart,
    FeatureImportanceChart,
    SectionLabel,
    StatCard,
    apple_font,
    apply_card_shadow,
    css_color,
    format_money,
    format_percent,
    is_dark_mode,
    style_secondary_button,
    theme_colors,
)


def _safe_money(value: Any, *, prefix: str = "") -> str:
    try:
        if value is None or value == "":
            return f"{prefix}pending".strip()
        numeric = float(value)
        if numeric != numeric:
            return f"{prefix}pending".strip()
        return f"{prefix}{format_money(numeric)}".strip()
    except (TypeError, ValueError):
        return f"{prefix}pending".strip()


def build_history_detail_cells(detail: dict[str, Any]) -> list[str]:
    realized = detail.get("realized_return_pct")
    high = detail.get("window_high_price")
    latest = detail.get("current_price")
    if realized is None:
        status = "Pending"
    else:
        status = "Target hit" if detail.get("hit_target") else "Missed"
    latest_text = _safe_money(high if high is not None else latest, prefix="High " if high is not None else "Latest ")
    return [
        f"  {detail.get('ticker', '')}",
        _safe_money(detail.get("entry_price"), prefix="Entry "),
        _safe_money(detail.get("target_price"), prefix="Target "),
        latest_text,
        f"{status} · {format_percent(float(realized))}" if realized is not None else status,
        str(detail.get("sector", "")),
    ]


class PerformanceView(QWidget):
    """Performance analytics screen."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        self.stat_cards = [
            StatCard("Weeks", "--", "Tracked"),
            StatCard("Target Hit Rate", "--", "Completed trades"),
            StatCard("Total Return", "--", "Cumulative"),
            StatCard("Best Week", "--", "Top result"),
            StatCard("Worst Week", "--", "Weakest result"),
            StatCard("Current Streak", "--", "Momentum"),
        ]
        stat_grid = QGridLayout()
        stat_grid.setHorizontalSpacing(12)
        stat_grid.setVerticalSpacing(12)
        for index, card in enumerate(self.stat_cards):
            stat_grid.addWidget(card, index // 3, index % 3)
        root.addLayout(stat_grid)

        middle_row = QHBoxLayout()
        middle_row.setSpacing(12)

        self.line_panel = QFrame()
        line_layout = QVBoxLayout(self.line_panel)
        line_layout.setContentsMargins(20, 20, 20, 20)
        line_layout.setSpacing(12)
        line_layout.addWidget(SectionLabel("Cumulative Return"))
        self.line_chart = CumulativeReturnChart()
        line_layout.addWidget(self.line_chart)
        middle_row.addWidget(self.line_panel, 2)

        self.side_panel = QFrame()
        side_layout = QVBoxLayout(self.side_panel)
        side_layout.setContentsMargins(20, 20, 20, 20)
        side_layout.setSpacing(10)
        side_layout.addWidget(SectionLabel("Performance Breakdown"))
        self.sector_list = QListWidget()
        self.vix_list = QListWidget()
        self.tier_list = QListWidget()
        for label, widget in (("Sectors", self.sector_list), ("VIX Levels", self.vix_list), ("Score Tiers", self.tier_list)):
            caption = QLabel(label)
            caption.setFont(apple_font("text", 13, QFont.Weight.DemiBold))
            side_layout.addWidget(caption)
            side_layout.addWidget(widget, 1)
        middle_row.addWidget(self.side_panel, 1)
        root.addLayout(middle_row)

        self.feature_panel = QFrame()
        feature_layout = QVBoxLayout(self.feature_panel)
        feature_layout.setContentsMargins(20, 20, 20, 20)
        feature_layout.setSpacing(12)
        feature_layout.addWidget(SectionLabel("Feature Importance"))
        self.feature_chart = FeatureImportanceChart()
        feature_layout.addWidget(self.feature_chart)
        root.addWidget(self.feature_panel)
        self.apply_theme()

    def apply_theme(self) -> None:
        colors = theme_colors(self)
        self.setStyleSheet(f"background: {css_color(colors['window'])};")
        panel_style = f"background: {css_color(colors['base'])}; border: 1px solid {css_color(colors['border'])}; border-radius: 12px;"
        for panel in (self.line_panel, self.side_panel, self.feature_panel):
            panel.setStyleSheet(panel_style)
            apply_card_shadow(panel, enabled=not is_dark_mode(panel))
        list_style = (
            f"background: transparent; border: 0; color: {css_color(colors['text'])};"
            f"font-size: 13px; outline: 0;"
        )
        for widget in (self.sector_list, self.vix_list, self.tier_list):
            widget.setStyleSheet(list_style)

    def changeEvent(self, event) -> None:  # type: ignore[override]
        super().changeEvent(event)

    def update_data(self, state: dict[str, Any]) -> None:
        import logging
        logger = logging.getLogger("ui.history_view")

        rolling = state.get("rolling_summary", {})
        weekly_rows = state.get("weekly_rows", [])
        cumulative_rows = state.get("cumulative_rows", [])
        model_meta = state.get("model_metadata", {})
        breakdowns = state.get("breakdowns", {})

        total_weeks = len(weekly_rows)
        completed_count = sum(1 for row in weekly_rows if row.get("avg_return") is not None)

        if not rolling:
            logger.info("rolling_summary unavailable. Dashboard showing empty states.")
            pending_text = f"{total_weeks} pending" if total_weeks else "None yet"
            self.stat_cards[0].set_data("Total Weeks", str(total_weeks), pending_text, "neutral")
            self.stat_cards[1].set_data("Target Hit Rate", "Not yet tracked", "No completed results yet", "warning")
            self.stat_cards[2].set_data("Total Return", "Not yet tracked", "cumulative", "warning")
            self.stat_cards[3].set_data("Best Week", "Not yet tracked", "-", "neutral")
            self.stat_cards[4].set_data("Worst Week", "Not yet tracked", "-", "neutral")
            self.stat_cards[5].set_data("Current Streak", "Not yet tracked", "No completed results yet", "neutral")
        else:
            self.stat_cards[0].set_data("Total Weeks", str(total_weeks), f"{completed_count} completed", "neutral")
            
            target_hit_rate = rolling.get("target_hit_rate")
            if target_hit_rate is None:
                target_hit_rate = rolling.get("win_rate")

            if target_hit_rate is not None:
                target_hit_rate = float(target_hit_rate)
                pos_return_val = rolling.get("positive_return_rate")
                pos_str = f"positive {float(pos_return_val):.0f}%" if pos_return_val is not None else ""
                self.stat_cards[1].set_data(
                    "Target Hit Rate",
                    f"{target_hit_rate:.0f}%",
                    pos_str,
                    "good" if target_hit_rate >= 50 else "bad",
                )
            else:
                logger.info("target_hit_rate missing from rolling_summary")
                self.stat_cards[1].set_data("Target Hit Rate", "Not yet tracked", "No completed results yet", "warning")

            if cumulative_rows:
                total_return = float(cumulative_rows[-1]["cumulative_return"])
                self.stat_cards[2].set_data("Total Return", f"{total_return:+.2f}%", "cumulative", "good" if total_return >= 0 else "bad")
            else:
                logger.info("cumulative_rows unavailable for Total Return")
                self.stat_cards[2].set_data("Total Return", "Not yet tracked", "cumulative", "warning")

            best_return = rolling.get("best_week_return")
            if best_return is not None:
                best_str = f"{float(best_return):+.1f}%"
                self.stat_cards[3].set_data("Best Week", best_str, str(rolling.get("best_week", "-")), "good")
            else:
                logger.info("best_week_return missing")
                self.stat_cards[3].set_data("Best Week", "Not yet tracked", "-", "neutral")

            worst_return = rolling.get("worst_week_return")
            if worst_return is not None:
                worst_str = f"{float(worst_return):+.1f}%"
                self.stat_cards[4].set_data("Worst Week", worst_str, str(rolling.get("worst_week", "-")), "bad")
            else:
                logger.info("worst_week_return missing")
                self.stat_cards[4].set_data("Worst Week", "Not yet tracked", "-", "neutral")

            streak = rolling.get("streak_weeks")
            if streak is not None and int(streak) > 0:
                streak_direction = str(rolling.get("streak_direction", ""))
                streak_label = f"{int(streak)} wk {streak_direction}"
                self.stat_cards[5].set_data(
                    "Current Streak",
                    streak_label,
                    str(rolling.get("streak_basis", "latest momentum")).replace("_", " "),
                    "good" if streak_direction == "winning" else "warning",
                )
            else:
                self.stat_cards[5].set_data("Current Streak", "No streak", str(rolling.get("streak_basis", "latest momentum")).replace("_", " "), "neutral")

        self.line_chart.set_rows(cumulative_rows)
        self.feature_chart.set_items(model_meta.get("feature_importance", {}))
        self._populate_breakdown(self.sector_list, breakdowns.get("sector", []), formatter="{label}  {value:+.1f}%")
        self._populate_breakdown(self.vix_list, breakdowns.get("vix_bucket", []), formatter="{label}  {value:+.1f}%")
        self._populate_breakdown(self.tier_list, breakdowns.get("tier", []), formatter="{label}  {value:+.1f}%")

    def _populate_breakdown(self, widget: QListWidget, rows: list[dict[str, Any]], *, formatter: str) -> None:
        widget.clear()
        if not rows:
            widget.addItem("Not enough completed trades yet.")
            return
        for row in rows[:6]:
            label = str(row.get("label", row.get("bucket", row.get("name", ""))))
            value = float(row.get("avg_return", row.get("value", 0.0)))
            count = int(row.get("count", 0))
            item = QListWidgetItem(f"{formatter.format(label=label, value=value)}   ({count})")
            item.setForeground(QColor(COLORS["green"] if value >= 0 else COLORS["red"]))
            widget.addItem(item)


class HistoryView(QWidget):
    """Weekly history table with expandable picks."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        header = QHBoxLayout()
        self.title = QLabel("History")
        self.title.setFont(apple_font("display", 28, QFont.Weight.DemiBold))
        header.addWidget(self.title)
        header.addStretch(1)
        self.export_button = QPushButton("Export to CSV")
        style_secondary_button(self.export_button)
        self.export_button.clicked.connect(self._export_csv)
        header.addWidget(self.export_button)
        root.addLayout(header)

        self.table_card = QFrame()
        card_layout = QVBoxLayout(self.table_card)
        card_layout.setContentsMargins(20, 20, 20, 20)
        card_layout.setSpacing(12)
        card_layout.addWidget(SectionLabel("Scan History"))
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(
            ["Run / Pick", "Picks / Entry", "Target Hits / Target", "Hit Rate / Latest", "Avg Return / Status", "Best Pick / Sector"]
        )
        self.tree.setAlternatingRowColors(False)
        card_layout.addWidget(self.tree, 1)
        root.addWidget(self.table_card, 1)
        self._history_details: list[dict[str, Any]] = []
        self.apply_theme()

    def apply_theme(self) -> None:
        colors = theme_colors(self)
        self.setStyleSheet(f"background: {css_color(colors['window'])};")
        self.title.setStyleSheet(f"color: {css_color(colors['text'])};")
        self.table_card.setStyleSheet(
            f"background: {css_color(colors['base'])}; border: 1px solid {css_color(colors['border'])}; border-radius: 12px;"
        )
        apply_card_shadow(self.table_card, enabled=not is_dark_mode(self.table_card))
        self.tree.setStyleSheet(
            f"""
            QTreeWidget {{
                background: transparent;
                color: {css_color(colors['text'])};
                border: 0;
                font-size: 13px;
            }}
            QHeaderView::section {{
                background: transparent;
                color: {css_color(colors['text_secondary'])};
                border: 0;
                border-bottom: 1px solid {css_color(colors['separator'])};
                padding: 8px;
                font-weight: 600;
            }}
            """
        )

    def changeEvent(self, event) -> None:  # type: ignore[override]
        super().changeEvent(event)

    def update_data(self, state: dict[str, Any]) -> None:
        rows = state.get("history_rows") or state.get("history_weeks", [])
        self._history_details = state.get("history_details", [])
        self.tree.clear()
        for row in rows:
            item = QTreeWidgetItem(
                [
                    str(row.get("run_label") or row.get("week_label", "")),
                    str(row.get("picks", "-")),
                    str(row.get("target_hits", row.get("winners", "-"))),
                    str(row.get("target_hit_rate_label", row.get("hit_rate_label", row.get("hit_rate", "-")))),
                    str(row.get("avg_return_label", row.get("avg_return", "-"))),
                    str(row.get("best_pick", "-")),
                ]
            )
            run_id = row.get("run_id")
            for detail in [entry for entry in self._history_details if entry.get("run_id") == run_id]:
                child = QTreeWidgetItem(build_history_detail_cells(detail))
                item.addChild(child)
            self.tree.addTopLevelItem(item)
        self.tree.expandToDepth(0)
        for column in range(self.tree.columnCount()):
            self.tree.resizeColumnToContents(column)

    def _export_csv(self) -> None:
        if not self._history_details:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export History", "stock-predictor-history.csv", "CSV Files (*.csv)")
        if not path:
            return
        try:
            import pandas as pd

            pd.DataFrame(self._history_details).to_csv(path, index=False)
        except Exception:
            return
