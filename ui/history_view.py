"""Performance and history screens for the native Stock Predictor app."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

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


class BreakdownSection(QWidget):
    """A titled group of plain-text breakdown rows — no scroll, no border."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: list[QLabel] = []
        # Shield against parent panel border cascading into this widget
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("background: transparent; border: none;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._caption = QLabel(title)
        self._caption.setFont(apple_font("text", 10, QFont.Weight.Bold))
        layout.addWidget(self._caption)

        self._rows_layout = QVBoxLayout()
        self._rows_layout.setContentsMargins(0, 4, 0, 0)
        self._rows_layout.setSpacing(1)
        layout.addLayout(self._rows_layout)

    def populate(self, rows: list[dict[str, Any]], colors: dict) -> None:
        for lbl in self._rows:
            lbl.deleteLater()
        self._rows.clear()

        if not rows:
            lbl = QLabel("Not enough data yet")
            lbl.setFont(apple_font("text", 12, QFont.Weight.Normal))
            lbl.setStyleSheet(
                "color: " + css_color(colors["text_tertiary"]) + "; background: transparent;"
            )
            self._rows_layout.addWidget(lbl)
            self._rows.append(lbl)
            return

        dim = css_color(colors["text_tertiary"])
        for row in rows[:6]:
            label_text = str(row.get("label", row.get("bucket", row.get("name", "")))).strip()
            if not label_text:
                continue

            value = float(row.get("avg_return", row.get("value", 0.0)))
            count = int(row.get("count", 0))
            value_color = COLORS["green"] if value >= 0 else COLORS["red"]
            html = (
                f"<span style='color:{value_color}'>{label_text}&nbsp;&nbsp;&nbsp;"
                f"{value:+.1f}%</span>"
                f"&nbsp;&nbsp;<span style='color:{dim}'>({count})</span>"
            )
            lbl = QLabel()
            lbl.setText(html)
            lbl.setTextFormat(Qt.TextFormat.RichText)
            lbl.setFont(apple_font("text", 12, QFont.Weight.Normal))
            lbl.setStyleSheet("background: transparent; padding: 3px 0;")
            self._rows_layout.addWidget(lbl)
            self._rows.append(lbl)

    def set_title(self, title: str) -> None:
        self._caption.setText(title)

    def apply_theme(self, colors: dict) -> None:
        self._caption.setStyleSheet(
            "color: " + css_color(colors["text_tertiary"]) + "; letter-spacing: 1.5px; background: transparent;"
        )


class PerformanceView(QWidget):
    """Performance analytics screen."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 28, 28, 28)
        root.setSpacing(16)

        # Page title
        self.page_title = QLabel("Performance")
        self.page_title.setFont(apple_font("display", 24, QFont.Weight.Bold))
        root.addWidget(self.page_title)

        # Stat cards — 2 rows × 3 columns
        self.stat_cards = [
            StatCard("Weeks", "--", "tracked"),
            StatCard("Hit Rate", "--", "target hit rate"),
            StatCard("Total Return", "--", "cumulative"),
            StatCard("Best Week", "--", "top result"),
            StatCard("Worst Week", "--", "weakest"),
            StatCard("Streak", "--", "momentum"),
        ]
        stat_grid = QGridLayout()
        stat_grid.setHorizontalSpacing(12)
        stat_grid.setVerticalSpacing(12)
        for index, card in enumerate(self.stat_cards):
            stat_grid.addWidget(card, index // 3, index % 3)
        root.addLayout(stat_grid)

        # Middle section: chart + breakdown
        middle_row = QHBoxLayout()
        middle_row.setSpacing(12)

        # Cumulative Return chart
        self.line_panel = QFrame()
        line_layout = QVBoxLayout(self.line_panel)
        line_layout.setContentsMargins(20, 20, 20, 20)
        line_layout.setSpacing(12)
        line_layout.addWidget(SectionLabel("Cumulative Return"))
        self.line_chart = CumulativeReturnChart()
        line_layout.addWidget(self.line_chart)
        middle_row.addWidget(self.line_panel, 2)

        # Performance Breakdown panel — plain labels, no scroll
        self.side_panel = QFrame()
        self.side_panel.setObjectName("sidePanel")
        side_layout = QVBoxLayout(self.side_panel)
        side_layout.setContentsMargins(20, 20, 20, 20)
        side_layout.setSpacing(16)
        side_layout.addWidget(SectionLabel("Breakdown"))
        self.sector_section = BreakdownSection("SECTORS")
        self.vix_section = BreakdownSection("VIX LEVELS")
        self.tier_section = BreakdownSection("SCORE TIERS")
        for section in (self.sector_section, self.vix_section, self.tier_section):
            side_layout.addWidget(section)
        side_layout.addStretch(1)
        middle_row.addWidget(self.side_panel, 1)
        root.addLayout(middle_row)

        # Feature Importance panel
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
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"{self.__class__.__name__} {{ background-color: {css_color(colors['window'])}; }}"
        )
        self.page_title.setStyleSheet(
            f"color: {css_color(colors['text'])}; background: transparent;"
        )
        base_bg = css_color(colors["base"])
        border_c = css_color(colors["border"])
        # Use object-name scoped rules so border never cascades to children
        self.line_panel.setStyleSheet(
            f"QFrame {{ background: {base_bg}; border: 1px solid {border_c}; border-radius: 10px; }}"
            f" QWidget {{ background: transparent; border: none; }}"
        )
        self.side_panel.setStyleSheet(
            f"QFrame#sidePanel {{ background: {base_bg}; border: 1px solid {border_c}; border-radius: 10px; }}"
            f" QFrame#sidePanel > QWidget {{ background: transparent; border: none; }}"
            f" QLabel {{ background: transparent; border: none; }}"
        )
        self.feature_panel.setStyleSheet(
            f"QFrame {{ background: {base_bg}; border: 1px solid {border_c}; border-radius: 10px; }}"
            f" QWidget {{ background: transparent; border: none; }}"
        )
        for panel in (self.line_panel, self.side_panel, self.feature_panel):
            apply_card_shadow(panel, enabled=not is_dark_mode(panel))
        for section in (self.sector_section, self.vix_section, self.tier_section):
            section.apply_theme(colors)

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
        colors = theme_colors(self)

        if not rolling:
            logger.info("rolling_summary unavailable. Dashboard showing empty states.")
            pending_text = f"{total_weeks} pending" if total_weeks else "None yet"
            self.stat_cards[0].set_data("Weeks", str(total_weeks), pending_text, "neutral")
            self.stat_cards[1].set_data("Hit Rate", "Not yet tracked", "no data", "warning")
            self.stat_cards[2].set_data("Total Return", "Not yet tracked", "cumulative", "warning")
            self.stat_cards[3].set_data("Best Week", "Not yet tracked", "-", "neutral")
            self.stat_cards[4].set_data("Worst Week", "Not yet tracked", "-", "neutral")
            self.stat_cards[5].set_data("Streak", "Not yet tracked", "no data", "neutral")
        else:
            self.stat_cards[0].set_data("Weeks", str(total_weeks), f"{completed_count} completed", "neutral")

            target_hit_rate = rolling.get("target_hit_rate")
            if target_hit_rate is None:
                target_hit_rate = rolling.get("win_rate")

            if target_hit_rate is not None:
                target_hit_rate = float(target_hit_rate)
                pos_return_val = rolling.get("positive_return_rate")
                pos_str = f"positive {float(pos_return_val):.0f}%" if pos_return_val is not None else ""
                self.stat_cards[1].set_data(
                    "Hit Rate",
                    f"{target_hit_rate:.0f}%",
                    pos_str,
                    "good" if target_hit_rate >= 50 else "bad",
                )
            else:
                logger.info("target_hit_rate missing from rolling_summary")
                self.stat_cards[1].set_data("Hit Rate", "Not yet tracked", "no data", "warning")

            if cumulative_rows:
                total_return = float(cumulative_rows[-1]["cumulative_return"])
                self.stat_cards[2].set_data(
                    "Total Return", f"{total_return:+.2f}%", "cumulative",
                    "good" if total_return >= 0 else "bad"
                )
            else:
                logger.info("cumulative_rows unavailable for Total Return")
                self.stat_cards[2].set_data("Total Return", "Not yet tracked", "cumulative", "warning")

            best_return = rolling.get("best_week_return")
            if best_return is not None:
                self.stat_cards[3].set_data(
                    "Best Week", f"{float(best_return):+.1f}%",
                    str(rolling.get("best_week", "-")), "good"
                )
            else:
                logger.info("best_week_return missing")
                self.stat_cards[3].set_data("Best Week", "Not yet tracked", "-", "neutral")

            worst_return = rolling.get("worst_week_return")
            if worst_return is not None:
                self.stat_cards[4].set_data(
                    "Worst Week", f"{float(worst_return):+.1f}%",
                    str(rolling.get("worst_week", "-")), "bad"
                )
            else:
                logger.info("worst_week_return missing")
                self.stat_cards[4].set_data("Worst Week", "Not yet tracked", "-", "neutral")

            streak = rolling.get("streak_weeks")
            if streak is not None and int(streak) > 0:
                streak_direction = str(rolling.get("streak_direction", ""))
                self.stat_cards[5].set_data(
                    "Streak",
                    f"{int(streak)} wk {streak_direction}",
                    str(rolling.get("streak_basis", "latest momentum")).replace("_", " "),
                    "good" if streak_direction == "winning" else "warning",
                )
            else:
                self.stat_cards[5].set_data(
                    "Streak", "No streak",
                    str(rolling.get("streak_basis", "latest momentum")).replace("_", " "),
                    "neutral"
                )

        self.line_chart.set_rows(cumulative_rows)
        self.feature_chart.set_items(model_meta.get("feature_importance", {}))
        self.sector_section.set_title("SECTORS")
        self.sector_section.populate(breakdowns.get("sector", []), colors)
        regime_rows = []
        for regime, metrics in (model_meta.get("regime_summary", {}) or {}).items():
            avg_return = float(metrics.get("average_return", 0.0))
            if abs(avg_return) <= 1.0:
                avg_return *= 100.0
            regime_rows.append(
                {
                    "label": str(regime).replace("_", " ").title(),
                    "avg_return": avg_return,
                    "count": int(metrics.get("weeks", 0)),
                }
            )
        if regime_rows:
            self.vix_section.set_title("REGIME HISTORY")
            self.vix_section.populate(regime_rows, colors)
        else:
            self.vix_section.set_title("VIX LEVELS")
            self.vix_section.populate(breakdowns.get("vix_bucket", []), colors)
        feature_health_rows = []
        for feature, record in (model_meta.get("feature_health", {}) or {}).items():
            feature_health_rows.append(
                {
                    "label": f"{str(feature).replace('_', ' ').title()} · {record.get('status', 'ACTIVE')}",
                    "avg_return": float(record.get("hit_rate", 0.0)) * 100.0 - 50.0,
                    "count": int(record.get("sample_count", 0)),
                }
            )
        if feature_health_rows:
            self.tier_section.set_title("FEATURE HEALTH")
            self.tier_section.populate(feature_health_rows, colors)
        else:
            self.tier_section.set_title("SCORE TIERS")
            self.tier_section.populate(breakdowns.get("tier", []), colors)


class HistoryView(QWidget):
    """Weekly history table with expandable picks."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 28, 28, 28)
        root.setSpacing(16)

        header = QHBoxLayout()
        self.title = QLabel("History")
        self.title.setFont(apple_font("display", 24, QFont.Weight.Bold))
        header.addWidget(self.title)
        header.addStretch(1)
        self.export_button = QPushButton("Export to CSV")
        style_secondary_button(self.export_button)
        self.export_button.clicked.connect(self._export_csv)
        header.addWidget(self.export_button)
        root.addLayout(header)

        self.table_card = QFrame()
        card_layout = QVBoxLayout(self.table_card)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.setSpacing(0)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Date", "Ticker", "Entry", "Target", "Status", "Sector"])
        self.tree.setRootIsDecorated(True)
        self.tree.setIndentation(20)
        self.tree.setUniformRowHeights(True)
        card_layout.addWidget(self.tree, 1)
        root.addWidget(self.table_card, 1)
        self._history_details: list[dict[str, Any]] = []
        self.apply_theme()

    def apply_theme(self) -> None:
        colors = theme_colors(self)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"{self.__class__.__name__} {{ background-color: {css_color(colors['window'])}; }}"
        )
        self.title.setStyleSheet(f"color: {css_color(colors['text'])}; background: transparent;")
        self.table_card.setStyleSheet(
            f"background: {css_color(colors['base'])}; "
            f"border: 1px solid {css_color(colors['border'])}; "
            "border-radius: 10px;"
        )
        apply_card_shadow(self.table_card, enabled=not is_dark_mode(self.table_card))
        self.tree.setAlternatingRowColors(True)
        alt_bg = css_color(colors["alt_base"], 0.4)
        text_color = css_color(colors["text"])
        secondary = css_color(colors["text_secondary"])
        border_col = css_color(colors["border"])
        hover = css_color(colors["hover"])
        sel = css_color(colors["selection"])
        tertiary = css_color(colors["text_tertiary"])
        base = css_color(colors["base"])
        self.tree.setStyleSheet(
            f"""
            QTreeWidget {{
                background: {base};
                alternate-background-color: {alt_bg};
                color: {text_color};
                border: 0;
                border-radius: 10px;
                font-size: 13px;
                outline: none;
            }}
            QTreeWidget::item {{
                padding: 8px 12px;
                border: none;
            }}
            QTreeWidget::item:hover {{
                background: {hover};
            }}
            QTreeWidget::item:selected {{
                background: {sel};
                color: {text_color};
            }}
            QHeaderView::section {{
                background: {base};
                color: {secondary};
                border: none;
                border-bottom: 1px solid {border_col};
                padding: 12px 12px;
                font-weight: 700;
                font-size: 11px;
            }}
            QTreeWidget::branch {{
                background: transparent;
                border: none;
            }}
            QScrollBar:vertical {{
                background: transparent;
                width: 6px;
                border: none;
            }}
            QScrollBar::handle:vertical {{
                background: {tertiary};
                border-radius: 3px;
                min-height: 30px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: transparent;
            }}
            """
        )

    def changeEvent(self, event) -> None:  # type: ignore[override]
        super().changeEvent(event)

    def update_data(self, state: dict[str, Any]) -> None:
        rows = state.get("history_rows") or state.get("history_weeks", [])
        self._history_details = state.get("history_details", [])
        self.tree.clear()
        colors = theme_colors(self)
        parent_font = apple_font("text", 13, QFont.Weight.DemiBold)
        child_font = apple_font("text", 13, QFont.Weight.Normal)

        for row in rows:
            picks_count = str(row.get("picks", "-"))
            hit_rate = row.get("target_hit_rate_label", row.get("hit_rate_label", row.get("hit_rate", "")))
            avg_ret = row.get("avg_return_label", row.get("avg_return", ""))
            run_label = str(row.get("run_label") or row.get("week_label", ""))

            item = QTreeWidgetItem([
                run_label,
                f"{picks_count} picks",
                "",
                "",
                str(hit_rate) if hit_rate else "Pending",
                str(avg_ret) if avg_ret else "",
            ])
            for col in range(6):
                item.setFont(col, parent_font)
            item.setForeground(0, QColor(css_color(colors["text"])))
            item.setForeground(1, QColor(COLORS["blue"]))

            run_id = row.get("run_id")
            for detail in [e for e in self._history_details if e.get("run_id") == run_id]:
                ticker = detail.get("ticker", "")
                entry_price = detail.get("entry_price")
                target_price = detail.get("target_price")
                realized = detail.get("realized_return_pct")

                entry_str = f"${float(entry_price):,.2f}" if entry_price is not None else "–"
                target_str = f"${float(target_price):,.2f}" if target_price is not None else "–"

                if realized is not None:
                    status = "Hit" if detail.get("hit_target") else f"{float(realized):+.1f}%"
                else:
                    status = "Pending"

                sector = str(detail.get("sector", ""))
                child = QTreeWidgetItem(["", ticker, entry_str, target_str, status, sector])
                for col in range(6):
                    child.setFont(col, child_font)

                child.setForeground(1, QColor(css_color(colors["text"])))
                if realized is not None:
                    if detail.get("hit_target"):
                        child.setForeground(4, QColor(COLORS["green"]))
                    elif float(realized) >= 0:
                        child.setForeground(4, QColor(COLORS["blue"]))
                    else:
                        child.setForeground(4, QColor(COLORS["red"]))
                else:
                    child.setForeground(4, QColor(css_color(colors["text_tertiary"])))
                child.setForeground(5, QColor(css_color(colors["text_secondary"])))
                item.addChild(child)

            self.tree.addTopLevelItem(item)

        self.tree.expandToDepth(0)
        for column in range(self.tree.columnCount()):
            self.tree.resizeColumnToContents(column)

    def _export_csv(self) -> None:
        if not self._history_details:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export History", "stock-predictor-history.csv", "CSV Files (*.csv)"
        )
        if not path:
            return
        try:
            import pandas as pd
            pd.DataFrame(self._history_details).to_csv(path, index=False)
        except Exception:
            return
