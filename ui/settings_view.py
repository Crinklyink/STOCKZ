"""Settings screen for the native Stock Predictor app."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QEvent, QTime, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QButtonGroup,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QTimeEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ui.widgets import (
    SectionLabel,
    ToggleSwitch,
    apple_font,
    apply_card_shadow,
    css_color,
    is_dark_mode,
    style_primary_button,
    style_secondary_button,
    theme_colors,
)


class SettingsView(QWidget):
    """Settings and controls for notifications, scheduling, and risk thresholds."""

    settings_saved = Signal(dict)
    test_email_requested = Signal(dict)
    retrain_requested = Signal()
    clear_cache_requested = Signal()
    view_logs_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent;")
        root.addWidget(scroll)

        container = QWidget()
        self.layout_main = QVBoxLayout(container)
        self.layout_main.setContentsMargins(24, 24, 24, 24)
        self.layout_main.setSpacing(16)
        scroll.setWidget(container)

        self.email_toggle = ToggleSwitch()
        self.email_field = QLineEdit()
        self.user_field = QLineEdit()
        self.password_field = QLineEdit()
        self.password_field.setEchoMode(QLineEdit.EchoMode.Password)
        self.show_password_button = QToolButton()
        self.show_password_button.setText("Show")
        self.show_password_button.clicked.connect(self._toggle_password)
        self.test_email_button = QPushButton("Test Email")
        self.test_email_button.clicked.connect(lambda: self.test_email_requested.emit(self.collect_settings()))

        self.sunday_time = QTimeEdit()
        self.sunday_time.setDisplayFormat("h:mm AP")
        self.wednesday_time = QTimeEdit()
        self.wednesday_time.setDisplayFormat("h:mm AP")
        self.auto_open_toggle = ToggleSwitch()
        self.retrain_toggle = ToggleSwitch()

        self.vix_slider = self._build_slider(25, 35)
        self.score_slider = self._build_slider(54, 70)
        self.rr_slider = self._build_slider(10, 20)
        self.max_picks_slider = self._build_slider(5, 10)
        self.vix_value = QLabel()
        self.score_value = QLabel()
        self.rr_value = QLabel()
        self.max_picks_value = QLabel()

        self.mini_radio = QRadioButton("Mini (69 tickers — fast)")
        self.full_radio = QRadioButton("Full (565 tickers — recommended)")
        self.us_market_radio = QRadioButton("US Market (~10,000 tickers — ex-Penny)")
        self.universe_group = QButtonGroup(self)
        self.universe_group.addButton(self.mini_radio)
        self.universe_group.addButton(self.full_radio)
        self.universe_group.addButton(self.us_market_radio)

        self.version_label = QLabel("Version: 3.0")
        self.stack_label = QLabel("Model stack: --")
        self.profile_label = QLabel("Profile: --")
        self.model_label = QLabel("Model trained: --")
        self.auc_label = QLabel("AUC: --")
        self.samples_label = QLabel("Samples: --")
        self.blend_label = QLabel("Blend: --")

        self.retrain_button = QPushButton("Retrain Model Now")
        self.clear_cache_button = QPushButton("Clear Cache")
        self.view_logs_button = QPushButton("View Logs")
        self.save_button = QPushButton("Save Settings")
        self.retrain_button.clicked.connect(self.retrain_requested.emit)
        self.clear_cache_button.clicked.connect(self.clear_cache_requested.emit)
        self.view_logs_button.clicked.connect(self.view_logs_requested.emit)
        self.save_button.clicked.connect(lambda: self.settings_saved.emit(self.collect_settings()))

        self._build_sections()
        self._wire_sliders()
        self.apply_theme()

    def _build_sections(self) -> None:
        self.notifications_card = self._section_card("Notifications")
        notifications_layout = QFormLayout(self.notifications_card)
        notifications_layout.addRow(self._toggle_row("Email alerts", self.email_toggle))
        notifications_layout.addRow("Email address", self.email_field)
        notifications_layout.addRow("SMTP user", self.user_field)
        password_row = QHBoxLayout()
        password_row.addWidget(self.password_field, 1)
        password_row.addWidget(self.show_password_button)
        notifications_layout.addRow("App password", password_row)
        notifications_layout.addRow("", self.test_email_button)

        self.schedule_card = self._section_card("Scan Schedule")
        schedule_layout = QFormLayout(self.schedule_card)
        schedule_layout.addRow("Sunday scan time", self.sunday_time)
        schedule_layout.addRow("Wednesday check-in", self.wednesday_time)
        schedule_layout.addRow(self._toggle_row("Auto-open app after scan", self.auto_open_toggle))
        schedule_layout.addRow(self._toggle_row("Retrain model weekly", self.retrain_toggle))

        self.risk_card = self._section_card("Risk Settings")
        risk_layout = QGridLayout(self.risk_card)
        self._add_slider_row(risk_layout, 0, "VIX cutoff (no picks above)", self.vix_slider, self.vix_value)
        self._add_slider_row(risk_layout, 1, "Minimum score threshold", self.score_slider, self.score_value)
        self._add_slider_row(risk_layout, 2, "Minimum R:R ratio", self.rr_slider, self.rr_value)
        self._add_slider_row(risk_layout, 3, "Max picks per week", self.max_picks_slider, self.max_picks_value)

        self.universe_card = self._section_card("Universe")
        universe_layout = QVBoxLayout(self.universe_card)
        universe_layout.addWidget(self.mini_radio)
        universe_layout.addWidget(self.full_radio)
        universe_layout.addWidget(self.us_market_radio)

        self.about_card = self._section_card("About")
        about_layout = QVBoxLayout(self.about_card)
        about_layout.addWidget(self.version_label)
        about_layout.addWidget(self.stack_label)
        about_layout.addWidget(self.profile_label)
        about_layout.addWidget(self.model_label)
        about_layout.addWidget(self.auc_label)
        about_layout.addWidget(self.samples_label)
        about_layout.addWidget(self.blend_label)
        actions = QHBoxLayout()
        actions.addWidget(self.retrain_button)
        actions.addWidget(self.clear_cache_button)
        actions.addWidget(self.view_logs_button)
        actions.addStretch(1)
        about_layout.addLayout(actions)

        for title, card in (
            ("Notifications", self.notifications_card),
            ("Scan Schedule", self.schedule_card),
            ("Risk Settings", self.risk_card),
            ("Universe", self.universe_card),
            ("About", self.about_card),
        ):
            self.layout_main.addWidget(SectionLabel(title))
            self.layout_main.addWidget(card)

        footer = QHBoxLayout()
        footer.addStretch(1)
        footer.addWidget(self.save_button)
        self.layout_main.addLayout(footer)
        self.layout_main.addStretch(1)

    def _section_card(self, _: str) -> QFrame:
        card = QFrame()
        card.setObjectName("settingsCard")
        return card

    def _toggle_row(self, text: str, toggle: ToggleSwitch) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel(text)
        label.setFont(apple_font("text", 14, QFont.Weight.Normal))
        layout.addWidget(label)
        layout.addStretch(1)
        layout.addWidget(toggle)
        return row

    def _add_slider_row(self, layout: QGridLayout, row: int, title: str, slider: QSlider, value_label: QLabel) -> None:
        label = QLabel(title)
        label.setFont(apple_font("text", 14, QFont.Weight.Normal))
        value_label.setFont(apple_font("rounded", 13, QFont.Weight.Medium))
        layout.addWidget(label, row, 0)
        layout.addWidget(slider, row, 1)
        layout.addWidget(value_label, row, 2)

    def _wire_sliders(self) -> None:
        self.vix_slider.valueChanged.connect(lambda value: self.vix_value.setText(str(value)))
        self.score_slider.valueChanged.connect(lambda value: self.score_value.setText(str(value)))
        self.rr_slider.valueChanged.connect(lambda value: self.rr_value.setText(f"{value / 10:.1f}"))
        self.max_picks_slider.valueChanged.connect(lambda value: self.max_picks_value.setText(str(value)))

    def _build_slider(self, minimum: int, maximum: int) -> QSlider:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        return slider

    def apply_theme(self) -> None:
        colors = theme_colors(self)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(f"{self.__class__.__name__} {{ background-color: {css_color(colors['window'])}; }}")
        card_style = (
            f"background: {css_color(colors['base'])}; "
            f"border: 1px solid rgba(255, 255, 255, 0.07); "
            "border-radius: 10px;"
        )
        for card in (self.notifications_card, self.schedule_card, self.risk_card, self.universe_card, self.about_card):
            card.setStyleSheet(card_style)
            apply_card_shadow(card, enabled=not is_dark_mode(card))
        for field in (self.email_field, self.user_field, self.password_field, self.sunday_time, self.wednesday_time):
            field.setFont(apple_font("text", 14))
            field.setStyleSheet(
                f"""
                background: {css_color(colors['alt_base'])};
                color: {css_color(colors['text'])};
                border: 1px solid rgba(255, 255, 255, 0.07);
                border-radius: 6px;
                padding: 8px;
                min-height: 34px;
                selection-background-color: {css_color('#007AFF')};
                """
            )
        for label in (
            self.version_label,
            self.stack_label,
            self.profile_label,
            self.model_label,
            self.auc_label,
            self.samples_label,
            self.blend_label,
            self.vix_value,
            self.score_value,
            self.rr_value,
            self.max_picks_value,
        ):
            label.setFont(apple_font("text", 13))
            label.setStyleSheet(f"color: {css_color(colors['text_secondary'])};")
        for radio in (self.mini_radio, self.full_radio, self.us_market_radio):
            radio.setFont(apple_font("text", 14))
            radio.setStyleSheet(f"color: {css_color(colors['text'])};")
        style_secondary_button(self.test_email_button)
        style_secondary_button(self.retrain_button)
        style_secondary_button(self.clear_cache_button)
        style_secondary_button(self.view_logs_button)
        style_primary_button(self.save_button)
        self.show_password_button.setFont(apple_font("text", 13, QFont.Weight.Medium))
        self.show_password_button.setStyleSheet(
            f"""
            QToolButton {{
                background: transparent;
                color: #007AFF;
                border: 1px solid rgba(255, 255, 255, 0.07);
                border-radius: 6px;
                padding: 6px 10px;
            }}
            QToolButton:hover {{
                background: {css_color(colors['button_hover'])};
            }}
            """
        )

    def changeEvent(self, event) -> None:  # type: ignore[override]
        super().changeEvent(event)

    def update_data(self, state: dict[str, Any]) -> None:
        settings = state.get("settings", {})
        self.email_toggle.setChecked(bool(settings.get("email_enabled", False)))
        self.email_field.setText(str(settings.get("alert_email", "")))
        self.user_field.setText(str(settings.get("smtp_user", "")))
        self.password_field.setText(str(settings.get("smtp_pass", "")))
        self.sunday_time.setTime(_parse_time(str(settings.get("sunday_time", "18:00"))))
        self.wednesday_time.setTime(_parse_time(str(settings.get("wednesday_time", "18:00"))))
        self.auto_open_toggle.setChecked(bool(settings.get("auto_open_app", True)))
        self.retrain_toggle.setChecked(bool(settings.get("retrain_weekly", True)))
        self.vix_slider.setValue(int(round(float(settings.get("vix_cutoff", 30)))))
        self.score_slider.setValue(int(round(float(settings.get("min_score_threshold", 54)))))
        self.rr_slider.setValue(int(round(float(settings.get("min_rr_ratio", 1.0)) * 10)))
        self.max_picks_slider.setValue(int(round(float(settings.get("max_picks", 5)))))
        universe = str(settings.get("universe", "full"))
        self.mini_radio.setChecked(universe == "mini")
        self.us_market_radio.setChecked(universe == "us_market")
        self.full_radio.setChecked(universe not in {"mini", "us_market"})

        model_meta = state.get("model_metadata", {})
        self.stack_label.setText(f"Model stack: {model_meta.get('model_stack', 'XGBoost')}")
        self.profile_label.setText(f"Profile: {model_meta.get('profile_label', '--')}")
        self.model_label.setText(f"Model trained: {model_meta.get('trained_at', '--')[:16].replace('T', ' ')}")
        self.auc_label.setText(f"AUC: {float(model_meta.get('auc', 0.0)):.3f}")
        self.samples_label.setText(f"Samples: {int(model_meta.get('training_samples', 0)):,}")
        weights = model_meta.get("ensemble_weights", {}) if isinstance(model_meta.get("ensemble_weights"), dict) else {}
        if weights:
            xgb_weight = float(weights.get("xgb", 0.0))
            lgbm_weight = float(weights.get("lgbm", 0.0))
            self.blend_label.setText(f"Blend: XGB {xgb_weight:.1f} · LGBM {lgbm_weight:.1f}")
        else:
            self.blend_label.setText("Blend: XGB only")

    def collect_settings(self) -> dict[str, Any]:
        return {
            "email_enabled": self.email_toggle.isChecked(),
            "alert_email": self.email_field.text().strip(),
            "smtp_user": self.user_field.text().strip(),
            "smtp_pass": self.password_field.text(),
            "sunday_time": self.sunday_time.time().toString("HH:mm"),
            "wednesday_time": self.wednesday_time.time().toString("HH:mm"),
            "auto_open_app": self.auto_open_toggle.isChecked(),
            "retrain_weekly": self.retrain_toggle.isChecked(),
            "vix_cutoff": self.vix_slider.value(),
            "min_score_threshold": self.score_slider.value(),
            "min_rr_ratio": self.rr_slider.value() / 10.0,
            "max_picks": self.max_picks_slider.value(),
            "universe": "mini" if self.mini_radio.isChecked() else "us_market" if self.us_market_radio.isChecked() else "full",
        }

    def _toggle_password(self) -> None:
        if self.password_field.echoMode() == QLineEdit.EchoMode.Password:
            self.password_field.setEchoMode(QLineEdit.EchoMode.Normal)
            self.show_password_button.setText("Hide")
        else:
            self.password_field.setEchoMode(QLineEdit.EchoMode.Password)
            self.show_password_button.setText("Show")


def _parse_time(value: str) -> QTime:
    parsed = QTime.fromString(value, "HH:mm")
    return parsed if parsed.isValid() else QTime(18, 0)
