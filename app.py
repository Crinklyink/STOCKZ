"""Native PySide6 desktop application entry point for Stock Predictor."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import QApplication

warnings.filterwarnings(
    "ignore",
    message=r".*urllib3.*doesn't match a supported version.*",
)
try:  # pragma: no cover - optional dependency warning class
    from requests import RequestsDependencyWarning

    warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
except Exception:
    pass

from app_window import StockPredictorWindow
from stock_predictor.config import get_resource_path

def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Stock Predictor")
    app.setApplicationDisplayName("Stock Predictor")
    app.setDesktopFileName("Stock Predictor")

    icon_path = get_resource_path("assets/icon.png")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    app.setFont(QFont(".AppleSystemUIFont", 13))
    app.setStyleSheet(
        """
        QToolTip {
            padding: 6px 8px;
            border-radius: 6px;
        }
        """
    )

    smoke_test = "--smoke-test" in sys.argv
    window = StockPredictorWindow(enable_tray=not smoke_test)
    window.show()

    if smoke_test:
        app.processEvents()
        window.close()
        return 0

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
