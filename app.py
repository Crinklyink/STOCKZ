"""Native PySide6 desktop application entry point for Stock Predictor."""

from __future__ import annotations

import sys
import warnings


def _configure_runtime_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r".*urllib3.*doesn't match a supported version.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Warning: You are sending unauthenticated requests to the HF Hub.*",
    )
    try:  # pragma: no cover - optional dependency warning class
        from requests import RequestsDependencyWarning

        warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
    except Exception:
        pass


def _missing_qt_message() -> str:
    return (
        "PySide6 is required to launch the Stock Predictor desktop app.\n\n"
        "Install it with:\n"
        "  python3 -m pip install -r requirements.txt\n\n"
        "Then run:\n"
        "  python3 app.py\n"
    )


def _missing_dependency_message(package: str) -> str:
    return (
        f"{package} is required to launch the Stock Predictor desktop app.\n\n"
        "Install the app dependencies with:\n"
        "  python3 -m pip install -r requirements.txt\n\n"
        "Then run:\n"
        "  python3 app.py\n"
    )


_configure_runtime_warnings()

try:
    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QFont, QIcon
    from PySide6.QtWidgets import QApplication
except ModuleNotFoundError as exc:
    if exc.name == "PySide6":
        print(_missing_qt_message(), file=sys.stderr)
        raise SystemExit(1)
    raise


from stock_predictor.config import get_resource_path

def main() -> int:
    try:
        from app_window import StockPredictorWindow
    except ModuleNotFoundError as exc:
        print(_missing_dependency_message(exc.name or "A Python package"), file=sys.stderr)
        return 1

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
