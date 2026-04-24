from __future__ import annotations

import warnings
import unittest

from app import _configure_runtime_warnings, _missing_qt_message


class AppBootstrapTests(unittest.TestCase):
    def test_missing_qt_message_is_actionable(self) -> None:
        message = _missing_qt_message()

        self.assertIn("PySide6", message)
        self.assertIn("python3 -m pip install -r requirements.txt", message)
        self.assertIn("python3 app.py", message)

    def test_runtime_warning_filter_suppresses_requests_dependency_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _configure_runtime_warnings()
            try:
                from requests import RequestsDependencyWarning
            except Exception as exc:  # pragma: no cover
                self.fail(f"RequestsDependencyWarning unavailable: {exc}")
            warnings.warn(
                "urllib3 (2.6.3) doesn't match a supported version!",
                RequestsDependencyWarning,
            )

        self.assertEqual(caught, [])


if __name__ == "__main__":
    unittest.main()
