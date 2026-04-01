from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from stock_predictor.config import AppConfig
from stock_predictor.output.pdf_report import generate_pdf_report
from tests.helpers import make_candidate


class FakeCanvas:
    def __init__(self, path: str, pagesize=None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def setFont(self, *args, **kwargs) -> None:
        return None

    def drawString(self, *args, **kwargs) -> None:
        return None

    def showPage(self) -> None:
        return None

    def drawImage(self, *args, **kwargs) -> None:
        return None

    def save(self) -> None:
        self.path.write_bytes(b"fake-pdf")


class PDFReportTests(unittest.TestCase):
    def test_generate_pdf_report_writes_report_and_latest_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report_dir = Path(tmpdir) / "reports"
            config = AppConfig(report_dir=report_dir, latest_report_pdf=report_dir / "latest.pdf")
            with patch("stock_predictor.output.pdf_report.canvas", SimpleNamespace(Canvas=FakeCanvas)), patch(
                "stock_predictor.output.pdf_report.letter",
                (612, 792),
            ), patch("stock_predictor.output.pdf_report._write_price_chart", return_value=None):
                path = generate_pdf_report(config, [make_candidate()], "Macro is supportive.", 60.0)
            self.assertIsNotNone(path)
            self.assertTrue(path.exists())
            self.assertTrue(config.latest_report_pdf.exists())


if __name__ == "__main__":
    unittest.main()
