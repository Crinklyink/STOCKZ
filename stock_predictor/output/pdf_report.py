"""PDF report generator for weekly pick summaries."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:  # pragma: no cover - optional dependency
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except Exception:  # pragma: no cover
    letter = None
    canvas = None

from stock_predictor.config import AppConfig
from stock_predictor.scoring.composite import CandidateScore


def generate_pdf_report(
    config: AppConfig,
    candidates: Sequence[CandidateScore],
    macro_summary: str,
    last_week_accuracy: float | None,
) -> Path | None:
    """Generate a Sunday PDF report with picks and charts."""

    if canvas is None or letter is None:
        return None
    report_path = config.report_dir / f"{datetime.now().date().isoformat()}.pdf"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(report_path), pagesize=letter)
    width, height = letter
    y = height - 40
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(40, y, "Elite Weekly Stock Predictor")
    y -= 24
    pdf.setFont("Helvetica", 10)
    pdf.drawString(40, y, f"Generated: {datetime.now().isoformat(timespec='minutes')}")
    y -= 16
    if last_week_accuracy is not None:
        pdf.drawString(40, y, f"Last week hit rate: {last_week_accuracy:.1f}%")
        y -= 16
    for line in split_lines(f"Macro: {macro_summary}", 110):
        pdf.drawString(40, y, line)
        y -= 12
    y -= 8
    for candidate in candidates:
        if y < 180:
            pdf.showPage()
            y = height - 40
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(
            40,
            y,
            (
                f"{candidate.ticker} | {candidate.company_name} | "
                f"Score {candidate.final_score:.1f} +/- {candidate.score_uncertainty:.1f}"
            ),
        )
        y -= 14
        pdf.setFont("Helvetica", 9)
        pdf.drawString(
            40,
            y,
            (
                f"Entry {candidate.current_price:.2f} | Stop {candidate.stop_loss:.2f} | "
                f"TP2 {candidate.targets['tp2']:.2f} | Confidence {candidate.confidence_label}"
            ),
        )
        y -= 12
        for line in split_lines(f"Why this pick: {candidate.ai_explanation}", 115):
            pdf.drawString(40, y, line)
            y -= 11
        for line in split_lines("Notes: " + " | ".join(candidate.notes or ["No additional notes"]), 115):
            pdf.drawString(40, y, line)
            y -= 11
        chart_path = _write_price_chart(config, candidate)
        if chart_path:
            pdf.drawImage(str(chart_path), 40, y - 80, width=240, height=75, preserveAspectRatio=True, mask="auto")
            y -= 88
        else:
            y -= 10
    pdf.save()
    shutil.copyfile(report_path, config.latest_report_pdf)
    return report_path


def _write_price_chart(config: AppConfig, candidate: CandidateScore) -> Path | None:
    if plt is None:
        return None
    price_chart = candidate.diagnostics.get("price_chart", [])
    if not price_chart:
        return None
    out_path = config.report_dir / f"{candidate.ticker}_chart.png"
    x = [row["date"] for row in price_chart]
    y = [row["close"] for row in price_chart]
    plt.figure(figsize=(4, 1.4))
    plt.plot(x, y, color="green", linewidth=1.5)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    return out_path


def split_lines(text: str, width: int) -> list[str]:
    words = text.split()
    lines = []
    current = []
    for word in words:
        if len(" ".join(current + [word])) > width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines
