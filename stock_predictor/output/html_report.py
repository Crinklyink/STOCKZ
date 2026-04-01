"""Standalone weekly HTML report generation."""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Iterable
import webbrowser

from stock_predictor.config import AppConfig


def generate_weekly_html_report(
    config: AppConfig,
    payload: dict,
    *,
    last_week_summary: dict | None = None,
    rolling_summary: dict | None = None,
    weekly_rows: Iterable[dict] | None = None,
    auto_open: bool = False,
) -> Path:
    html_path = config.weekly_report_html
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html = render_weekly_html_report(
        payload,
        last_week_summary=last_week_summary,
        rolling_summary=rolling_summary,
        weekly_rows=list(weekly_rows or []),
    )
    html_path.write_text(html, encoding="utf-8")
    if auto_open:
        webbrowser.open(html_path.as_uri())
    return html_path


def render_weekly_html_report(
    payload: dict,
    *,
    last_week_summary: dict | None = None,
    rolling_summary: dict | None = None,
    weekly_rows: list[dict] | None = None,
) -> str:
    scan_summary = payload.get("scan_summary", {})
    official_picks = list(payload.get("selected", []))
    display_candidates = list(payload.get("display_candidates", official_picks))
    showing_watchlist = not official_picks
    cards = display_candidates if showing_watchlist else official_picks
    header_date = scan_summary.get("date") or datetime.now().strftime("%Y-%m-%d")
    regime = escape(str(scan_summary.get("regime", "unknown")).upper())
    vix = float(scan_summary.get("vix", 0.0))
    spy_week = float(scan_summary.get("spy_week_return", 0.0))
    selection_warning = escape(str(scan_summary.get("selection_warning", "")))
    macro_summary = escape(str(payload.get("macro_summary", "")))
    chart_html = _build_weekly_chart(weekly_rows or [])
    results_html = _build_last_week_results(last_week_summary)
    rolling_html = _build_rolling_summary(rolling_summary)
    cards_html = "\n".join(_build_pick_card(candidate) for candidate in cards[:5])
    empty_state = (
        "<div class='empty-state'>"
        "<h2>No official picks this week</h2>"
        f"<p>{selection_warning or 'The regime filter blocked official entries this week.'}</p>"
        "<p>Watchlist names are shown below for monitoring only.</p>"
        "</div>"
        if showing_watchlist
        else ""
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Weekly Stock Report</title>
  <style>
    :root {{
      --bg: #f4f6ef;
      --ink: #17211c;
      --muted: #5d6b63;
      --line: #d2d9d0;
      --panel: #ffffff;
      --green: #1f7a3f;
      --yellow: #a67c00;
      --red: #b13a2f;
      --dark: #0f3b2b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #eef3e7 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    .wrap {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 28px 20px 40px;
    }}
    .hero {{
      background: radial-gradient(circle at top left, #ffffff 0%, #ecf2ea 70%);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 28px;
      box-shadow: 0 18px 50px rgba(17, 35, 24, 0.08);
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: 34px;
      line-height: 1.05;
      letter-spacing: -0.03em;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      margin-top: 18px;
      color: var(--muted);
      font-size: 15px;
    }}
    .badge {{
      display: inline-block;
      padding: 7px 12px;
      border-radius: 999px;
      font-weight: 700;
      font-size: 12px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      background: #e8f3eb;
      color: var(--green);
    }}
    .warning {{
      margin-top: 16px;
      border-left: 4px solid var(--yellow);
      padding: 12px 14px;
      background: #fff8de;
      border-radius: 12px;
      color: #6e5600;
    }}
    .subcopy {{
      margin-top: 14px;
      color: var(--muted);
      line-height: 1.5;
    }}
    .section {{
      margin-top: 26px;
    }}
    .section h2 {{
      margin: 0 0 14px;
      font-size: 22px;
      letter-spacing: -0.02em;
    }}
    .grid {{
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px;
      box-shadow: 0 10px 30px rgba(17, 35, 24, 0.06);
    }}
    .ticker {{
      font-size: 28px;
      font-weight: 800;
      letter-spacing: -0.03em;
    }}
    .company {{
      color: var(--muted);
      margin-top: 2px;
      font-size: 14px;
    }}
    .price-row {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      margin-top: 14px;
      font-size: 14px;
    }}
    .price-row strong {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 3px;
    }}
    .signal-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 14px;
    }}
    .dot {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 9px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      background: #eef3ef;
      color: var(--dark);
    }}
    .dot.green {{ background: #e3f5e8; color: var(--green); }}
    .dot.yellow {{ background: #fff6d8; color: var(--yellow); }}
    .dot.red {{ background: #fde9e6; color: var(--red); }}
    .why {{
      margin-top: 14px;
      line-height: 1.55;
      color: #304139;
      font-size: 14px;
    }}
    .empty-state {{
      margin-bottom: 18px;
      background: #fff7e8;
      border: 1px solid #f0d9aa;
      border-radius: 18px;
      padding: 18px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border-radius: 16px;
      overflow: hidden;
      border: 1px solid var(--line);
    }}
    th, td {{
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      font-size: 14px;
    }}
    th {{
      background: #eff3ef;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .chart {{
      display: grid;
      gap: 10px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 68px 1fr 74px;
      gap: 10px;
      align-items: center;
      font-size: 13px;
    }}
    .bar-track {{
      height: 12px;
      border-radius: 999px;
      background: #e6ece5;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, #2a7b4f, #5fb36e);
    }}
    @media (max-width: 700px) {{
      .hero h1 {{ font-size: 28px; }}
      .wrap {{ padding: 18px 14px 28px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <span class="badge">{'Watchlist Only' if showing_watchlist else 'Official Weekly Picks'}</span>
      <h1>{escape(header_date)} Weekly Stock Report</h1>
        <div class="meta">
        <span>VIX: {vix:.1f}</span>
        <span>SPY week: {spy_week:+.1f}%</span>
        <span>Regime: {regime}</span>
        <span>Above score threshold: {int(scan_summary.get('qualified_count', 0))}</span>
      </div>
      {f"<div class='warning'>{selection_warning}</div>" if selection_warning else ""}
      <div class="subcopy">{macro_summary}</div>
    </section>

    <section class="section">
      <h2>{"Watchlist" if showing_watchlist else "This Week's Picks"}</h2>
      {empty_state}
      <div class="grid">
        {cards_html}
      </div>
    </section>

    <section class="section">
      <h2>Last Week</h2>
      {results_html}
    </section>

    <section class="section">
      <h2>8-Week Track Record</h2>
      {rolling_html}
      {chart_html}
    </section>
  </div>
</body>
</html>"""


def _build_pick_card(candidate: dict) -> str:
    ticker = escape(str(candidate.get("ticker", "")))
    company = escape(str(candidate.get("company_name", ticker)))
    current = float(candidate.get("current_price", 0.0))
    target = float(candidate.get("targets", {}).get("tp2", current))
    stop = float(candidate.get("stop_loss", current))
    upside = ((target / current) - 1.0) * 100.0 if current else 0.0
    downside = ((stop / current) - 1.0) * 100.0 if current else 0.0
    signals = [
        _signal_dot("Tech", float(candidate.get("technical_score", 0.0))),
        _signal_dot("RS", float(candidate.get("rs_score", 0.0))),
        _signal_dot("Vol", float(candidate.get("volume_momentum_score", 0.0))),
        _signal_dot("ML", float(candidate.get("ml_score", 0.0))),
        _signal_dot("Pattern", float(candidate.get("pattern_score", 0.0)) * 10.0),
    ]
    why = escape(_candidate_why(candidate))
    return (
        "<article class='card'>"
        f"<div class='ticker'>{ticker}</div>"
        f"<div class='company'>{company}</div>"
        "<div class='price-row'>"
        f"<div><strong>Buy</strong>${current:.2f}</div>"
        f"<div><strong>Target</strong>${target:.2f} ({upside:+.1f}%)</div>"
        f"<div><strong>Stop</strong>${stop:.2f} ({downside:+.1f}%)</div>"
        "</div>"
        f"<div class='signal-row'>{''.join(signals)}</div>"
        f"<div class='why'>{why}</div>"
        "</article>"
    )


def _signal_dot(label: str, score: float) -> str:
    tone = "green" if score >= 70 else "yellow" if score >= 50 else "red"
    return f"<span class='dot {tone}'>{escape(label)} {int(round(score))}</span>"


def _candidate_why(candidate: dict) -> str:
    sector_tag = str(candidate.get("sector_temperature_tag", "")).strip()
    sector = str(candidate.get("sector", "Unknown"))
    pieces = [sector_tag or sector]
    if float(candidate.get("volume_momentum_score", 0.0)) >= 80:
        pieces.append("Vol surge")
    elif float(candidate.get("rs_score", 0.0)) >= 70:
        pieces.append("Strong RS")
    else:
        pieces.append("Momentum setup")
    pieces.append(f"ML {float(candidate.get('probability_4pct_5d', 0.0)):.0f}%")
    pieces.append(f"{int(candidate.get('confluence_count', 0))}/5 signals")
    return " | ".join(pieces)


def _build_last_week_results(last_week_summary: dict | None) -> str:
    if not last_week_summary:
        return "<div class='card'>No completed weekly results yet.</div>"
    return (
        "<table>"
        "<thead><tr><th>Picks</th><th>Target Hits</th><th>Target Hit Rate</th><th>Positive Return Rate</th><th>Avg Return</th><th>Best</th><th>Worst</th></tr></thead>"
        "<tbody><tr>"
        f"<td>{int(last_week_summary.get('picks', 0))}</td>"
        f"<td>{int(last_week_summary.get('target_hits', last_week_summary.get('winners', 0)))}</td>"
        f"<td>{float(last_week_summary.get('target_hit_rate', last_week_summary.get('hit_rate', 0.0))):.0f}%</td>"
        f"<td>{float(last_week_summary.get('positive_return_rate', 0.0)):.0f}%</td>"
        f"<td>{float(last_week_summary.get('avg_return', 0.0)):+.1f}%</td>"
        f"<td>{escape(str(last_week_summary.get('best_text', 'N/A')))}</td>"
        f"<td>{escape(str(last_week_summary.get('worst_text', 'N/A')))}</td>"
        "</tr></tbody></table>"
    )


def _build_rolling_summary(summary: dict | None) -> str:
    if not summary:
        return "<div class='card'>No rolling performance data yet.</div>"
    return (
        "<div class='card'>"
        f"<strong>8-week target hit rate:</strong> {float(summary.get('target_hit_rate', summary.get('win_rate', 0.0))):.0f}%"
        f" &nbsp; | &nbsp; <strong>Positive-return rate:</strong> {float(summary.get('positive_return_rate', 0.0)):.0f}%"
        f" &nbsp; | &nbsp; <strong>Avg return:</strong> {float(summary.get('average_return', 0.0)):+.1f}%/week"
        "</div>"
    )


def _build_weekly_chart(rows: list[dict]) -> str:
    if not rows:
        return "<div class='chart'>No completed weekly data yet.</div>"
    valid = [row for row in rows if row.get("week_label")]
    if not valid:
        return "<div class='chart'>No completed weekly data yet.</div>"
    lines = []
    for row in valid[-8:]:
        hit_rate = float(row.get("target_hit_rate", row.get("hit_rate", 0.0)))
        label = escape(str(row.get("week_label", "")))
        lines.append(
            "<div class='bar-row'>"
            f"<span>{label}</span>"
            f"<div class='bar-track'><div class='bar-fill' style='width:{max(min(hit_rate, 100.0), 0.0):.0f}%'></div></div>"
            f"<strong>{hit_rate:.0f}%</strong>"
            "</div>"
        )
    return f"<div class='chart'>{''.join(lines)}</div>"
