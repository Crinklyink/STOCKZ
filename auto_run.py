from __future__ import annotations

import logging
import os
import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

from main import run_results, run_scan, run_training
from stock_predictor.config import get_config
from stock_predictor.output.alerts import send_email
from stock_predictor.output.backtest import BacktestTracker, with_resolved_outcomes
from stock_predictor.output.html_report import generate_weekly_html_report


PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"
WEEKLY_LOG = LOG_DIR / "weekly.log"


def configure_logging(log_path: Path) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )
    return logging.getLogger("stockpredictor.auto")


def should_run_training(config) -> bool:
    if not config.xgb_metadata_path.exists():
        return True
    try:
        metadata = json.loads(config.xgb_metadata_path.read_text(encoding="utf-8"))
        trained_at = pd.to_datetime(metadata.get("trained_at"), utc=True, errors="coerce")
        if pd.isna(trained_at):
            return True
        now = pd.Timestamp.now(tz="UTC")
        return (now.isocalendar().week != trained_at.isocalendar().week) or (now.year != trained_at.year)
    except Exception:
        return True


def collect_results_snapshot(config) -> dict:
    tracker = BacktestTracker(config.paper_trade_db)
    frame = with_resolved_outcomes(tracker.paper_results_frame()).dropna(subset=["realized_return"]).copy()
    if frame.empty:
        return {"last_week": None, "rolling": {}, "weekly_rows": []}
    frame["positive_return"] = frame["realized_return"] > 0

    frame["created_ts"] = pd.to_datetime(frame["created_at"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["created_ts"]).copy()
    frame["week_start"] = frame["created_ts"].dt.tz_localize(None).dt.to_period("W").map(lambda period: period.start_time)
    frame["week_label"] = frame["created_ts"].dt.strftime("%m/%d")
    weekly = (
        frame.groupby("week_start", as_index=False)
        .agg(
            week_label=("week_label", "last"),
            picks=("ticker", "count"),
            target_hits=("resolved_target_hit", "sum"),
            target_hit_rate=("resolved_target_hit", lambda series: float(series.mean() * 100.0)),
            positive_return_rate=("positive_return", lambda series: float(series.mean() * 100.0)),
            avg_return=("realized_return", lambda series: float(series.mean() * 100.0)),
        )
        .sort_values("week_start")
        .tail(8)
    )
    latest_week = weekly.iloc[-1]
    latest_frame = frame.loc[frame["week_start"] == latest_week["week_start"]].copy()
    best = latest_frame.sort_values("realized_return", ascending=False).iloc[0]
    worst = latest_frame.sort_values("realized_return", ascending=True).iloc[0]
    def _result_label(row) -> str:
        if bool(row.get("resolved_target_hit")):
            return "target hit"
        if float(row.get("realized_return", 0.0)) > 0:
            return "positive, below target"
        return "below target"
    last_week = {
        "week_label": str(latest_week["week_label"]),
        "picks": int(latest_week["picks"]),
        "target_hits": int(latest_week["target_hits"]),
        "target_hit_rate": float(latest_week["target_hit_rate"]),
        "positive_return_rate": float(latest_week["positive_return_rate"]),
        "winners": int(latest_week["target_hits"]),
        "hit_rate": float(latest_week["target_hit_rate"]),
        "avg_return": float(latest_week["avg_return"]),
        "best_text": f"{best['ticker']} {float(best['realized_return']) * 100.0:+.1f}% ({_result_label(best)})",
        "worst_text": f"{worst['ticker']} {float(worst['realized_return']) * 100.0:+.1f}% ({_result_label(worst)})",
    }
    return {
        "last_week": last_week,
        "rolling": tracker.paper_trade_results_summary(),
        "weekly_rows": weekly.to_dict(orient="records"),
    }


def smtp_configured(config) -> bool:
    return bool(config.alert_email and config.smtp_host and config.smtp_username and config.smtp_password)


def smtp_setup_hint() -> str:
    return (
        "To enable email: Go to Gmail -> Settings -> Security\n"
        "-> 2-Step Verification -> App Passwords\n"
        "-> Generate password for 'Mail'\n"
        "Add to .env: SMTP_PASS=xxxx xxxx xxxx xxxx"
    )


def build_weekly_subject(payload: dict) -> str:
    picks = list(payload.get("selected", []))
    summary = payload.get("scan_summary", {})
    date_text = datetime.now().strftime("%b %d")
    if picks:
        headline = ", ".join(f"{candidate['ticker']} {round(float(candidate['final_score']))}" for candidate in picks[:3])
        return f"📈 {len(picks)} picks this week — {headline} [{date_text}]"
    return f"⛔ No picks this week — VIX too high ({float(summary.get('vix', 0.0)):.1f}) [{date_text}]"


def _why_text(candidate: dict) -> str:
    pieces = []
    sector_tag = str(candidate.get("sector_temperature_tag", "")).strip()
    pieces.append(sector_tag or str(candidate.get("sector", "Unknown")))
    if float(candidate.get("volume_momentum_score", 0.0)) >= 80:
        pieces.append("Vol surge")
    elif float(candidate.get("rs_score", 0.0)) >= 70:
        pieces.append("Strong RS")
    else:
        pieces.append("Momentum")
    pieces.append(f"ML {float(candidate.get('probability_4pct_5d', 0.0)):.0f}%")
    pieces.append(f"{int(candidate.get('confluence_count', 0))}/5 signals")
    return " | ".join(pieces)


def build_weekly_body(payload: dict, results_snapshot: dict) -> str:
    summary = payload.get("scan_summary", {})
    date_text = datetime.now().strftime("%b %d")
    picks = list(payload.get("selected", []))
    watchlist = list(payload.get("display_candidates", []))
    focus = picks if picks else watchlist[:3]
    lines = [
        "════════════════════════",
        f"WEEKLY PICKS — {date_text}",
        f"VIX: {float(summary.get('vix', 0.0)):.1f} | SPY: {float(summary.get('spy_week_return', 0.0)):+.1f}%",
        "════════════════════════",
        "",
    ]
    if not picks:
        lines.extend(
            [
                "No official picks this week.",
                str(summary.get("selection_warning", "Regime filter blocked entries this week.")),
                "",
                "WATCHLIST",
                "────────────────────────",
            ]
        )
    for candidate in focus:
        current = float(candidate.get("current_price", 0.0))
        target = float(candidate.get("targets", {}).get("tp2", current))
        stop = float(candidate.get("stop_loss", current))
        upside = ((target / current) - 1.0) * 100.0 if current else 0.0
        downside = ((stop / current) - 1.0) * 100.0 if current else 0.0
        lines.extend(
            [
                f"{candidate.get('tier_label', '•').split()[0]} {candidate['ticker']} — ${current:.2f}",
                f"   Buy at: ${current:.2f}",
                f"   Target: ${target:.2f} ({upside:+.1f}%)",
                f"   Stop: ${stop:.2f} ({downside:+.1f}%)",
                f"   Why: {_why_text(candidate)}",
                "",
            ]
        )
    last_week = results_snapshot.get("last_week")
    lines.extend(["────────────────────────", "LAST WEEK RESULTS", "────────────────────────"])
    if last_week:
        lines.extend(
            [
                (
                    f"Picks: {last_week['picks']} | Target hits: {last_week['target_hits']} | "
                    f"Target hit rate: {last_week['target_hit_rate']:.0f}%"
                ),
                (
                    f"Positive-return rate: {last_week['positive_return_rate']:.0f}% | "
                    f"Avg return: {last_week['avg_return']:+.1f}%"
                ),
                f"Best: {last_week['best_text']} | Worst: {last_week['worst_text']}",
            ]
        )
    else:
        lines.append("No completed weekly results yet.")
    rolling = results_snapshot.get("rolling") or {}
    lines.extend(["", "────────────────────────", "8-WEEK TRACK RECORD", "────────────────────────"])
    if rolling:
        lines.append(
            (
                f"Target hit rate: {float(rolling.get('target_hit_rate', rolling.get('win_rate', 0.0))):.0f}% | "
                f"Positive-return rate: {float(rolling.get('positive_return_rate', 0.0)):.0f}% | "
                f"Avg return: {float(rolling.get('average_return', 0.0)):+.1f}%/week"
            )
        )
    else:
        lines.append("No rolling track record yet.")
    return "\n".join(lines)


def send_weekly_digest(config, payload: dict, results_snapshot: dict, logger: logging.Logger) -> None:
    if not smtp_configured(config):
        logger.warning("Email not configured. %s", smtp_setup_hint().replace("\n", " | "))
        return
    send_email(
        config,
        config.alert_email,
        build_weekly_subject(payload),
        build_weekly_body(payload, results_snapshot),
    )
    logger.info("Weekly email digest sent to %s", config.alert_email)


def main() -> int:
    os.chdir(PROJECT_ROOT)
    logger = configure_logging(WEEKLY_LOG)
    config = get_config()
    logger.info("Sunday automation started")
    retrain_enabled = os.getenv("RETRAIN_MODEL_WEEKLY", "1") != "0"
    if retrain_enabled and should_run_training(config):
        training_text = run_training(fresh=False, universe_mode="full")
        logger = configure_logging(WEEKLY_LOG)
        logger.info("%s", training_text)
    else:
        if retrain_enabled:
            logger.info("Model already trained this week; skipping retrain.")
        else:
            logger.info("Weekly retraining disabled by settings; skipping retrain.")
    payload = run_scan(
        fresh=False,
        top_n=10,
        universe_mode="full",
        send_notifications=False,
    )
    logger = configure_logging(WEEKLY_LOG)
    logger.info("%s", payload.get("terminal_report", ""))
    results_text = run_results()
    logger = configure_logging(WEEKLY_LOG)
    logger.info("%s", results_text)
    results_snapshot = collect_results_snapshot(config)
    html_path = generate_weekly_html_report(
        config,
        payload,
        last_week_summary=results_snapshot.get("last_week"),
        rolling_summary=results_snapshot.get("rolling"),
        weekly_rows=results_snapshot.get("weekly_rows"),
        auto_open=os.getenv("AUTO_OPEN_APP_AFTER_SCAN", "1") != "0",
    )
    logger.info("Weekly HTML report written to %s", html_path)
    if os.getenv("AUTO_OPEN_APP_AFTER_SCAN", "1") != "0":
        app_bundle = PROJECT_ROOT / "dist" / "Stock Predictor.app"
        if app_bundle.exists():
            subprocess.run(["open", "-a", str(app_bundle)], check=False)
    send_weekly_digest(config, payload, results_snapshot, logger)
    logger.info("Sunday automation finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
