from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from stock_predictor.config import get_config
from stock_predictor.data.cache import SQLiteCache
from stock_predictor.data.fetcher import MarketDataFetcher
from stock_predictor.output.alerts import send_email


PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"
MIDWEEK_LOG = LOG_DIR / "midweek.log"


def configure_logging(log_path: Path) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )
    return logging.getLogger("stockpredictor.midweek")


def smtp_configured(config) -> bool:
    return bool(config.alert_email and config.smtp_host and config.smtp_username and config.smtp_password)


def load_latest_scan(config) -> dict:
    if not config.latest_scan_path.exists():
        return {}
    return json.loads(config.latest_scan_path.read_text(encoding="utf-8"))


def fetch_current_prices(config, tickers: list[str]) -> dict[str, float]:
    cache = SQLiteCache(config.cache_db)
    fetcher = MarketDataFetcher(config, cache)
    prices: dict[str, float] = {}
    for ticker in tickers:
        info = fetcher.fetch_info(ticker)
        prices[ticker] = float(info.get("currentPrice") or info.get("lastPrice") or 0.0)
    return prices


def build_midweek_subject(rows: list[dict]) -> str:
    date_text = datetime.now().strftime("%b %d")
    if not rows:
        return f"📊 Mid-week check — no active picks [{date_text}]"
    headline = " ".join(
        f"{row['ticker']} {row['return_pct']:+.1f}% {'✅' if row['return_pct'] >= 0 else '⚠️'}"
        for row in rows[:3]
    )
    return f"📊 Mid-week check — {headline} [{date_text}]"


def build_midweek_body(rows: list[dict]) -> str:
    date_text = datetime.now().strftime("%a %b %d")
    lines = [f"MID-WEEK UPDATE — {date_text}", "─────────────────────────────"]
    if not rows:
        lines.append("No active weekly picks were issued this week.")
        return "\n".join(lines)
    target_hits = sum(row["hit_target"] for row in rows)
    stop_hits = sum(row["hit_stop"] for row in rows)
    for row in rows:
        status = "✅ on track" if row["return_pct"] >= 0 else "⚠️ watching"
        if row["hit_target"]:
            status = "✅ target hit"
        elif row["hit_stop"]:
            status = "❌ below stop"
        lines.append(
            f"{row['ticker']}:  bought ${row['entry']:.2f} -> now ${row['current']:.2f}  {row['return_pct']:+.1f}% {status}"
        )
    best = max(rows, key=lambda row: row["return_pct"])
    lines.extend(
        [
            "",
            f"Targets hit: {target_hits}/{len(rows)} | Stops hit: {stop_hits}/{len(rows)}",
            f"Best performing: {best['ticker']} {best['return_pct']:+.1f}%",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    os.chdir(PROJECT_ROOT)
    logger = configure_logging(MIDWEEK_LOG)
    config = get_config()
    payload = load_latest_scan(config)
    picks = list(payload.get("selected", []))
    tickers = [str(candidate.get("ticker")) for candidate in picks]
    prices = fetch_current_prices(config, tickers) if tickers else {}
    rows = []
    for candidate in picks:
        ticker = str(candidate.get("ticker"))
        entry = float(candidate.get("current_price", 0.0))
        current = float(prices.get(ticker, entry))
        target = float(candidate.get("targets", {}).get("tp2", entry))
        stop = float(candidate.get("stop_loss", entry))
        return_pct = ((current / entry) - 1.0) * 100.0 if entry else 0.0
        rows.append(
            {
                "ticker": ticker,
                "entry": entry,
                "current": current,
                "return_pct": return_pct,
                "hit_target": current >= target if target else False,
                "hit_stop": current <= stop if stop else False,
            }
        )
    subject = build_midweek_subject(rows)
    body = build_midweek_body(rows)
    logger.info("%s\n%s", subject, body)
    if smtp_configured(config):
        send_email(config, config.alert_email, subject, body)
        logger.info("Mid-week update sent to %s", config.alert_email)
    else:
        logger.warning("Email not configured; skipping mid-week update email")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
