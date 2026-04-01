"""Discord and Telegram update integration."""

from __future__ import annotations

from typing import Iterable

import requests

from stock_predictor.config import AppConfig
from stock_predictor.scoring.composite import CandidateScore


def format_bot_message(title: str, candidates: Iterable[CandidateScore]) -> str:
    lines = [title, ""]
    for candidate in candidates:
        lines.append(
            (
                f"{candidate.ticker} | {candidate.sector} | score {candidate.final_score:.1f} "
                f"(+/- {candidate.score_uncertainty:.1f}, {candidate.confidence_label})"
            )
        )
        lines.append(
            (
                f"Entry {candidate.current_price:.2f} | Target {candidate.targets['tp2']:.2f} | "
                f"Stop {candidate.stop_loss:.2f} | Pattern {candidate.pattern_name}"
            )
        )
        lines.append(
            (
                f"Signals: ML {candidate.ml_score:.0f}, Tech {candidate.technical_score:.0f}, "
                f"Options {candidate.options_score:.0f}, Sentiment {candidate.sentiment_score:.0f}"
            )
        )
        lines.append(f"Why: {candidate.ai_explanation}")
        lines.append("")
    return "\n".join(lines)


def send_weekly_picks(config: AppConfig, candidates: Iterable[CandidateScore]) -> None:
    message = format_bot_message("Weekly top 10 picks", candidates)
    _send_message(config, message)


def send_midweek_update(config: AppConfig, summary: str) -> None:
    _send_message(config, f"Midweek update\n\n{summary}")


def send_endweek_results(config: AppConfig, summary: str) -> None:
    _send_message(config, f"End-of-week results\n\n{summary}")


def _send_message(config: AppConfig, message: str) -> None:
    if config.discord_webhook_url:
        requests.post(config.discord_webhook_url, json={"content": message}, timeout=20)
    if config.telegram_bot_token and config.telegram_chat_id:
        requests.post(
            f"https://api.telegram.org/bot{config.telegram_bot_token}/sendMessage",
            data={"chat_id": config.telegram_chat_id, "text": message},
            timeout=20,
        )
