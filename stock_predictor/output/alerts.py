"""Optional email and SMS alerts."""

from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage
from typing import Iterable

import requests

from stock_predictor.config import AppConfig
from stock_predictor.scoring.composite import CandidateScore

LOGGER = logging.getLogger(__name__)

def format_alert_body(
    candidates: Iterable[CandidateScore],
    *,
    summary_header: dict | None = None,
    footer_lines: Iterable[str] | None = None,
) -> str:
    lines = ["Weekly stock scan complete.", ""]
    if summary_header:
        lines.extend(
            [
                f"Date: {summary_header.get('date', '')}",
                (
                    f"Universe: {summary_header.get('universe_total', 0)} | "
                    f"Runtime: {summary_header.get('runtime_seconds', 0):.0f}s | "
                    f"Regime: {str(summary_header.get('regime', 'unknown')).upper()}"
                ),
                (
                    f"VIX: {summary_header.get('vix', 0):.2f} | "
                    f"Above score threshold: {summary_header.get('qualified_count', 0)} | "
                    f"Threshold: {summary_header.get('threshold_used', 0):.1f}"
                ),
                (
                    f"Top sector: {summary_header.get('top_sector', 'Unknown')} | "
                    f"Worst sector: {summary_header.get('worst_sector', 'Unknown')}"
                ),
                "",
            ]
        )
    lines.append("Top picks:")
    for index, candidate in enumerate(candidates, start=1):
        lines.extend(
            [
                (
                    f"{index}. {candidate.ticker} | {candidate.company_name} | "
                    f"Score {candidate.final_score:.1f} | Entry {candidate.current_price:.2f} | "
                    f"Target {candidate.targets['tp2']:.2f} | Stop {candidate.stop_loss:.2f}"
                ),
                f"   Why: {candidate.ai_explanation}",
            ]
        )
    if footer_lines:
        lines.extend(["", *[str(line) for line in footer_lines if str(line).strip()]])
    return "\n".join(lines)


def build_alert_subject(candidates: Iterable[CandidateScore], *, summary_header: dict | None = None) -> str:
    picks = list(candidates)[:3]
    headline = ", ".join(f"{candidate.ticker} {candidate.final_score:.1f}" for candidate in picks) or "No picks"
    date_text = summary_header.get("date", "") if summary_header else ""
    return f"Weekly Picks: {headline} {date_text}".strip()


def send_alerts(
    config: AppConfig,
    candidates: Iterable[CandidateScore],
    *,
    alert_email: str | None = None,
    alert_phone: str | None = None,
    summary_header: dict | None = None,
    footer_lines: Iterable[str] | None = None,
) -> None:
    picks = list(candidates)
    destination_email = alert_email or config.alert_email
    subject = build_alert_subject(picks, summary_header=summary_header)
    body = format_alert_body(picks, summary_header=summary_header, footer_lines=footer_lines)
    if destination_email:
        send_email(config, destination_email, subject, body)
    if alert_phone:
        send_sms(config, alert_phone, body[:1200])


def send_email(config: AppConfig, destination: str, subject: str, body: str) -> None:
    if not all([config.smtp_host, config.smtp_username, config.smtp_password, config.smtp_from_email]):
        LOGGER.warning("SMTP not configured; skipping weekly email digest")
        return
    message = EmailMessage()
    message["From"] = config.smtp_from_email
    message["To"] = destination
    message["Subject"] = subject
    message.set_content(body)
    with smtplib.SMTP(config.smtp_host, config.smtp_port) as server:
        server.starttls()
        server.login(config.smtp_username, config.smtp_password)
        server.send_message(message)


def send_sms(config: AppConfig, destination: str, body: str) -> None:
    if not all([config.twilio_account_sid, config.twilio_auth_token, config.twilio_from_number]):
        return
    requests.post(
        f"https://api.twilio.com/2010-04-01/Accounts/{config.twilio_account_sid}/Messages.json",
        auth=(config.twilio_account_sid, config.twilio_auth_token),
        data={
            "From": config.twilio_from_number,
            "To": destination,
            "Body": body,
        },
        timeout=20,
    )
