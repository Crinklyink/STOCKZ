from __future__ import annotations

import os
import plistlib
import subprocess
import sys
from pathlib import Path

from stock_predictor.config import get_config


PROJECT_ROOT = Path(__file__).resolve().parent
LAUNCH_AGENTS = Path.home() / "Library" / "LaunchAgents"
LOG_DIR = PROJECT_ROOT / "logs"
WEEKLY_PLIST = LAUNCH_AGENTS / "com.stockpredictor.weekly.plist"
MIDWEEK_PLIST = LAUNCH_AGENTS / "com.stockpredictor.midweek.plist"


def smtp_configured(config) -> bool:
    return bool(config.alert_email and config.smtp_host and config.smtp_username and config.smtp_password)


def _parse_hhmm(value: str, *, default_hour: int, default_minute: int) -> tuple[int, int]:
    try:
        hour_text, minute_text = value.split(":", 1)
        hour = max(0, min(23, int(hour_text)))
        minute = max(0, min(59, int(minute_text)))
        return hour, minute
    except Exception:
        return default_hour, default_minute


def write_plist(path: Path, *, label: str, script_name: str, weekday: int, hour: int, minute: int, stdout_name: str, stderr_name: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LAUNCH_AGENTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "Label": label,
        "ProgramArguments": [
            sys.executable,
            str(PROJECT_ROOT / script_name),
        ],
        "WorkingDirectory": str(PROJECT_ROOT),
        "StartCalendarInterval": {
            "Weekday": weekday,
            "Hour": hour,
            "Minute": minute,
        },
        "StandardOutPath": str(LOG_DIR / stdout_name),
        "StandardErrorPath": str(LOG_DIR / stderr_name),
    }
    with path.open("wb") as handle:
        plistlib.dump(payload, handle)


def load_launch_agent(path: Path) -> bool:
    subprocess.run(["launchctl", "unload", str(path)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    result = subprocess.run(["launchctl", "load", str(path)], check=False, capture_output=True, text=True)
    return result.returncode == 0


def main() -> int:
    os.chdir(PROJECT_ROOT)
    config = get_config()
    sunday_hour, sunday_minute = _parse_hhmm(os.getenv("SUNDAY_SCAN_TIME", "18:00"), default_hour=18, default_minute=0)
    wednesday_hour, wednesday_minute = _parse_hhmm(os.getenv("WEDNESDAY_CHECK_TIME", "18:00"), default_hour=18, default_minute=0)
    write_plist(
        WEEKLY_PLIST,
        label="com.stockpredictor.weekly",
        script_name="auto_run.py",
        weekday=0,
        hour=sunday_hour,
        minute=sunday_minute,
        stdout_name="weekly.log",
        stderr_name="weekly_err.log",
    )
    write_plist(
        MIDWEEK_PLIST,
        label="com.stockpredictor.midweek",
        script_name="midweek_check.py",
        weekday=3,
        hour=wednesday_hour,
        minute=wednesday_minute,
        stdout_name="midweek.log",
        stderr_name="midweek_err.log",
    )
    weekly_loaded = load_launch_agent(WEEKLY_PLIST)
    midweek_loaded = load_launch_agent(MIDWEEK_PLIST)

    print(("✅" if weekly_loaded else "⚠️") + " Sunday auto-run scheduled (every Sunday 6pm)")
    print(("✅" if midweek_loaded else "⚠️") + " Wednesday check-in scheduled")
    print("✅ Auto-open report/app enabled")
    if smtp_configured(config):
        print(f"✅ Weekly email enabled ({config.alert_email})")
    else:
        print("⚠️ Email not configured (add SMTP_PASS to .env to enable)")
        print("")
        print("To enable email: Go to Gmail → Settings → Security")
        print("→ 2-Step Verification → App Passwords")
        print("→ Generate password for 'Mail'")
        print("Add to .env: SMTP_PASS=xxxx xxxx xxxx xxxx")
    print("")
    print("You're all set. The system will run automatically.")
    print("Open the app or report every Sunday evening for picks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
