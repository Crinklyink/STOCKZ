# STOCKZ

Native macOS stock prediction app with weekly scans, model retraining, paper-trade tracking, performance history, HTML reports, and optional email alerts.

## Overview

STOCKZ is a desktop-first stock prediction system built around a native macOS app and a Python prediction pipeline.

It is designed to:

- scan a broad U.S. stock universe
- score and rank stocks for short-term upside potential
- apply regime-aware filters
- display weekly picks in a native Mac app
- track paper-trade outcomes honestly
- generate reports and HTML summaries
- automate Sunday runs and Wednesday updates
- optionally send email digests

The app reads live state from generated artifacts and databases, then presents that data through a native PySide6 interface.

---

## Main Features

### Prediction Engine

- Two-stage scan for speed
- Technical indicators, momentum, relative strength, volume, and pattern quality
- XGBoost + LightGBM ensemble scoring
- Multi-timeframe confirmation
- Regime-aware filtering using VIX, SPY trend, breadth, and sector context
- Risk/reward and confluence gating

### Tracking and Reporting

- Weekly paper-trade tracking
- Target-hit rate as the primary success metric
- Positive-return rate tracked separately
- Path-aware target resolution using evaluation-window highs
- Historical performance views
- HTML reports and email-ready summaries

### Native macOS App

- Built with PySide6
- Dashboard
- This Week's Picks
- Analyze Stock
- Performance
- History
- Settings
- Light/dark mode adaptive UI
- Native app packaging and DMG installer flow

### Automation

- Sunday workflow can retrain the model, run a scan, generate results, write an HTML report, and send an email digest
- Wednesday workflow can send a progress update on the current week's picks
- Launchd scheduling support on macOS

---

## Project Structure

```
STOCKZ/
├── app.py                  # Native desktop app entry point
├── app_window.py           # Main window and app data service
├── main.py                 # Core CLI entry point
├── auto_run.py             # Sunday automation workflow
├── midweek_check.py        # Wednesday update workflow
├── setup_schedule.py       # Installs launchd schedules
├── build_app.py            # Build macOS app bundle
├── build_installer.py      # Build macOS DMG installer
├── build_mac.py            # Alternate macOS build path
├── dashboard.py            # Streamlit dashboard
├── .env.example            # Example settings and email config
├── logs/                   # Automation logs
├── reports/                # Generated reports
├── dist/                   # Built app / installer outputs
└── stock_predictor/
    ├── analysis/           # Indicators, macro, RS, patterns, trade signals
    ├── data/               # Fetchers, caching, sentiment, dark-pool inputs
    ├── models/             # ML models and ensemble logic
    ├── output/             # Reports, alerts, tracking, HTML/PDF output
    ├── scoring/            # Composite scoring and portfolio logic
    └── artifacts/          # Latest scan and runtime outputs
```

## Requirements

- macOS
- Python 3.10+
- Recommended: virtual environment

For safest desktop compatibility, Python 3.13 or lower is a good choice.

Optional packaging tools:

- py2app
- PyInstaller
- macOS tools like sips, iconutil, and hdiutil

## Installation

1. Clone the repo
```bash
git clone https://github.com/Crinklyink/STOCKZ.git
cd STOCKZ
```

2. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Install desktop app dependency

If you want to run the native app directly:

```bash
pip install PySide6
```

5. Create your environment file
```bash
cp .env.example .env
```

Then edit .env with your preferred settings.

## Configuration

Example .env:

```
ALERT_EMAIL=your@gmail.com
SMTP_USER=your@gmail.com
SMTP_PASS=xxxx xxxx xxxx xxxx
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SUNDAY_SCAN_TIME=18:00
WEDNESDAY_CHECK_TIME=18:00
AUTO_OPEN_APP_AFTER_SCAN=1
RETRAIN_MODEL_WEEKLY=1
APP_VIX_CUTOFF=30
APP_MIN_SCORE_THRESHOLD=54
APP_MIN_RR_RATIO=1.0
APP_MAX_PICKS=5
DEFAULT_UNIVERSE=full
```

## Important Settings

- **ALERT_EMAIL** — where email digests are sent
- **SMTP_USER / SMTP_PASS** — SMTP credentials for alerts
- **SUNDAY_SCAN_TIME** — weekly Sunday run time
- **WEDNESDAY_CHECK_TIME** — midweek update time
- **AUTO_OPEN_APP_AFTER_SCAN** — open the app after automated scan finishes
- **RETRAIN_MODEL_WEEKLY** — retrain automatically each week
- **APP_VIX_CUTOFF** — market volatility cutoff
- **APP_MIN_SCORE_THRESHOLD** — minimum score for picks
- **APP_MIN_RR_RATIO** — minimum risk/reward ratio
- **APP_MAX_PICKS** — maximum number of official picks
- **DEFAULT_UNIVERSE** — default stock universe mode

## Running the Project

### Run a Weekly Scan
```bash
python3 main.py --top-n 10 --universe full
```

### Train the Models
```bash
python3 main.py --train --universe full
```

### View Historical Results
```bash
python3 main.py --results
```

### Analyze One Stock
```bash
python3 main.py --analyze AAPL --universe full
```

### Launch the Native App
```bash
python3 app.py
```

From the app, you can:

- view the dashboard
- inspect this week's picks
- analyze individual stocks
- review performance and history
- change settings
- run a scan immediately

## Automation

Set up weekly and mid-week runs on macOS:

```bash
python3 setup_schedule.py
```

This is intended to configure:

- Sunday weekly run
- Wednesday check-in
- browser-friendly report generation
- optional email delivery if SMTP is configured

### Manual Sunday Automation Run
```bash
python3 auto_run.py
```

## Email Configuration

Email is optional.

Create a .env file with:

```
ALERT_EMAIL=you@example.com
SMTP_USER=you@example.com
SMTP_PASS=your_app_password
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
```

Gmail app passwords are recommended if you are using Gmail SMTP.

## Outputs

The system writes artifacts such as:

- latest scan JSON
- markdown and HTML reports
- feature importance charts
- backtest summaries
- weekly HTML report for browser viewing
- performance history data
- logs

Common runtime locations include:

- stock_predictor/artifacts/
- logs/weekly.log
- logs/midweek.log
- reports/
- dist/

## Performance Philosophy

This project is built to avoid fake confidence.

Key principles:

- target-hit rate is the primary success metric
- positive-return rate is tracked separately
- regime filters can block picks entirely in hostile markets
- weak pattern signals do not receive free score credit
- reporting aims to reflect actual trade intent, not just "finished green"

## Current Workflow

1. Train or refresh models
2. Run weekly scan
3. Review picks in the app or HTML report
4. Track outcomes through weekly results
5. Analyze individual stocks on demand in the app

## Packaging

### Build the macOS app
```bash
python3 build_app.py
```

### Build the installer DMG
```bash
python3 build_installer.py
```

Typical outputs:

- dist/Stock Predictor.app
- dist/Stock Predictor Installer.dmg

## Logs

Automation logs are written to:

- logs/weekly.log
- logs/midweek.log

These are the first place to check if a scan, report, or email step fails.

## Troubleshooting

### App opens but shows no useful data

Run a scan first, then confirm expected artifact files and databases exist.

### Email is not sending

Check:

- .env values
- Gmail app password
- SMTP settings
- logs

### Packaging fails

Make sure required packaging tools are installed:

- py2app or PyInstaller
- macOS command-line tools
- valid icon assets in assets/

### Schedule does not run

Re-run:

```bash
python3 setup_schedule.py
```

Then inspect logs and launchd setup.

## Disclaimer

This software is for research and educational use only. It is not financial advice. Markets are uncertain, and any trading strategy can lose money.

## License

Add a license section here if this project is meant to be public or shared.