# Stock Predictor

A native macOS stock analysis app and Python prediction pipeline for ranking U.S. stocks by short-term upside potential.

It combines technical signals, relative strength, market regime filters, earnings context, sentiment, options/dark-pool signals, and ML models to generate weekly picks, track results, and display everything in a native Mac app.

## What It Does

- Scans a broad U.S. stock universe
- Scores stocks for short-horizon upside potential
- Applies regime-aware filters using VIX, SPY trend, breadth, and sector context
- Tracks weekly picks and evaluates target-hit outcomes honestly
- Generates reports, HTML summaries, and app-ready artifacts
- Runs as a native macOS desktop app with weekly automation support

## Main Features

### Prediction Engine
- Two-stage scan for speed
- Technical indicators, momentum, RS, volume, pattern quality
- XGBoost + LightGBM ensemble
- Multi-timeframe confirmation
- Regime-aware filtering
- Risk/reward and confluence gating

### Tracking & Reporting
- Weekly paper-trade tracking
- Target-hit rate as primary success metric
- Positive-return rate tracked separately
- Path-aware target resolution using evaluation-window highs
- HTML reports and email-ready summaries

### Native macOS App
- Built with PySide6
- Dashboard, Picks, Performance, History, Settings, and Analyze tabs
- Light/dark mode adaptive UI
- Native app packaging and DMG installer flow

## Project Structure

```text
stock_predictor/
├── analysis/        # indicators, macro, RS, patterns, trade signals
├── data/            # fetchers, caching, sentiment, congress, dark pool
├── models/          # XGBoost, LightGBM, TFT/LSTM scaffolding, ensemble
├── output/          # reports, backtest tracking, alerts, HTML/PDF output
├── scoring/         # composite scoring, adaptive weighting, portfolio logic
└── artifacts/       # latest scan/report outputs
Top-level app files:

main.py — CLI entry point
app.py — native macOS app entry point
auto_run.py — weekly automation runner
midweek_check.py — Wednesday update runner
setup_schedule.py — launchd automation setup
build_app.py / setup.py — app packaging
build_installer.py — DMG installer build
Installation
Python
Recommended: Python 3.13 or lower for safest desktop compatibility.

Install dependencies:

pip install -r requirements.txt
Desktop App Dependencies
If running the native app directly:

pip install PySide6
Running the Project
Run a Weekly Scan
python3 main.py --top-n 10 --universe full
Train the Models
python3 main.py --train --universe full
View Historical Results
python3 main.py --results
Analyze One Stock
python3 main.py --analyze AAPL --universe full
Launch the Native App
python3 app.py
Automation
Set up weekly and mid-week runs on macOS:

python3 setup_schedule.py
This configures:

Sunday 6:00 PM weekly run
Wednesday 6:00 PM check-in
browser-friendly report generation
optional email delivery if SMTP is configured
Email Configuration
Create a .env file with:

ALERT_EMAIL=you@example.com
SMTP_USER=you@example.com
SMTP_PASS=your_app_password
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
Gmail app passwords are recommended if using Gmail SMTP.

Outputs
The system writes artifacts such as:

latest scan JSON
markdown/HTML reports
feature importance charts
backtest summaries
weekly HTML report for browser viewing
Performance Philosophy
This project is built to avoid fake confidence.

Key principles:

target-hit rate is the primary success metric
positive-return rate is tracked separately
regime filters can block picks entirely in hostile markets
weak pattern signals do not receive free score credit
reporting aims to reflect actual trade intent, not just “finished green”
Current Workflow
Train or refresh models
Run weekly scan
Review picks in app or HTML report
Track outcomes through weekly results
Analyze individual stocks on demand in the app
Packaging
To build the macOS app:

python3 build_app.py
To build the installer DMG:

python3 build_installer.py
Disclaimer
This software is for research and educational use only. It is not financial advice. Markets are uncertain, and any trading strategy can lose money.
