# STOCKZ

Swift macOS stock-scanning app with a Python prediction backend for weekly AI-assisted watchlists, model diagnostics, risk review, alerts, paper-trade tracking, reports, and automation.

## Overview

STOCKZ is built around a Swift macOS app backed by the existing Python stock prediction pipeline. The app reads generated scan artifacts, model metadata, reports, and paper-trade history from the repository, then presents them in a native desktop workflow.

It is designed to:

- scan and rank U.S. stocks for short-term upside potential
- explain why each pick ranked
- compare watchlist candidates side by side
- simulate paper portfolio sizing and risk
- track historical scan outcomes
- surface model performance, drift, and feature importance
- automate weekly scans and midweek updates

## Swift App

The primary app is now the Swift package in `SwiftStockPredictor`.

```bash
cd SwiftStockPredictor
swift build
./build_app.sh
open "../dist/Stock Predictor Swift.app"
```

The packaged app is written to:

```text
dist/Stock Predictor Swift.app
```

## Current App Features

- Dashboard with current watchlist, model sync, market regime, and clickable ticker rows.
- Pick detail drilldown with chart, score breakdown, why it ranked, stop/target, sector strength, model confidence, and invalidation rules.
- Picks modes for ranked view, side-by-side comparison, paper portfolio simulator, and "Why Not Picked?" review.
- Live scan progress stages: fetching data, scoring, filtering, ranking, and saving artifacts.
- Risk dashboard for VIX regime, sector crowding, stop distance, and sizing warnings.
- Alerts page for target, stop, score-improvement, and top-10 movement rules.
- Model Lab with AUC, training samples, label target, feature importance, calibration, and drift warnings.
- Scan history timeline backed by saved scan and backtest artifacts.

## Backend Commands

The Swift app reads and runs the Python backend from the repository root. Key artifacts are generated under `stock_predictor/artifacts`.

```bash
python3 main.py --top-n 10 --paper-trade
python3 main.py --train
python3 main.py --backtest-adaptive
python3 main.py --analyze AAPL --universe full
```

The app looks for a local Python environment at `.venv/bin/python3` first, then falls back to system Python.

## Setup

```bash
git clone https://github.com/Crinklyink/STOCKZ.git
cd STOCKZ
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then edit `.env` with any email, scan, and model settings you want.

## Automation

Set up weekly and midweek macOS jobs:

```bash
python3 setup_schedule.py
```

Manual automation run:

```bash
python3 auto_run.py
```

## Project Structure

```text
STOCKZ/
├── SwiftStockPredictor/        # Primary Swift macOS app
├── dist/                       # Packaged app outputs
├── main.py                     # Core Python CLI entry point
├── auto_run.py                 # Sunday automation workflow
├── midweek_check.py            # Wednesday update workflow
├── setup_schedule.py           # Installs launchd schedules
├── stock_predictor/
│   ├── analysis/               # Indicators, macro, RS, patterns, trade signals
│   ├── data/                   # Fetchers, caching, sentiment, fundamentals
│   ├── models/                 # ML models, validation, monitoring
│   ├── output/                 # Reports, alerts, tracking, HTML/PDF output
│   ├── scoring/                # Composite scoring and portfolio logic
│   └── artifacts/              # Latest scan and runtime outputs
└── tests/                      # Python test suite
```

## Git LFS

Large app bundles, DMGs, and model pickle files are tracked with Git LFS. Install LFS before cloning or pushing large artifacts:

```bash
git lfs install
git lfs pull
```

Tracked LFS patterns include:

```text
dist/*.dmg
dist/*.app/**
stock_predictor/models/*.pkl
```

## Legacy Python UI

The older Python UI and automation scripts remain in the repo for compatibility, but new product work should target the Swift app.

## Disclaimer

This software is for research and educational use only. It is not financial advice. Markets are uncertain, and any trading strategy can lose money.
