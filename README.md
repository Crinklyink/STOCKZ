# STOCKZ

Swift macOS stock-scanning app for weekly AI-assisted watchlists, model diagnostics, risk review, alerts, and paper portfolio planning.

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

## Current Features

- Dashboard with current watchlist, model sync, market regime, and clickable ticker rows.
- Pick detail drilldown with chart, score breakdown, why it ranked, stop/target, sector strength, model confidence, and invalidation rules.
- Picks modes for ranked view, side-by-side comparison, paper portfolio simulator, and "Why Not Picked?" review.
- Live scan progress stages: fetching data, scoring, filtering, ranking, and saving artifacts.
- Risk dashboard for VIX regime, sector crowding, stop distance, and sizing warnings.
- Alerts page for target, stop, score-improvement, and top-10 movement rules.
- Model Lab with AUC, training samples, label target, feature importance, calibration, and drift warnings.
- Scan history timeline backed by saved scan and backtest artifacts.

## Backend

The Swift app reads and runs the existing Python backend from the repository root. Key artifacts are generated under `stock_predictor/artifacts`.

Common backend commands:

```bash
python3 main.py --top-n 10 --paper-trade
python3 main.py --train
python3 main.py --backtest-adaptive
```

The app looks for a local Python environment at `.venv/bin/python3` first, then falls back to system Python.

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
