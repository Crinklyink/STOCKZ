# Stock Predictor
1. Copy `.env.example` to `.env` and fill in Gmail credentials if you want email alerts.
2. Run `python3 setup_schedule.py` once to install the Sunday and Wednesday launchd jobs.
3. Open the native app with `python3 app.py`.
4. The app reads directly from `stock_predictor/artifacts/latest_scan.json`, `backtest.db`, and `paper_trades.db`.
5. Use **Run Scan Now** in the app to launch a live scan without opening Terminal.
6. Sunday automation runs training, scan, results, HTML report, and email automatically.
7. Wednesday automation sends a quick progress update on the current week's picks.
8. Logs live in `logs/weekly.log` and `logs/midweek.log`.
9. Package the app with `python3 build_app.py` or `python3 setup.py py2app`.
10. Create a drag-to-Applications installer with `python3 build_installer.py` and share `dist/Stock Predictor Installer.dmg`.
