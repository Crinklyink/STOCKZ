# macOS Distribution Guide for Stock Predictor

This application has been structured to easily package entirely into a self-contained `.app` application and `.dmg` volume. At runtime, the compiled program will intelligently sandbox all cache, DB, and temporary artifact footprints directly into `~/Library/Application Support/Stock Predictor` and `~/Library/Logs/Stock Predictor`, leaving zero residual configuration files inside your App Bundle, and fully protecting it continuously.

## Build Requirements
- Valid Python 3.10+ environment
- `PyInstaller` (`pip install pyinstaller`)
- Apple `sips` and `iconutil` (included natively with macOS)
- Apple `hdiutil` (included natively with macOS)

## Quick Build Command
Run these sequentially from the project root:

```bash
# 1. Compile the self-contained macOS Application bundle
python3 build_mac.py

# 2. Package the output into a drag-and-drop DMG Install Volume
chmod +x scripts/create_dmg.sh
./scripts/create_dmg.sh
```

## Verify Target Outputs
Once finished, navigate to the local `./dist/` directory. You will find two isolated products:
- `dist/Stock Predictor.app`
- `dist/Stock Predictor.dmg`

## First Run & Verification Checklist
To confirm the app functions flawlessly in distribution:
1. Double-click the generated `Stock Predictor.dmg`.
2. Drag the `Stock Predictor.app` icon over the `Applications` folder alias within the window.
3. Open `Launchpad` (or Spotlight) and launch "Stock Predictor".
4. *Verify Paths:* Press `CMD + Space` and open `~/Library/Application Support/Stock Predictor`. You should see `artifacts`, `checkpoints`, `reports`, and your `paper_trade.db` successfully populating! Note: No fake test data has been shipped.
