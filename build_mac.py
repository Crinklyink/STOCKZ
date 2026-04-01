"""Helper script to package the PySide6 Stock Predictor into a macOS .app bundle using PyInstaller."""

from __future__ import annotations

import subprocess
import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"
ICON_PNG = ASSETS_DIR / "icon.png"
ICON_ICNS = ASSETS_DIR / "icon.icns"
ICONSET_DIR = ASSETS_DIR / "icon.iconset"
APP_NAME = "Stock Predictor.app"
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"


def _strip_metadata(path: Path) -> None:
    if not path.exists():
        return
    subprocess.run(["xattr", "-cr", str(path)], check=False)


def _clean_previous_builds() -> None:
    for path in (DIST_DIR, BUILD_DIR):
        if path.exists():
            _strip_metadata(path)
            shutil.rmtree(path, ignore_errors=True)


def build_icns() -> None:
    if not ICON_PNG.exists():
        print(f"Skipping ICNS generation. Missing icon source: {ICON_PNG}")
        return
    _strip_metadata(ASSETS_DIR)
    ICONSET_DIR.mkdir(parents=True, exist_ok=True)
    sizes = [16, 32, 64, 128, 256, 512]
    for size in sizes:
        output = ICONSET_DIR / f"icon_{size}x{size}.png"
        subprocess.run(["sips", "-z", str(size), str(size), str(ICON_PNG), "--out", str(output)], check=True)
        retina = ICONSET_DIR / f"icon_{size}x{size}@2x.png"
        retina_size = size * 2
        if retina_size <= 1024:
            subprocess.run(["sips", "-z", str(retina_size), str(retina_size), str(ICON_PNG), "--out", str(retina)], check=True)
    subprocess.run(["iconutil", "-c", "icns", str(ICONSET_DIR), "-o", str(ICON_ICNS)], check=True)
    shutil.rmtree(ICONSET_DIR, ignore_errors=True)


def build_app_bundle() -> Path:
    _clean_previous_builds()
    build_icns()
    
    # Require pyinstaller presence
    try:
        __import__("PyInstaller")
    except ImportError:
        print("Error: PyInstaller is not installed. Please run `pip install pyinstaller`.")
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--windowed",
        "--name", "Stock Predictor",
    ]
    if ICON_ICNS.exists():
        cmd.extend(["--icon", str(ICON_ICNS)])
        
    cmd.extend([
        "--add-data", f"{ASSETS_DIR}:assets",
        "--add-data", f"{PROJECT_ROOT / '.env.example'}:.",
        str(PROJECT_ROOT / "app.py")
    ])
    
    # Optional implicit dependencies
    cmd.extend([
        "--hidden-import", "requests", 
        "--hidden-import", "pandas",
        "--hidden-import", "xgboost",
    ])
    
    print("Running PyInstaller...")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    return DIST_DIR / APP_NAME


def main() -> int:
    app_path = build_app_bundle()
    print(f"\nSuccessfully built macOS App Bundle at: {app_path}")
    print("To install via raw DMG, run: ./scripts/create_dmg.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
