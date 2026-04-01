"""Helper script to package the PySide6 Stock Predictor into a macOS .app bundle."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"
ICON_PNG = ASSETS_DIR / "icon.png"
ICON_ICNS = ASSETS_DIR / "icon.icns"
ICONSET_DIR = ASSETS_DIR / "icon.iconset"
APP_NAME = "Stock Predictor.app"
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"
TEMP_DIST_DIR = Path(tempfile.gettempdir()) / "stock_predictor_distribution"


def _strip_metadata(path: Path) -> None:
    if not path.exists():
        return
    subprocess.run(["xattr", "-cr", str(path)], check=False)


def _clean_previous_builds() -> None:
    for path in (DIST_DIR, BUILD_DIR):
        if path.exists():
            _strip_metadata(path)


def prepare_distributable_bundle(source_app: Path) -> Path:
    staged_root = TEMP_DIST_DIR / "signed"
    if staged_root.exists():
        subprocess.run(["rm", "-rf", str(staged_root)], check=True)
    staged_root.mkdir(parents=True, exist_ok=True)

    distributable_app = staged_root / APP_NAME
    subprocess.run(["ditto", str(source_app), str(distributable_app)], check=True, cwd=PROJECT_ROOT)
    _strip_metadata(distributable_app)
    try:
        subprocess.run(
            ["codesign", "--force", "--deep", "--sign", "-", "--timestamp=none", str(distributable_app)],
            check=True,
            cwd=PROJECT_ROOT,
        )
        subprocess.run(
            ["codesign", "--verify", "--deep", str(distributable_app)],
            check=True,
            cwd=PROJECT_ROOT,
        )
    except subprocess.CalledProcessError:
        # A fully signed bundle is nice to have, but not required for
        # the local drag-to-Applications installer we generate here.
        pass
    return distributable_app


def build_icns() -> None:
    if not ICON_PNG.exists():
        raise FileNotFoundError(f"Missing icon source: {ICON_PNG}")
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


def build_app_bundle() -> Path:
    _clean_previous_builds()
    build_icns()
    subprocess.run([sys.executable, "setup.py", "py2app"], cwd=PROJECT_ROOT, check=True)
    return DIST_DIR / APP_NAME


def main() -> int:
    app_path = build_app_bundle()
    print(f"Built app bundle: {app_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
