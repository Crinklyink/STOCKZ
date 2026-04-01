"""Build a macOS DMG installer for Stock Predictor."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from build_app import APP_NAME, PROJECT_ROOT, build_app_bundle, prepare_distributable_bundle


DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = Path(tempfile.gettempdir()) / "stock_predictor_installer"
DMG_NAME = "Stock Predictor Installer.dmg"
VOLUME_NAME = "Stock Predictor"


def _safe_remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def _strip_metadata(path: Path) -> None:
    if path.exists():
        subprocess.run(["xattr", "-cr", str(path)], check=False)


def _stage_installer(app_bundle: Path) -> Path:
    _safe_remove(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    staged_app = BUILD_DIR / APP_NAME
    shutil.copytree(app_bundle, staged_app, symlinks=True)
    _strip_metadata(staged_app)

    applications_link = BUILD_DIR / "Applications"
    _safe_remove(applications_link)
    applications_link.symlink_to("/Applications")

    instructions = BUILD_DIR / "Install Stock Predictor.txt"
    instructions.write_text(
        "\n".join(
            [
                "Stock Predictor",
                "",
                "To install:",
                "1. Drag 'Stock Predictor.app' into 'Applications'.",
                "2. Open it from Applications like any normal Mac app.",
                "",
                "If macOS warns that the app was downloaded from the internet,",
                "right-click the app in Applications and choose Open once.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return BUILD_DIR


def build_installer() -> Path:
    if shutil.which("hdiutil") is None:
        raise RuntimeError("hdiutil is required to build the macOS installer DMG.")

    app_bundle = build_app_bundle()
    if not app_bundle.exists():
        raise FileNotFoundError(f"Expected app bundle not found: {app_bundle}")
    distributable_app = prepare_distributable_bundle(app_bundle)

    staging_dir = _stage_installer(distributable_app)
    _strip_metadata(staging_dir)
    dmg_path = DIST_DIR / DMG_NAME
    _safe_remove(dmg_path)

    subprocess.run(
        [
            "hdiutil",
            "create",
            "-volname",
            VOLUME_NAME,
            "-srcfolder",
            str(staging_dir),
            "-format",
            "UDZO",
            "-imagekey",
            "zlib-level=9",
            "-ov",
            str(dmg_path),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    return dmg_path


def main() -> int:
    try:
        dmg_path = build_installer()
    except Exception as exc:
        print(f"Failed to build installer: {exc}", file=sys.stderr)
        return 1
    print(f"Built installer: {dmg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
