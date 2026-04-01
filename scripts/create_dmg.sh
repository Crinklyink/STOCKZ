#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${PROJECT_ROOT}/dist"
APP_NAME="Stock Predictor.app"
DMG_NAME="Stock Predictor.dmg"
APP_PATH="${DIST_DIR}/${APP_NAME}"
DMG_PATH="${DIST_DIR}/${DMG_NAME}"
STAGING_DIR="${DIST_DIR}/dmg_staging"

if [ ! -d "${APP_PATH}" ]; then
    echo "Error: ${APP_PATH} does not exist. Please run 'python3 build_mac.py' first to compile the App bundle."
    exit 1
fi

if [ -f "${DMG_PATH}" ]; then
    echo "Removing old DMG..."
    rm "${DMG_PATH}"
fi

echo "Staging DMG contents..."
rm -rf "${STAGING_DIR}"
mkdir -p "${STAGING_DIR}"

# Copy the app bundle
cp -R "${APP_PATH}" "${STAGING_DIR}/"

# Create a symlink to /Applications for standard drag-and-drop installation
ln -s /Applications "${STAGING_DIR}/Applications"

echo "Creating compressed DMG image via hdiutil..."
hdiutil create -volname "Stock Predictor" -srcfolder "${STAGING_DIR}" -ov -format UDZO "${DMG_PATH}"

rm -rf "${STAGING_DIR}"
echo "----------------------------------------"
echo "Success! Your distributable DMG is ready:"
echo "-> ${DMG_PATH}"
