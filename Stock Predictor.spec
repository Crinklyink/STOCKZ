# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['/Users/crinklyink/Desktop/idk project/app.py'],
    pathex=[],
    binaries=[],
    datas=[('/Users/crinklyink/Desktop/idk project/assets', 'assets'), ('/Users/crinklyink/Desktop/idk project/.env.example', '.')],
    hiddenimports=['requests', 'pandas', 'xgboost'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Stock Predictor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['/Users/crinklyink/Desktop/idk project/assets/icon.icns'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Stock Predictor',
)
app = BUNDLE(
    coll,
    name='Stock Predictor.app',
    icon='/Users/crinklyink/Desktop/idk project/assets/icon.icns',
    bundle_identifier=None,
)
