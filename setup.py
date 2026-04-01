"""py2app packaging entrypoint for the PySide6 Stock Predictor."""

from setuptools import setup
import py2app.recipes.qt6


# Force py2app to stay on the PySide6 recipe even if PyQt6 is installed
# elsewhere on the machine.
py2app.recipes.qt6.check = lambda cmd, mf: None


APP = ["app.py"]
OPTIONS = {
    "argv_emulation": True,
    "iconfile": "assets/icon.icns",
    "packages": [
        "PySide6",
        "pandas",
        "numpy",
    ],
    "includes": ["shiboken6"],
    "excludes": [
        "PyQt6",
        "PyQt6-Charts",
        "PyQt6_sip",
        "torch",
        "transformers",
        "xgboost",
        "lightgbm",
        "sklearn",
        "scipy",
        "matplotlib",
        "streamlit",
        "plotly",
        "openai",
        "praw",
        "ta_lib",
        "talib",
        "lightning",
        "pytorch_forecasting",
    ],
    "plist": {
        "CFBundleName": "Stock Predictor",
        "CFBundleDisplayName": "Stock Predictor",
        "CFBundleVersion": "3.0",
        "CFBundleShortVersionString": "3.0",
        "NSHighResolutionCapable": True,
    },
}


setup(app=APP, options={"py2app": OPTIONS})
