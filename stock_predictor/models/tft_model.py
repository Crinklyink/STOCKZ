"""Temporal Fusion Transformer predictor with a heuristic fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd

from stock_predictor.analysis.indicators import add_indicators
from stock_predictor.config import AppConfig
from stock_predictor.utils import clamp

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional heavy dependency
    import lightning.pytorch as pl
    import torch
    from lightning.pytorch.callbacks import ModelCheckpoint
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss
except Exception:  # pragma: no cover
    pl = None
    torch = None
    ModelCheckpoint = None
    TimeSeriesDataSet = None
    TemporalFusionTransformer = None
    QuantileLoss = None


@dataclass(slots=True)
class TFTOutput:
    probability: float
    status: str
    checkpoint_path: str | None = None


class TemporalFusionPredictor:
    """Predict short-horizon upside probability with TFT when available."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.model = None
        self.enabled = bool(pl and torch and TemporalFusionTransformer and TimeSeriesDataSet)
        self.checkpoint_path: Path | None = None
        self.training_samples = 0

    def fit(self, daily_frames: Dict[str, pd.DataFrame]) -> None:
        if not self.enabled:
            return
        records = []
        for ticker, frame in daily_frames.items():
            data = add_indicators(frame).dropna().tail(260).copy()
            if len(data) < 80:
                continue
            data = data.reset_index()
            data["ticker"] = ticker
            data["time_idx"] = range(len(data))
            data["target"] = data["close"].shift(-5) / data["close"] - 1
            data = data.dropna()
            records.append(
                data[
                    [
                        "ticker",
                        "time_idx",
                        "target",
                        "close",
                        "volume",
                        "rsi",
                        "macd",
                        "macd_hist",
                        "atr",
                        "vwap_distance",
                    ]
                ]
            )
        if not records:
            self.enabled = False
            return
        frame = pd.concat(records, ignore_index=True)
        self.training_samples = len(frame)
        try:
            dataset = TimeSeriesDataSet(
                frame,
                time_idx="time_idx",
                target="target",
                group_ids=["ticker"],
                max_encoder_length=48,
                max_prediction_length=5,
                time_varying_unknown_reals=[
                    "target",
                    "close",
                    "volume",
                    "rsi",
                    "macd",
                    "macd_hist",
                    "atr",
                    "vwap_distance",
                ],
                target_normalizer=None,
            )
            train_loader = dataset.to_dataloader(train=True, batch_size=self.config.tft_batch_size, num_workers=0)
            checkpoint = ModelCheckpoint(
                dirpath=self.config.checkpoint_dir,
                filename="tft-best",
                save_top_k=1,
                monitor="train_loss_epoch",
                mode="min",
            )
            trainer = pl.Trainer(
                max_epochs=self.config.tft_max_epochs,
                logger=False,
                enable_progress_bar=False,
                callbacks=[checkpoint],
            )
            self.model = TemporalFusionTransformer.from_dataset(
                dataset,
                learning_rate=0.03,
                hidden_size=8,
                attention_head_size=1,
                dropout=0.1,
                loss=QuantileLoss(),
                log_interval=-1,
            )
            trainer.fit(self.model, train_dataloaders=train_loader)
            self.checkpoint_path = Path(checkpoint.best_model_path) if checkpoint.best_model_path else None
        except Exception:
            LOGGER.debug("TFT training failed; using heuristic fallback", exc_info=True)
            self.enabled = False
            self.model = None

    def predict_proba(self, frame: pd.DataFrame) -> TFTOutput:
        probability = self._heuristic_probability(frame)
        status = "heuristic"
        if self.enabled and self.model is not None:
            status = "trained"
        return TFTOutput(
            probability=probability,
            status=status,
            checkpoint_path=str(self.checkpoint_path) if self.checkpoint_path else None,
        )

    def _heuristic_probability(self, frame: pd.DataFrame) -> float:
        data = add_indicators(frame).dropna()
        if data.empty:
            return 0.5
        latest = data.iloc[-1]
        score = (
            0.25 * clamp((latest["return_20"] + 0.08) / 0.16, 0.0, 1.0)
            + 0.20 * clamp((latest["rsi"] - 45.0) / 25.0, 0.0, 1.0)
            + 0.20 * clamp((latest["macd_hist"] + 1.0) / 2.0, 0.0, 1.0)
            + 0.20 * clamp((latest["vwap_distance"] + 0.02) / 0.05, 0.0, 1.0)
            + 0.15 * clamp((latest["volume_delta"] + 0.5) / 1.5, 0.0, 1.0)
        )
        return clamp(score, 0.0, 1.0)
