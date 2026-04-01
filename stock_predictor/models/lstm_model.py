"""Attention-based LSTM for short-horizon probability estimates."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from stock_predictor.analysis.indicators import add_indicators
from stock_predictor.utils import clamp

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional heavy dependency
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


FEATURE_COLUMNS = [
    "close",
    "volume",
    "return_1",
    "return_5",
    "macd",
    "macd_hist",
    "rsi",
    "atr",
    "mfi",
    "obv",
    "vwap_distance",
]


@dataclass(slots=True)
class LSTMOutput:
    probability: float
    status: str


if nn is not None:  # pragma: no branch
    class AttentionLSTM(nn.Module):
        def __init__(self, input_size: int, hidden_size: int = 48) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=2,
                dropout=0.2,
                batch_first=True,
            )
            self.attention = nn.Linear(hidden_size, 1)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size // 2, 1),
            )

        def forward(self, x):  # type: ignore[no-untyped-def]
            seq, _ = self.lstm(x)
            weights = torch.softmax(self.attention(seq).squeeze(-1), dim=1)
            context = (seq * weights.unsqueeze(-1)).sum(dim=1)
            return self.classifier(context).squeeze(-1)


class LSTMPredictor:
    """Train a compact attention LSTM across the ticker universe."""

    def __init__(self, sequence_length: int = 48, horizon_bars: int = 35) -> None:
        self.sequence_length = sequence_length
        self.horizon_bars = horizon_bars
        self.model = None
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.enabled = torch is not None
        self.training_samples = 0

    def fit(self, hourly_frames: Dict[str, pd.DataFrame]) -> None:
        if not self.enabled:
            return
        sequences, labels = self._build_training_set(hourly_frames)
        self.training_samples = len(sequences)
        if len(sequences) < 200:
            LOGGER.info("Skipping LSTM training; not enough sequences (%s)", len(sequences))
            self.enabled = False
            return
        features = np.asarray(sequences, dtype=np.float32)
        labels_arr = np.asarray(labels, dtype=np.float32)
        self.feature_mean = features.mean(axis=(0, 1))
        self.feature_std = features.std(axis=(0, 1)) + 1e-6
        normalized = (features - self.feature_mean) / self.feature_std
        dataset = TensorDataset(
            torch.tensor(normalized, dtype=torch.float32),
            torch.tensor(labels_arr, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        self.model = AttentionLSTM(input_size=normalized.shape[-1])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = nn.BCEWithLogitsLoss()
        self.model.train()
        for _ in range(3):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()

    def predict_proba(self, frame: pd.DataFrame) -> LSTMOutput:
        if not self.enabled or self.model is None or self.feature_mean is None or self.feature_std is None:
            return LSTMOutput(probability=self._heuristic_probability(frame), status="heuristic")
        features = self._latest_sequence(frame)
        if features is None:
            return LSTMOutput(probability=self._heuristic_probability(frame), status="heuristic")
        normalized = (features - self.feature_mean) / self.feature_std
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(normalized[None, :, :], dtype=torch.float32))
            probability = torch.sigmoid(logits).item()
        return LSTMOutput(probability=clamp(float(probability), 0.0, 1.0), status="trained")

    def _build_training_set(self, frames: Dict[str, pd.DataFrame]) -> Tuple[List[np.ndarray], List[int]]:
        sequences: List[np.ndarray] = []
        labels: List[int] = []
        for ticker, frame in frames.items():
            indicator_frame = add_indicators(frame).dropna()
            if len(indicator_frame) < self.sequence_length + self.horizon_bars + 5:
                continue
            indicator_frame["future_return"] = (
                indicator_frame["close"].shift(-self.horizon_bars) / indicator_frame["close"] - 1
            )
            indicator_frame["target"] = (indicator_frame["future_return"] >= 0.04).astype(int)
            samples = 0
            for index in range(self.sequence_length, len(indicator_frame) - self.horizon_bars, 8):
                window = indicator_frame.iloc[index - self.sequence_length : index]
                target = int(indicator_frame.iloc[index]["target"])
                if window[FEATURE_COLUMNS].isna().any().any():
                    continue
                sequences.append(window[FEATURE_COLUMNS].to_numpy())
                labels.append(target)
                samples += 1
                if samples >= 250:
                    break
        return sequences, labels

    def _latest_sequence(self, frame: pd.DataFrame) -> np.ndarray | None:
        indicator_frame = add_indicators(frame).dropna()
        if len(indicator_frame) < self.sequence_length:
            return None
        return indicator_frame[FEATURE_COLUMNS].tail(self.sequence_length).to_numpy(dtype=np.float32)

    def _heuristic_probability(self, frame: pd.DataFrame) -> float:
        indicator_frame = add_indicators(frame).dropna()
        if indicator_frame.empty:
            return 0.5
        latest = indicator_frame.iloc[-1]
        score = (
            0.25 * clamp((latest["rsi"] - 45.0) / 25.0, 0.0, 1.0)
            + 0.25 * clamp((latest["macd_hist"] + 1.0) / 2.0, 0.0, 1.0)
            + 0.2 * clamp((latest["vwap_distance"] + 0.03) / 0.06, 0.0, 1.0)
            + 0.15 * clamp((latest["return_5"] + 0.05) / 0.10, 0.0, 1.0)
            + 0.15 * clamp((latest["volume_delta"] + 0.5) / 1.5, 0.0, 1.0)
        )
        return clamp(float(score), 0.0, 1.0)
