"""
LSTM-based sequence autoencoder for sensor anomaly detection.

Unlike :class:`AutoencoderDetector`, which treats each cycle independently,
this model consumes sliding windows of length ``T`` over the raw sensor
channels. It reconstructs the full window, so the loss couples adjacent
cycles — degradation that manifests as a *sequential drift* (slow trend
over many cycles) shows up here as a reconstruction error rise, whereas
a feedforward AE is order-invariant by design and misses it.

The detector exposes the same surface as the other models:
``fit / predict / score_samples / save / load`` plus a ``threshold``
attribute (F1-optimal from the PR curve, set externally after training).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class LSTMSensorAutoencoder(nn.Module):
    """Seq2seq autoencoder. Encoder LSTM → linear bottleneck → decoder LSTM."""

    def __init__(
        self,
        n_sensors: int,
        seq_len: int,
        hidden_dim: int = 32,
        encoding_dim: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_sensors = n_sensors
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.encoding_dim = encoding_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=n_sensors,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.to_bottleneck = nn.Linear(hidden_dim, encoding_dim)
        self.from_bottleneck = nn.Linear(encoding_dim, hidden_dim)
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_dim, n_sensors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, n_sensors)
        _, (h_n, _) = self.encoder(x)
        # h_n: (num_layers, B, hidden_dim) — take the last layer's state
        last_hidden = h_n[-1]                       # (B, hidden_dim)
        z = self.to_bottleneck(last_hidden)         # (B, encoding_dim)
        dec_in_step = self.from_bottleneck(z)       # (B, hidden_dim)
        # Repeat the bottleneck as the decoder input across all T steps
        dec_in = dec_in_step.unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out, _ = self.decoder(dec_in)
        return self.output_layer(dec_out)


class LSTMAutoencoderDetector:
    """
    Wrapper around :class:`LSTMSensorAutoencoder` providing the project's
    standard detector API.

    The input is expected to be sliding windows of shape ``(N, T, n_sensors)``
    over raw sensor readings; build them via
    :func:`src.preprocessing.create_sequences`.
    """

    def __init__(
        self,
        n_sensors: int,
        seq_len: int = 30,
        hidden_dim: int = 32,
        encoding_dim: int = 8,
        num_layers: int = 2,
        lr: float = 1e-3,
        epochs: int = 150,
        batch_size: int = 256,
        threshold_percentile: float = 95.0,
        device: Optional[str] = None,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.n_sensors = n_sensors
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.encoding_dim = encoding_dim
        self.num_layers = num_layers
        self.model = LSTMSensorAutoencoder(
            n_sensors=n_sensors,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            encoding_dim=encoding_dim,
            num_layers=num_layers,
        ).to(self.device)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold_percentile = threshold_percentile
        self.threshold: Optional[float] = None
        self.train_losses: list[float] = []

    def fit(self, X: np.ndarray) -> "LSTMAutoencoderDetector":
        """Train on healthy windows of shape (N, T, n_sensors)."""
        if X.ndim != 3:
            raise ValueError(
                f"Expected 3D input (N, T, n_sensors), got shape {X.shape}"
            )

        logger.info(
            f"Training LSTM AE on {X.shape[0]} windows of length {X.shape[1]} "
            f"({X.shape[2]} sensors), device={self.device}"
        )

        dataset = TensorDataset(torch.FloatTensor(X))
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=10, factor=0.5
        )
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                output = self.model(batch)
                loss = criterion(output, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch)

            avg_loss = epoch_loss / len(dataset)
            self.train_losses.append(avg_loss)
            scheduler.step(avg_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"  Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}"
                )

        train_errors = self.score_samples(X)
        self.threshold = float(
            np.percentile(train_errors, self.threshold_percentile)
        )
        logger.info(
            f"Default percentile threshold: {self.threshold:.6f} "
            f"(overwrite with the F1-optimal value before saving)"
        )
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Per-window MSE reconstruction error."""
        if X.ndim != 3:
            raise ValueError(
                f"Expected 3D input (N, T, n_sensors), got shape {X.shape}"
            )
        self.model.eval()
        scores = []
        with torch.no_grad():
            for i in range(0, X.shape[0], self.batch_size):
                batch = torch.FloatTensor(X[i : i + self.batch_size]).to(
                    self.device
                )
                recon = self.model(batch)
                # Mean over (T, n_sensors) → one score per window
                err = (batch - recon).pow(2).mean(dim=(1, 2))
                scores.append(err.cpu().numpy())
        return np.concatenate(scores)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary predictions: 1 = anomaly, 0 = normal. Uses ``threshold``."""
        if self.threshold is None:
            raise RuntimeError(
                "Threshold not set. Either call fit() (which sets a default "
                "percentile threshold) or assign an F1-optimal value to "
                "self.threshold before predict()."
            )
        return (self.score_samples(X) > self.threshold).astype(int)

    def save(self, path: str) -> None:
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "threshold": self.threshold,
                "n_sensors": self.n_sensors,
                "seq_len": self.seq_len,
                "hidden_dim": self.hidden_dim,
                "encoding_dim": self.encoding_dim,
                "num_layers": self.num_layers,
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> "LSTMAutoencoderDetector":
        checkpoint = torch.load(
            path, map_location=self.device, weights_only=False
        )
        self.n_sensors = checkpoint["n_sensors"]
        self.seq_len = checkpoint["seq_len"]
        self.hidden_dim = checkpoint["hidden_dim"]
        self.encoding_dim = checkpoint["encoding_dim"]
        self.num_layers = checkpoint["num_layers"]
        self.model = LSTMSensorAutoencoder(
            n_sensors=self.n_sensors,
            seq_len=self.seq_len,
            hidden_dim=self.hidden_dim,
            encoding_dim=self.encoding_dim,
            num_layers=self.num_layers,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.threshold = checkpoint["threshold"]
        logger.info(f"Model loaded from {path}")
        return self
