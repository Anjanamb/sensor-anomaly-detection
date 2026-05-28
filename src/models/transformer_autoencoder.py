"""
Transformer-based sequence autoencoder for sensor anomaly detection.

A step up from :class:`LSTMAutoencoderDetector`. Instead of squeezing a
30-cycle window through a single recurrent hidden state, a self-attention
encoder lets every cycle attend to every other cycle directly, and a
learned positional embedding makes the model order-aware. A per-timestep
bottleneck (8 dims, below the 15 raw sensors) forces genuine compression,
so it stays a true autoencoder rather than learning the identity.

Same detector surface as the other models:
``fit / predict / score_samples / save / load`` plus a ``threshold``
attribute set externally to the F1-optimal point on the PR curve.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class TransformerSensorAutoencoder(nn.Module):
    """Attention encoder → per-timestep bottleneck → attention decoder."""

    def __init__(
        self,
        n_sensors: int,
        seq_len: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        bottleneck_dim: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_sensors = n_sensors
        self.seq_len = seq_len
        self.d_model = d_model

        self.input_proj = nn.Linear(n_sensors, d_model)
        # Learned positional embedding (fixed window length, so this is simple
        # and trains fine — sinusoidal would also work).
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm — more stable to train
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        self.to_bottleneck = nn.Linear(d_model, bottleneck_dim)
        self.from_bottleneck = nn.Linear(bottleneck_dim, d_model)

        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            dec_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.output_proj = nn.Linear(d_model, n_sensors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, n_sensors)
        h = self.input_proj(x) + self.pos_embed     # (B, T, d_model)
        h = self.encoder(h)                          # (B, T, d_model)
        z = self.to_bottleneck(h)                    # (B, T, bottleneck)
        d = self.from_bottleneck(z) + self.pos_embed  # (B, T, d_model)
        d = self.decoder(d)                          # (B, T, d_model)
        return self.output_proj(d)                   # (B, T, n_sensors)


class TransformerAutoencoderDetector:
    """
    Wrapper around :class:`TransformerSensorAutoencoder` providing the
    project's standard detector API. Input: sliding windows of shape
    ``(N, T, n_sensors)`` over raw sensor readings (build them with
    :func:`src.preprocessing.create_sequences`).
    """

    def __init__(
        self,
        n_sensors: int,
        seq_len: int = 30,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        bottleneck_dim: int = 8,
        lr: float = 1e-3,
        epochs: int = 120,
        batch_size: int = 256,
        threshold_percentile: float = 95.0,
        device: Optional[str] = None,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.n_sensors = n_sensors
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.bottleneck_dim = bottleneck_dim
        self.model = TransformerSensorAutoencoder(
            n_sensors=n_sensors,
            seq_len=seq_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            bottleneck_dim=bottleneck_dim,
        ).to(self.device)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold_percentile = threshold_percentile
        self.threshold: Optional[float] = None
        self.train_losses: list[float] = []

    def fit(self, X: np.ndarray) -> "TransformerAutoencoderDetector":
        """Train on healthy windows of shape (N, T, n_sensors)."""
        if X.ndim != 3:
            raise ValueError(
                f"Expected 3D input (N, T, n_sensors), got shape {X.shape}"
            )

        logger.info(
            f"Training Transformer AE on {X.shape[0]} windows of length "
            f"{X.shape[1]} ({X.shape[2]} sensors), device={self.device}"
        )

        dataset = TensorDataset(torch.FloatTensor(X))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=8, factor=0.5
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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
                "d_model": self.d_model,
                "nhead": self.nhead,
                "num_layers": self.num_layers,
                "dim_feedforward": self.dim_feedforward,
                "bottleneck_dim": self.bottleneck_dim,
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> "TransformerAutoencoderDetector":
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.n_sensors = ckpt["n_sensors"]
        self.seq_len = ckpt["seq_len"]
        self.d_model = ckpt["d_model"]
        self.nhead = ckpt["nhead"]
        self.num_layers = ckpt["num_layers"]
        self.dim_feedforward = ckpt["dim_feedforward"]
        self.bottleneck_dim = ckpt["bottleneck_dim"]
        self.model = TransformerSensorAutoencoder(
            n_sensors=self.n_sensors,
            seq_len=self.seq_len,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            bottleneck_dim=self.bottleneck_dim,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.threshold = ckpt["threshold"]
        logger.info(f"Model loaded from {path}")
        return self
