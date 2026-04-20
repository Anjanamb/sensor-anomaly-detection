"""
Autoencoder-based anomaly detection using PyTorch.
Trained to reconstruct normal patterns — high reconstruction error = anomaly.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class SensorAutoencoder(nn.Module):
    """
    Symmetric autoencoder for sensor data.
    Bottleneck forces compression → anomalies produce high reconstruction error.
    """

    def __init__(self, input_dim: int, encoding_dim: int = 16):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoencoderDetector:
    """
    Wrapper for training and using the autoencoder for anomaly detection.
    """

    def __init__(
        self,
        input_dim: int,
        encoding_dim: int = 16,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 256,
        threshold_percentile: float = 95.0,
        device: Optional[str] = None,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = SensorAutoencoder(input_dim, encoding_dim).to(
            self.device
        )
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold_percentile = threshold_percentile
        self.threshold: Optional[float] = None
        self.train_losses: list[float] = []

    def fit(self, X: np.ndarray) -> "AutoencoderDetector":
        """Train on healthy data."""
        logger.info(
            f"Training Autoencoder on {X.shape[0]} samples, "
            f"device={self.device}"
        )

        dataset = TensorDataset(
            torch.FloatTensor(X).to(self.device)
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                output = self.model(batch)
                loss = criterion(output, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch)

            avg_loss = epoch_loss / len(dataset)
            self.train_losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(f"  Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")

        # Set threshold based on training reconstruction errors
        train_errors = self.reconstruction_error(X)
        self.threshold = np.percentile(
            train_errors, self.threshold_percentile
        )
        logger.info(
            f"Threshold set at {self.threshold_percentile}th percentile: "
            f"{self.threshold:.6f}"
        )

        return self

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Compute per-sample MSE reconstruction error."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            X_reconstructed = self.model(X_tensor)
            errors = (
                (X_tensor - X_reconstructed).pow(2).mean(dim=1).cpu().numpy()
            )
        return errors

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns binary predictions: 1 = anomaly, 0 = normal."""
        errors = self.reconstruction_error(X)
        return (errors > self.threshold).astype(int)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Returns anomaly scores (reconstruction error)."""
        return self.reconstruction_error(X)

    def save(self, path: str) -> None:
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "threshold": self.threshold,
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> "AutoencoderDetector":
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.threshold = checkpoint["threshold"]
        logger.info(f"Model loaded from {path}")
        return self
