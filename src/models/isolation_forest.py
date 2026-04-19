"""
Isolation Forest anomaly detection model.
"""

import logging
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

logger = logging.getLogger(__name__)


class IsolationForestDetector:
    """
    Isolation Forest-based anomaly detector.
    Trained on healthy data (low contamination), detects anomalies
    as points that are easy to isolate.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 200,
        max_samples: str = "auto",
        random_state: int = 42,
    ):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        """Fit on (ideally) healthy data."""
        logger.info(f"Fitting Isolation Forest on {X.shape[0]} samples...")
        self.model.fit(X)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns binary predictions: 1 = anomaly, 0 = normal."""
        raw = self.model.predict(X)
        # sklearn: -1 = anomaly, 1 = normal → convert to 1 = anomaly, 0 = normal
        return (raw == -1).astype(int)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Returns anomaly scores. More negative = more anomalous.
        We negate so higher = more anomalous (intuitive).
        """
        return -self.model.score_samples(X)

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> "IsolationForestDetector":
        self.model = joblib.load(path)
        self.is_fitted = True
        logger.info(f"Model loaded from {path}")
        return self
