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

    A custom ``threshold`` (e.g. F1-optimal from the PR curve) can be
    attached after training; if set, ``predict()`` uses
    ``score_samples > threshold`` instead of sklearn's contamination
    cutoff. The threshold is persisted by ``save()`` / ``load()``.
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
        self.threshold: Optional[float] = None

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        """Fit on (ideally) healthy data."""
        logger.info(f"Fitting Isolation Forest on {X.shape[0]} samples...")
        self.model.fit(X)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns binary predictions: 1 = anomaly, 0 = normal.

        Uses the stored F1-optimal ``threshold`` if set; otherwise falls
        back to sklearn's contamination-based decision boundary.
        """
        if self.threshold is not None:
            return (self.score_samples(X) > self.threshold).astype(int)
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
        joblib.dump(
            {"model": self.model, "threshold": self.threshold}, path
        )
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> "IsolationForestDetector":
        artefact = joblib.load(path)
        if isinstance(artefact, dict):
            self.model = artefact["model"]
            self.threshold = artefact.get("threshold")
        else:
            # Back-compat: older saves stored the bare sklearn model
            self.model = artefact
            self.threshold = None
        self.is_fitted = True
        logger.info(f"Model loaded from {path}")
        return self
