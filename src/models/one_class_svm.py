"""
One-Class SVM anomaly detection model.
"""

import logging

import numpy as np
from sklearn.svm import OneClassSVM
import joblib

logger = logging.getLogger(__name__)


class OneClassSVMDetector:
    """
    One-Class SVM learns a boundary around normal data.
    Points outside the boundary are flagged as anomalies.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        gamma: str = "scale",
        nu: float = 0.1,
    ):
        self.model = OneClassSVM(
            kernel=kernel,
            gamma=gamma,
            nu=nu,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> "OneClassSVMDetector":
        """
        Fit on healthy data. Note: O(n^2) memory, so subsample
        if dataset is very large.
        """
        if X.shape[0] > 10000:
            logger.warning(
                f"Large dataset ({X.shape[0]} samples). "
                f"Subsampling to 10000 for One-Class SVM."
            )
            idx = np.random.RandomState(42).choice(
                X.shape[0], 10000, replace=False
            )
            X = X[idx]

        logger.info(f"Fitting One-Class SVM on {X.shape[0]} samples...")
        self.model.fit(X)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns binary predictions: 1 = anomaly, 0 = normal."""
        raw = self.model.predict(X)
        return (raw == -1).astype(int)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Returns anomaly scores. Negated so higher = more anomalous.
        """
        return -self.model.score_samples(X)

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> "OneClassSVMDetector":
        self.model = joblib.load(path)
        self.is_fitted = True
        logger.info(f"Model loaded from {path}")
        return self
