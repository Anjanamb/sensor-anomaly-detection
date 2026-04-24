"""Tests for anomaly detection models."""
import numpy as np
import sys
sys.path.insert(0, ".")

from src.models import IsolationForestDetector, AutoencoderDetector, OneClassSVMDetector


def make_data():
    np.random.seed(42)
    X_normal = np.random.randn(200, 10)
    X_anomaly = np.random.randn(20, 10) + 5
    return X_normal, X_anomaly


def test_isolation_forest():
    X_normal, X_anomaly = make_data()
    model = IsolationForestDetector(contamination=0.1, n_estimators=50)
    model.fit(X_normal)

    preds = model.predict(X_anomaly)
    assert preds.shape == (20,)
    assert preds.sum() > 10  # Most anomalies should be detected

    scores = model.score_samples(X_anomaly)
    assert scores.shape == (20,)


def test_autoencoder():
    X_normal, X_anomaly = make_data()
    model = AutoencoderDetector(input_dim=10, encoding_dim=4, epochs=10, batch_size=64)
    model.fit(X_normal)

    preds = model.predict(X_anomaly)
    assert preds.shape == (20,)

    scores = model.score_samples(X_anomaly)
    normal_scores = model.score_samples(X_normal)
    # Anomaly scores should be higher on average
    assert scores.mean() > normal_scores.mean()


def test_one_class_svm():
    X_normal, X_anomaly = make_data()
    model = OneClassSVMDetector(nu=0.1)
    model.fit(X_normal)

    preds = model.predict(X_anomaly)
    assert preds.shape == (20,)
    assert preds.sum() > 10