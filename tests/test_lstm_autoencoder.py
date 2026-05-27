"""Smoke tests for the LSTM autoencoder detector."""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from src.models.lstm_autoencoder import LSTMAutoencoderDetector


def _toy_windows(n_windows=64, seq_len=10, n_sensors=5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_windows, seq_len, n_sensors)).astype(np.float32)


def _make_detector(seq_len=10, n_sensors=5):
    return LSTMAutoencoderDetector(
        n_sensors=n_sensors,
        seq_len=seq_len,
        hidden_dim=16,
        encoding_dim=4,
        num_layers=1,
        epochs=2,
        batch_size=16,
        device="cpu",
    )


def test_fit_then_score_shapes():
    X = _toy_windows()
    det = _make_detector().fit(X)
    scores = det.score_samples(X)
    assert scores.shape == (X.shape[0],)
    assert np.all(scores >= 0)


def test_fit_sets_default_threshold():
    X = _toy_windows()
    det = _make_detector().fit(X)
    assert det.threshold is not None
    assert det.threshold > 0


def test_predict_uses_threshold():
    X = _toy_windows()
    det = _make_detector().fit(X)
    # Force a very large threshold → expect zero alarms
    det.threshold = float(det.score_samples(X).max() * 10)
    assert det.predict(X).sum() == 0
    # Force a very small threshold → expect all alarms
    det.threshold = float(det.score_samples(X).min() / 10)
    assert det.predict(X).sum() == X.shape[0]


def test_save_load_roundtrip_preserves_threshold():
    X = _toy_windows()
    det = _make_detector().fit(X)
    det.threshold = 0.1234

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "lstm.pt")
        det.save(path)

        det2 = LSTMAutoencoderDetector(
            n_sensors=5, seq_len=10, hidden_dim=16, encoding_dim=4,
            num_layers=1, device="cpu",
        ).load(path)

    assert det2.threshold == 0.1234
    # Reconstruction error of the loaded model should match the saved one
    np.testing.assert_allclose(
        det.score_samples(X), det2.score_samples(X), rtol=1e-5, atol=1e-6
    )


def test_rejects_2d_input():
    det = _make_detector()
    flat = np.zeros((32, 5), dtype=np.float32)
    try:
        det.fit(flat)
    except ValueError as e:
        assert "3D" in str(e)
    else:
        raise AssertionError("Expected ValueError for 2D input")
