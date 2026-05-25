"""Smoke tests for SHAP-based explainability on the Isolation Forest."""

import sys

import numpy as np

sys.path.insert(0, ".")

from src.explainability import (
    FEATURE_FAMILIES,
    _classify_feature,
    explain,
    top_features_for_sample,
)
from src.models.isolation_forest import IsolationForestDetector


def _toy_data(n_samples: int = 200, n_features: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_samples, n_features))


def _toy_feature_names():
    return [
        "sensor_2_roll_mean_10",
        "sensor_2_roll_std_10",
        "sensor_4_ewma_5",
        "sensor_4_lag_1",
        "sensor_7_skew_20",
        "sensor_7_kurt_20",
    ]


def test_classify_feature_recognises_families():
    assert _classify_feature("sensor_2_roll_mean_10") == ("sensor_2", "roll_mean")
    assert _classify_feature("sensor_4_lag_1") == ("sensor_4", "lag")
    assert _classify_feature("sensor_7_kurt_20") == ("sensor_7", "kurt")
    assert _classify_feature("sensor_9") == ("sensor_9", "raw")
    assert _classify_feature("cycle_norm") == ("cycle_norm", "other")


def test_explain_returns_aggregations():
    X = _toy_data()
    detector = IsolationForestDetector(n_estimators=20, contamination=0.1).fit(X)

    feature_names = _toy_feature_names()
    explanation = explain(detector, X[:50], feature_names, background=X)

    assert explanation.shap_values.shape == (50, len(feature_names))
    assert list(explanation.per_feature["feature"]) and (
        set(explanation.per_feature["feature"]) == set(feature_names)
    )
    assert set(explanation.per_sensor["base_sensor"]) == {
        "sensor_2",
        "sensor_4",
        "sensor_7",
    }
    assert set(explanation.per_family["family"]).issubset(set(FEATURE_FAMILIES))


def test_top_features_for_sample_returns_k_rows():
    X = _toy_data()
    detector = IsolationForestDetector(n_estimators=20).fit(X)
    explanation = explain(detector, X[:10], _toy_feature_names(), background=X)

    top = top_features_for_sample(explanation, sample_index=0, k=3)
    assert len(top) == 3
    assert list(top.columns) == ["feature", "shap_value"]
