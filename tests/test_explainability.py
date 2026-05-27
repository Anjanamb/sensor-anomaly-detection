"""Smoke tests for SHAP-based explainability on the Isolation Forest."""

import sys

import numpy as np

sys.path.insert(0, ".")

from src.explainability import (
    FEATURE_FAMILIES,
    ShapExplanation,
    _classify_feature,
    explain,
    feature_glossary,
    narrate,
    pretty_feature_label,
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


# ── Plain-English layer ─────────────────────────────────────────────────


def test_pretty_feature_label_covers_each_family():
    assert pretty_feature_label("sensor_2_roll_mean_10") == \
        "Sensor 2 — 10-cycle moving average"
    assert pretty_feature_label("sensor_4_roll_std_10") == \
        "Sensor 4 — volatility (10-cycle window)"
    assert pretty_feature_label("sensor_9_ewma_5") == \
        "Sensor 9 — fast-reacting average (span 5)"
    assert pretty_feature_label("sensor_4_lag_1") == \
        "Sensor 4 — value 1 cycles ago"
    assert pretty_feature_label("sensor_9_diff_5") == \
        "Sensor 9 — change over 5 cycles"
    assert pretty_feature_label("sensor_7_skew_20") == \
        "Sensor 7 — 20-cycle distribution skew"
    assert pretty_feature_label("sensor_7_kurt_20") == \
        "Sensor 7 — 20-cycle tail behaviour"


def test_pretty_feature_label_handles_raw_and_lifecycle():
    assert pretty_feature_label("sensor_2") == "Sensor 2 (raw reading)"
    assert pretty_feature_label("cycle_norm") == \
        "Lifecycle position (0 = new, 1 = end of life)"


def test_feature_glossary_returns_table():
    g = feature_glossary()
    assert len(g) >= 8
    assert list(g.columns) == ["pattern", "label", "meaning"]
    assert any("Sensor N" in row for row in g["label"])
    assert any("cycle_norm" in row for row in g["pattern"])


def _make_explanation(values, feature_names):
    """Synthetic ShapExplanation for narrate tests."""
    import pandas as pd
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    per_feature = pd.DataFrame({"feature": feature_names})  # not used by narrate
    per_sensor = pd.DataFrame()
    per_family = pd.DataFrame()
    return ShapExplanation(
        feature_names=list(feature_names),
        shap_values=arr,
        expected_value=0.0,
        per_feature=per_feature,
        per_sensor=per_sensor,
        per_family=per_family,
    )


def test_narrate_identifies_top_sensors_and_families():
    names = [
        "sensor_9_diff_1", "sensor_9_diff_5",
        "sensor_14_roll_std_10", "sensor_4_ewma_5",
        "sensor_2_roll_mean_5",
    ]
    # sensor_9 dominates via diff (change), sensor_14 via roll_std (volatility)
    values = [0.5, 0.3, 0.4, 0.05, -0.1]
    expl = _make_explanation(values, names)
    s = narrate(expl, sample_index=0, k=5)
    assert "Sensor 9" in s
    assert "Sensor 14" in s
    assert "changing rapidly" in s
    assert "volatile" in s


def test_narrate_falls_back_when_all_negative():
    names = ["sensor_9_diff_1", "sensor_14_roll_std_10"]
    values = [-0.3, -0.2]
    expl = _make_explanation(values, names)
    s = narrate(expl, sample_index=0, k=5)
    assert "No single feature dominates" in s


def test_narrate_appends_lifecycle_clause_when_strong():
    names = ["sensor_9_diff_5", "cycle_norm"]
    values = [0.5, 0.3]  # cycle_norm is 60% of the leader → above 30% threshold
    expl = _make_explanation(values, names)
    s = narrate(expl, sample_index=0, k=5)
    assert "Sensor 9" in s
    assert "late in its lifecycle" in s
