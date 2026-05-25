"""
SHAP-based explanations for the Isolation Forest detector.

The detector is trained on the full engineered feature set (~184 columns:
rolling stats, lag/diff, EWMA, skew/kurt, cycle_norm). SHAP must score the
same feature space — but raw 184-bar plots are unreadable for end users,
so this module also aggregates contributions back to the underlying base
sensors and feature families for display.

Sign convention: SHAP values are computed against the negated raw anomaly
score (``-clf.score_samples``), so positive SHAP values consistently mean
"this feature pushed the sample towards anomaly".
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from src.models.isolation_forest import IsolationForestDetector

logger = logging.getLogger(__name__)

FEATURE_FAMILIES: tuple[str, ...] = (
    "roll_mean",
    "roll_std",
    "ewma",
    "lag",
    "diff",
    "skew",
    "kurt",
    "raw",
    "other",
)

_FAMILY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("roll_mean", re.compile(r"_roll_mean_\d+$")),
    ("roll_std", re.compile(r"_roll_std_\d+$")),
    ("ewma", re.compile(r"_ewma_\d+$")),
    ("lag", re.compile(r"_lag_\d+$")),
    ("diff", re.compile(r"_diff_\d+$")),
    ("skew", re.compile(r"_skew_\d+$")),
    ("kurt", re.compile(r"_kurt_\d+$")),
]


@dataclass
class ShapExplanation:
    """Container for SHAP values plus aggregated views for display."""

    feature_names: list[str]
    shap_values: np.ndarray
    expected_value: float
    per_feature: pd.DataFrame
    per_sensor: pd.DataFrame
    per_family: pd.DataFrame


def _classify_feature(name: str) -> tuple[str, str]:
    """Return ``(base_sensor, family)`` for an engineered feature name."""
    for family, pattern in _FAMILY_PATTERNS:
        match = pattern.search(name)
        if match:
            base = name[: match.start()]
            return base, family
    if name.startswith("sensor_"):
        return name, "raw"
    return name, "other"


def build_explainer(
    detector: IsolationForestDetector,
    background: np.ndarray,
    max_background: int = 200,
    seed: int = 42,
) -> shap.TreeExplainer:
    """
    Build a ``TreeExplainer`` for the Isolation Forest.

    A small background sample (default 200 healthy rows) is enough for
    TreeExplainer — it uses the trees' structure, not the background, to
    compute exact Shapley values. The background sets the expected-value
    baseline.
    """
    if not detector.is_fitted:
        raise ValueError("Detector must be fitted before building explainer.")

    rng = np.random.default_rng(seed)
    if background.shape[0] > max_background:
        idx = rng.choice(background.shape[0], max_background, replace=False)
        background = background[idx]

    logger.info(
        "Building TreeExplainer (background n=%d, features=%d)",
        background.shape[0],
        background.shape[1],
    )
    return shap.TreeExplainer(
        detector.model,
        data=background,
        feature_perturbation="interventional",
    )


def explain(
    detector: IsolationForestDetector,
    X: np.ndarray,
    feature_names: list[str],
    background: np.ndarray | None = None,
    explainer: shap.TreeExplainer | None = None,
) -> ShapExplanation:
    """
    Compute SHAP values for ``X`` and aggregate by sensor + feature family.

    Either ``background`` or a pre-built ``explainer`` must be supplied.
    """
    if explainer is None:
        if background is None:
            raise ValueError("Provide either background or explainer.")
        explainer = build_explainer(detector, background)

    raw_values = explainer.shap_values(X, check_additivity=False)
    if isinstance(raw_values, list):
        raw_values = raw_values[0]
    raw_values = np.asarray(raw_values)

    # sklearn IsolationForest returns the *normality* score (higher = normal).
    # The detector exposes higher = more anomalous via negation, so flip SHAP
    # to match. Now positive SHAP = pushes sample towards anomaly.
    shap_values = -raw_values
    expected_value = float(-np.asarray(explainer.expected_value).ravel()[0])

    per_feature = _per_feature_table(shap_values, feature_names)
    per_sensor, per_family = _aggregate_tables(per_feature)

    return ShapExplanation(
        feature_names=feature_names,
        shap_values=shap_values,
        expected_value=expected_value,
        per_feature=per_feature,
        per_sensor=per_sensor,
        per_family=per_family,
    )


def _per_feature_table(
    shap_values: np.ndarray, feature_names: list[str]
) -> pd.DataFrame:
    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)
    bases, families = zip(*(_classify_feature(name) for name in feature_names))
    return pd.DataFrame(
        {
            "feature": feature_names,
            "base_sensor": bases,
            "family": families,
            "mean_abs_shap": mean_abs,
            "mean_signed_shap": mean_signed,
        }
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)


def _aggregate_tables(
    per_feature: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_sensor = (
        per_feature.groupby("base_sensor", as_index=False)
        .agg(
            total_abs_shap=("mean_abs_shap", "sum"),
            mean_signed_shap=("mean_signed_shap", "sum"),
            n_features=("feature", "count"),
        )
        .sort_values("total_abs_shap", ascending=False, ignore_index=True)
    )
    per_family = (
        per_feature.groupby("family", as_index=False)
        .agg(
            total_abs_shap=("mean_abs_shap", "sum"),
            mean_signed_shap=("mean_signed_shap", "sum"),
            n_features=("feature", "count"),
        )
        .sort_values("total_abs_shap", ascending=False, ignore_index=True)
    )
    return per_sensor, per_family


def top_features_for_sample(
    explanation: ShapExplanation,
    sample_index: int,
    k: int = 10,
) -> pd.DataFrame:
    """
    Return the top ``k`` SHAP contributors for a single sample, sorted by
    signed contribution (most anomaly-pushing first).
    """
    values = explanation.shap_values[sample_index]
    df = pd.DataFrame(
        {
            "feature": explanation.feature_names,
            "shap_value": values,
            "abs_shap": np.abs(values),
        }
    )
    return (
        df.sort_values("abs_shap", ascending=False)
        .head(k)
        .sort_values("shap_value", ascending=False, ignore_index=True)
        .drop(columns="abs_shap")
    )
