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


# ── Plain-English layer ─────────────────────────────────────────────────

_FAMILY_PRETTY: dict[str, str] = {
    "roll_mean": "{}-cycle moving average",
    "roll_std": "volatility ({}-cycle window)",
    "ewma": "fast-reacting average (span {})",
    "lag": "value {} cycles ago",
    "diff": "change over {} cycles",
    "skew": "{}-cycle distribution skew",
    "kurt": "{}-cycle tail behaviour",
}

_FAMILY_PHRASE: dict[str, str] = {
    "roll_mean": "drifting from its baseline",
    "roll_std": "becoming more volatile",
    "ewma": "trending away from healthy recently",
    "lag": "differing from its recent past",
    "diff": "changing rapidly",
    "skew": "showing an asymmetric reading distribution",
    "kurt": "throwing more extreme spikes than usual",
    "raw": "off its healthy baseline",
}


def _parse_window(name: str, family: str) -> str | None:
    """Extract the trailing window/lag number from a feature name."""
    if family in {"roll_mean", "roll_std", "ewma", "lag", "diff", "skew", "kurt"}:
        # Format: ..._family_K → grab the digits after the final underscore
        tail = name.rsplit("_", 1)[-1]
        if tail.isdigit():
            return tail
    return None


def pretty_feature_label(name: str) -> str:
    """
    Render an engineered feature name as a human-readable label.

    Examples:
        sensor_9_diff_5      → "Sensor 9 — change over 5 cycles"
        sensor_4_roll_std_10 → "Sensor 4 — volatility (10-cycle window)"
        sensor_2             → "Sensor 2 (raw reading)"
        cycle_norm           → "Lifecycle position (0 = new, 1 = end of life)"
    """
    if name == "cycle_norm":
        return "Lifecycle position (0 = new, 1 = end of life)"

    base, family = _classify_feature(name)
    sensor_label = base.replace("sensor_", "Sensor ") if base.startswith("sensor_") else base

    if family == "raw":
        return f"{sensor_label} (raw reading)"
    if family == "other":
        return name  # unknown name → fall back to raw form

    window = _parse_window(name, family)
    if window is None:
        return f"{sensor_label} — {family}"
    return f"{sensor_label} — {_FAMILY_PRETTY[family].format(window)}"


def feature_glossary() -> pd.DataFrame:
    """
    Return the full mapping table that powers ``pretty_feature_label``.
    Useful for dashboard expanders and README examples.
    """
    rows = [
        ("sensor_N_roll_mean_K", "Sensor N — K-cycle moving average",
         "Smoothed average over the last K cycles. Reveals slow drift."),
        ("sensor_N_roll_std_K", "Sensor N — volatility (K-cycle window)",
         "Standard deviation over the last K cycles. Catches rising noise."),
        ("sensor_N_ewma_K", "Sensor N — fast-reacting average (span K)",
         "Exponentially weighted moving average; weights recent cycles more."),
        ("sensor_N_lag_K", "Sensor N — value K cycles ago",
         "Raw reading from K cycles in the past, exposed as a feature."),
        ("sensor_N_diff_K", "Sensor N — change over K cycles",
         "Difference between the current reading and the value K cycles ago."),
        ("sensor_N_skew_K", "Sensor N — K-cycle distribution skew",
         "Asymmetry of recent readings. Healthy = symmetric; drift skews it."),
        ("sensor_N_kurt_K", "Sensor N — K-cycle tail behaviour",
         "How often extreme readings appear. Spikes appear before mean drift."),
        ("sensor_N", "Sensor N (raw reading)",
         "Raw sensor value at the current cycle."),
        ("cycle_norm", "Lifecycle position (0 = new, 1 = end of life)",
         "Where the engine is in its run-to-failure trajectory."),
    ]
    return pd.DataFrame(rows, columns=["pattern", "label", "meaning"])


def narrate(
    explanation: ShapExplanation,
    sample_index: int,
    k: int = 5,
) -> str:
    """
    Build a one-sentence summary of the top anomaly-pushing SHAP
    contributors for a single sample.

    Groups positive SHAP values by base sensor, picks the top 2–3 sensors,
    describes each one's dominant family in plain English, and appends a
    lifecycle clause if ``cycle_norm`` is also a strong contributor.

    Falls back to a generic message if no positive contributions stand out.
    """
    values = explanation.shap_values[sample_index]
    names = explanation.feature_names

    contribs = pd.DataFrame({"feature": names, "shap": values})
    contribs = contribs[contribs["shap"] > 0].copy()
    if contribs.empty:
        return (
            "No single feature dominates — small contributions across many "
            "sensors push this sample just above the threshold."
        )

    contribs["base"], contribs["family"] = zip(
        *(_classify_feature(n) for n in contribs["feature"])
    )

    # Separate the lifecycle-context contributor
    lifecycle_row = contribs[contribs["base"] == "cycle_norm"]
    lifecycle_strength = float(lifecycle_row["shap"].sum()) if not lifecycle_row.empty else 0.0

    sensor_contribs = contribs[contribs["base"].str.startswith("sensor_")]
    if sensor_contribs.empty:
        if lifecycle_strength > 0:
            return "Flagged primarily because the engine is late in its lifecycle."
        return (
            "No single feature dominates — small contributions across many "
            "sensors push this sample just above the threshold."
        )

    # Top sensors by total positive contribution
    by_sensor = (
        sensor_contribs.groupby("base", as_index=False)["shap"]
        .sum()
        .sort_values("shap", ascending=False)
        .head(3)
    )
    if by_sensor.empty:
        return "Flagged by a diffuse pattern of small contributions."

    # Include the top 2 sensors unconditionally (when present); add a 3rd only
    # if it's within ~60% of the leader, so the sentence stays focused.
    leader = by_sensor.iloc[0]["shap"]
    cutoff = leader * 0.6
    keep_mask = (by_sensor["shap"] >= cutoff)
    keep_mask.iloc[: min(2, len(by_sensor))] = True
    top_sensors = by_sensor[keep_mask].head(3)

    clauses: list[str] = []
    for _, row in top_sensors.iterrows():
        base = row["base"]
        sensor_label = base.replace("sensor_", "Sensor ")
        # Dominant family for this sensor
        fam_rank = (
            sensor_contribs[sensor_contribs["base"] == base]
            .groupby("family")["shap"]
            .sum()
            .sort_values(ascending=False)
        )
        if fam_rank.empty:
            phrase = "deviating from its healthy baseline"
        else:
            top_family = fam_rank.index[0]
            phrase = _FAMILY_PHRASE.get(top_family, "deviating from its healthy baseline")
        clauses.append(f"**{sensor_label}** is {phrase}")

    if len(clauses) == 1:
        body = clauses[0]
    elif len(clauses) == 2:
        body = " and ".join(clauses)
    else:
        body = ", ".join(clauses[:-1]) + ", and " + clauses[-1]

    sentence = f"Flagged because {body}."

    # Append lifecycle clause if it's a meaningful contributor (≥30% of leader)
    if lifecycle_strength >= 0.3 * leader:
        sentence = sentence.rstrip(".") + "; the engine is also late in its lifecycle."

    return sentence
