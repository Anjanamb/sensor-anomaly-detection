"""
Per-regime sensor normalisation for C-MAPSS multi-condition subsets.

On FD002/FD004 each sensor reads wildly different depending on which
operating regime the engine is in (cruise vs takeoff vs idle). Treating
all readings on the same scale destroys the anomaly signal. The fix:
cluster the operating-settings columns into N regimes via KMeans, fit
a separate ``StandardScaler`` on each regime's *healthy* training rows,
then apply each row's regime scaler to its sensor values.

Single-regime subsets (FD001, FD003) bypass this — use ``normalize_global``
from ``src.preprocessing`` instead.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

OP_SETTING_COLS: tuple[str, ...] = (
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
)


def fit_regime_normalisation(
    df: pd.DataFrame,
    sensor_cols: list[str],
    n_regimes: int,
    anomaly_col: str = "anomaly",
    seed: int = 42,
) -> tuple[pd.DataFrame, KMeans, list[StandardScaler]]:
    """
    Cluster operating settings into ``n_regimes``, fit a per-regime
    ``StandardScaler`` on healthy rows, and return the normalised df
    alongside the fitted ``kmeans`` and the list of scalers (indexed by
    regime id).

    The returned df has its sensor columns overwritten with normalised
    values and a new ``regime`` column with the cluster label.

    Caller is responsible for casting sensor columns to float64 before
    calling this (some C-MAPSS sensors are int64-typed; the scaler output
    is float and won't write back to int columns).
    """
    df = df.copy()
    op_values = df[list(OP_SETTING_COLS)].values
    kmeans = KMeans(n_clusters=n_regimes, n_init=10, random_state=seed).fit(
        op_values
    )
    df["regime"] = kmeans.labels_

    sensor_values = df[sensor_cols].values.copy()
    regime_labels = df["regime"].values
    anomaly_mask = df[anomaly_col].values == 0

    regime_scalers: list[StandardScaler] = []
    for r in range(n_regimes):
        fit_mask = (regime_labels == r) & anomaly_mask
        scaler = StandardScaler().fit(sensor_values[fit_mask])
        regime_scalers.append(scaler)
        # Apply to all rows of this regime (healthy + anomalous)
        sensor_values[regime_labels == r] = scaler.transform(
            sensor_values[regime_labels == r]
        )

    df[sensor_cols] = sensor_values
    logger.info(
        f"Per-regime normalisation: {n_regimes} clusters, "
        f"sizes {dict(zip(*np.unique(kmeans.labels_, return_counts=True)))}"
    )
    return df, kmeans, regime_scalers


def apply_regime_normalisation(
    df: pd.DataFrame,
    sensor_cols: list[str],
    kmeans: KMeans,
    regime_scalers: list[StandardScaler],
) -> pd.DataFrame:
    """
    Apply previously-fit per-regime normalisation to fresh data.

    Predicts each row's regime via ``kmeans``, then transforms the row's
    sensor values with that regime's scaler. Returns a copy of ``df`` with
    normalised sensor columns and an added ``regime`` column.
    """
    df = df.copy()
    op_values = df[list(OP_SETTING_COLS)].values
    df["regime"] = kmeans.predict(op_values)

    sensor_values = df[sensor_cols].astype(float).values.copy()
    regime_labels = df["regime"].values
    for r in range(len(regime_scalers)):
        mask = regime_labels == r
        if mask.any():
            sensor_values[mask] = regime_scalers[r].transform(
                sensor_values[mask]
            )
    df[sensor_cols] = sensor_values
    return df
