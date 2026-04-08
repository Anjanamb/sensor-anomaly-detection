"""
Preprocessing pipeline for sensor time-series data.
Handles cleaning, normalization, and train/test splitting.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


def remove_constant_sensors(
    df: pd.DataFrame, sensor_cols: list[str], threshold: float = 1e-6
) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove sensor columns with near-zero variance (constant readings).
    Returns filtered dataframe and list of kept sensor columns.
    """
    variances = df[sensor_cols].var()
    kept = variances[variances > threshold].index.tolist()
    dropped = set(sensor_cols) - set(kept)

    if dropped:
        logger.info(f"Dropped constant sensors: {dropped}")

    return df.drop(columns=list(dropped)), kept


def normalize_per_unit(
    df: pd.DataFrame,
    sensor_cols: list[str],
    method: str = "minmax",
) -> pd.DataFrame:
    """
    Normalize sensor readings per engine unit to account for
    different operating conditions across units.
    """
    df = df.copy()
    scaler_cls = MinMaxScaler if method == "minmax" else StandardScaler

    for unit_id in df["unit_id"].unique():
        mask = df["unit_id"] == unit_id
        scaler = scaler_cls()
        df.loc[mask, sensor_cols] = scaler.fit_transform(
            df.loc[mask, sensor_cols]
        )

    return df


def normalize_global(
    df: pd.DataFrame,
    sensor_cols: list[str],
    method: str = "standard",
    scaler: Optional[object] = None,
) -> tuple[pd.DataFrame, object]:
    """
    Global normalization across all units.
    Returns the dataframe and fitted scaler (for reuse on test data).
    """
    df = df.copy()
    scaler_cls = MinMaxScaler if method == "minmax" else StandardScaler

    if scaler is None:
        scaler = scaler_cls()
        df[sensor_cols] = scaler.fit_transform(df[sensor_cols])
    else:
        df[sensor_cols] = scaler.transform(df[sensor_cols])

    return df, scaler


def clip_rul(df: pd.DataFrame, max_rul: int = 125) -> pd.DataFrame:
    """
    Clip RUL values at a maximum. Early cycles where the engine is
    perfectly healthy all get the same max RUL — the degradation signal
    only matters closer to failure.
    """
    df = df.copy()
    df["rul"] = df["rul"].clip(upper=max_rul)
    return df


def create_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    sequence_length: int = 30,
    target_col: str = "anomaly",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for each engine unit.
    Returns X (n_samples, seq_len, n_features) and y (n_samples,).
    """
    X_list, y_list = [], []

    for unit_id in df["unit_id"].unique():
        unit_data = df[df["unit_id"] == unit_id].sort_values("cycle")
        features = unit_data[feature_cols].values
        targets = unit_data[target_col].values

        for i in range(sequence_length, len(features)):
            X_list.append(features[i - sequence_length : i])
            y_list.append(targets[i])

    X = np.array(X_list)
    y = np.array(y_list)

    logger.info(
        f"Created {len(X)} sequences of length {sequence_length}, "
        f"anomaly ratio: {y.mean():.3f}"
    )

    return X, y


def train_test_split_by_unit(
    df: pd.DataFrame, test_ratio: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by engine unit (not by row) to avoid data leakage.
    """
    rng = np.random.RandomState(seed)
    units = df["unit_id"].unique()
    rng.shuffle(units)

    split_idx = int(len(units) * (1 - test_ratio))
    train_units = units[:split_idx]
    test_units = units[split_idx:]

    train_df = df[df["unit_id"].isin(train_units)].copy()
    test_df = df[df["unit_id"].isin(test_units)].copy()

    logger.info(
        f"Split: {len(train_units)} train units, "
        f"{len(test_units)} test units"
    )

    return train_df, test_df
