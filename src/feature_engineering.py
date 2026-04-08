"""
Feature engineering for time-series sensor data.
Generates rolling statistics, spectral features, and lag features.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


def add_rolling_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    windows: list[int] = [5, 10, 20],
) -> pd.DataFrame:
    """
    Add rolling mean, std, min, max for each sensor at multiple windows.
    Computed per engine unit.
    """
    df = df.copy()
    new_cols = []

    for unit_id in df["unit_id"].unique():
        mask = df["unit_id"] == unit_id
        unit_data = df.loc[mask, sensor_cols]

        for w in windows:
            rolling = unit_data.rolling(window=w, min_periods=1)

            for stat_name, stat_func in [
                ("mean", rolling.mean),
                ("std", rolling.std),
            ]:
                result = stat_func()
                result.columns = [
                    f"{col}_roll_{stat_name}_{w}" for col in sensor_cols
                ]
                if unit_id == df["unit_id"].unique()[0]:
                    new_cols.extend(result.columns.tolist())
                df.loc[mask, result.columns] = result.values

    # Fill NaN from rolling windows
    df[new_cols] = df[new_cols].bfill()

    logger.info(f"Added {len(new_cols)} rolling features")
    return df


def add_lag_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    lags: list[int] = [1, 5, 10],
) -> pd.DataFrame:
    """
    Add lagged values and differences (rate of change) per engine unit.
    """
    df = df.copy()
    new_cols = []

    for unit_id in df["unit_id"].unique():
        mask = df["unit_id"] == unit_id

        for lag in lags:
            for col in sensor_cols:
                lag_col = f"{col}_lag_{lag}"
                diff_col = f"{col}_diff_{lag}"

                df.loc[mask, lag_col] = df.loc[mask, col].shift(lag)
                df.loc[mask, diff_col] = df.loc[mask, col].diff(lag)

                if unit_id == df["unit_id"].unique()[0]:
                    new_cols.extend([lag_col, diff_col])

    df[new_cols] = df[new_cols].fillna(0)
    logger.info(f"Added {len(new_cols)} lag features")
    return df


def add_ewma_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    spans: list[int] = [5, 15],
) -> pd.DataFrame:
    """
    Add exponentially weighted moving averages per engine unit.
    EWMA reacts faster to recent changes — useful for degradation detection.
    """
    df = df.copy()
    new_cols = []

    for unit_id in df["unit_id"].unique():
        mask = df["unit_id"] == unit_id

        for span in spans:
            ewma = df.loc[mask, sensor_cols].ewm(span=span).mean()
            ewma.columns = [f"{col}_ewma_{span}" for col in sensor_cols]
            if unit_id == df["unit_id"].unique()[0]:
                new_cols.extend(ewma.columns.tolist())
            df.loc[mask, ewma.columns] = ewma.values

    logger.info(f"Added {len(new_cols)} EWMA features")
    return df


def add_statistical_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    window: int = 20,
) -> pd.DataFrame:
    """
    Add higher-order statistical features (skewness, kurtosis) per window.
    These capture distributional shifts that signal degradation.
    """
    df = df.copy()
    new_cols = []

    for unit_id in df["unit_id"].unique():
        mask = df["unit_id"] == unit_id

        for col in sensor_cols:
            skew_col = f"{col}_skew_{window}"
            kurt_col = f"{col}_kurt_{window}"

            series = df.loc[mask, col]
            df.loc[mask, skew_col] = (
                series.rolling(window, min_periods=1).skew()
            )
            df.loc[mask, kurt_col] = (
                series.rolling(window, min_periods=1).kurt()
            )

            if unit_id == df["unit_id"].unique()[0]:
                new_cols.extend([skew_col, kurt_col])

    df[new_cols] = df[new_cols].fillna(0)
    logger.info(f"Added {len(new_cols)} statistical features")
    return df


def add_cycle_normalized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a normalized cycle feature (0 to 1) per engine unit.
    Represents how far through its lifecycle the engine is.
    """
    df = df.copy()
    max_cycles = df.groupby("unit_id")["cycle"].transform("max")
    df["cycle_norm"] = df["cycle"] / max_cycles
    return df


def build_feature_pipeline(
    df: pd.DataFrame,
    sensor_cols: list[str],
    rolling_windows: list[int] = [5, 10, 20],
    lags: list[int] = [1, 5],
    ewma_spans: list[int] = [5, 15],
) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline.
    """
    logger.info("Starting feature engineering pipeline...")

    df = add_cycle_normalized(df)
    df = add_rolling_features(df, sensor_cols, rolling_windows)
    df = add_lag_features(df, sensor_cols, lags)
    df = add_ewma_features(df, sensor_cols, ewma_spans)
    df = add_statistical_features(df, sensor_cols)

    logger.info(f"Final feature count: {df.shape[1]} columns")
    return df
