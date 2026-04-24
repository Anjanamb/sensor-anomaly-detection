"""Tests for feature engineering."""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")

from src.feature_engineering import add_rolling_features, add_lag_features, add_cycle_normalized


def make_sample_df():
    records = []
    for unit in [1, 2]:
        for cycle in range(1, 31):
            records.append({
                "unit_id": unit,
                "cycle": cycle,
                "sensor_1": np.sin(cycle * 0.1) + unit,
                "sensor_2": cycle * 0.5 + unit,
            })
    return pd.DataFrame(records)


def test_add_rolling_features():
    df = make_sample_df()
    result = add_rolling_features(df, ["sensor_1", "sensor_2"], windows=[5])
    assert "sensor_1_roll_mean_5" in result.columns
    assert "sensor_2_roll_std_5" in result.columns
    assert not result["sensor_1_roll_mean_5"].isna().any()


def test_add_lag_features():
    df = make_sample_df()
    result = add_lag_features(df, ["sensor_1"], lags=[1, 3])
    assert "sensor_1_lag_1" in result.columns
    assert "sensor_1_diff_3" in result.columns


def test_cycle_normalized():
    df = make_sample_df()
    result = add_cycle_normalized(df)
    assert "cycle_norm" in result.columns
    assert result["cycle_norm"].max() <= 1.0
    assert result["cycle_norm"].min() >= 0.0