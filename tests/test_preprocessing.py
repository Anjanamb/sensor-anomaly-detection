"""Tests for preprocessing module."""

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, ".")

from src.preprocessing import (
    remove_constant_sensors,
    clip_rul,
    train_test_split_by_unit,
)


def make_sample_df():
    """Create a minimal test dataframe."""
    np.random.seed(42)
    records = []
    for unit in [1, 2, 3]:
        for cycle in range(1, 51):
            records.append({
                "unit_id": unit,
                "cycle": cycle,
                "sensor_1": np.random.normal(100, 5),
                "sensor_2": 42.0,  # constant
                "sensor_3": np.random.normal(50, 2),
                "rul": 50 - cycle,
            })
    return pd.DataFrame(records)


def test_remove_constant_sensors():
    df = make_sample_df()
    sensor_cols = ["sensor_1", "sensor_2", "sensor_3"]
    filtered_df, kept = remove_constant_sensors(df, sensor_cols)

    assert "sensor_2" not in filtered_df.columns
    assert "sensor_1" in kept
    assert "sensor_3" in kept
    assert "sensor_2" not in kept


def test_clip_rul():
    df = make_sample_df()
    clipped = clip_rul(df, max_rul=30)
    assert clipped["rul"].max() == 30
    assert clipped["rul"].min() == df["rul"].min()


def test_train_test_split_by_unit():
    df = make_sample_df()
    train, test = train_test_split_by_unit(df, test_ratio=0.34, seed=42)

    # No unit overlap
    train_units = set(train["unit_id"].unique())
    test_units = set(test["unit_id"].unique())
    assert train_units.isdisjoint(test_units)

    # All units accounted for
    assert train_units | test_units == {1, 2, 3}
