"""Sanity checks for the FD004 loader."""
import pandas as pd
import pytest

from src.data import (
    ALL_COLS, SENSOR_COLS,
    load_fd004, load_rul_fd004, add_rul_train,
)


def test_load_fd004_train_has_expected_columns():
    df = load_fd004("train")
    assert list(df.columns) == ALL_COLS
    assert len(SENSOR_COLS) == 21
    assert df["unit"].min() == 1


def test_load_fd004_bad_kind_raises():
    with pytest.raises(ValueError):
        load_fd004("validation")


def test_load_rul_fd004_shape_matches_test_units():
    rul = load_rul_fd004()
    test_df = load_fd004("test")
    # One RUL value per unique test engine.
    assert len(rul) == test_df["unit"].nunique()


def test_add_rul_train_is_zero_at_failure():
    df = load_fd004("train")
    with_rul = add_rul_train(df)
    # For each engine, the last recorded cycle is failure → RUL should be 0.
    last_cycles = with_rul.sort_values("cycle").groupby("unit").tail(1)
    assert (last_cycles["rul"] == 0).all()
    # And RUL should never be negative.
    assert (with_rul["rul"] >= 0).all()


def test_add_rul_train_does_not_mutate_input():
    df = load_fd004("train")
    cols_before = set(df.columns)
    _ = add_rul_train(df)
    assert set(df.columns) == cols_before  # original untouched
