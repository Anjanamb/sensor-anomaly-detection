"""Sanity checks for the feature primitives."""
import numpy as np
import pandas as pd

from src.features import (
    rolling_mean, rolling_std, ewma, rolling_slope, deviation_from_baseline,
)


def _series(values):
    return pd.Series(values, dtype=float)


def test_rolling_mean_preserves_length_and_starts_with_input():
    s = _series([1, 2, 3, 4, 5])
    out = rolling_mean(s, window=3)
    assert len(out) == len(s)
    # min_periods=1, so the first value is the first input value itself.
    assert out.iloc[0] == 1.0
    # Window of 3 at index 2: mean of [1, 2, 3] = 2.
    assert out.iloc[2] == 2.0


def test_rolling_std_no_nan_at_start():
    s = _series([1, 1, 1, 5])
    out = rolling_std(s, window=3)
    # First value would be NaN (std of one element); should be filled with 0.
    assert out.iloc[0] == 0.0
    assert not out.isna().any()


def test_ewma_reacts_faster_than_rolling_mean():
    # A step change: 20 zeros then 20 ones. EWMA(span=5) should overtake
    # rolling_mean(window=5) at least once during the transition.
    s = _series([0]*20 + [1]*20)
    r = rolling_mean(s, window=5)
    e = ewma(s, span=5)
    diff_at_step = e.iloc[21] - r.iloc[21]
    assert diff_at_step > 0, "EWMA should climb faster right after the step"


def test_rolling_slope_of_ascending_line_is_positive():
    s = _series(np.arange(20))
    slope = rolling_slope(s, window=5)
    # Ignore the very first cycle where window is size 1; slope is 0 there.
    assert (slope.iloc[4:] > 0).all()


def test_deviation_from_baseline_is_zero_when_flat():
    s = _series([2.0] * 50)
    out = deviation_from_baseline(s, baseline_cycles=30)
    assert (out == 0.0).all()


def test_deviation_from_baseline_captures_drift():
    healthy = [1.0] * 30
    drift = list(np.linspace(1.0, 5.0, 20))
    s = _series(healthy + drift)
    out = deviation_from_baseline(s, baseline_cycles=30)
    # Healthy portion sits at ~0; drift portion grows.
    assert abs(out.iloc[15]) < 1e-9
    assert out.iloc[-1] > 3.0
