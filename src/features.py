"""Small primitives for time-series feature engineering on FD004.

Each function takes a pandas Series and returns a new Series of the same
length. Per-engine application is done by the caller using
``df.groupby("unit")["s2"].transform(lambda s: rolling_mean(s, 5))``. This
keeps the primitives free of dataset-specific knowledge and easy to test.

The choice of which primitive to apply to which sensor, with what window,
lives in ``notebooks/02_feature_engineering.ipynb`` alongside the hypothesis
and validation plot that motivates it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_mean(s: pd.Series, window: int) -> pd.Series:
    """Rolling window mean. min_periods=1 so early cycles aren't NaN."""
    return s.rolling(window, min_periods=1).mean()


def rolling_std(s: pd.Series, window: int) -> pd.Series:
    """Rolling window standard deviation. NaN at the first cycle becomes 0."""
    return s.rolling(window, min_periods=1).std().fillna(0.0)


def ewma(s: pd.Series, span: int) -> pd.Series:
    """Exponentially-weighted moving average with the given span.

    Compared to rolling_mean, EWMA reacts faster to recent changes while
    still smoothing out single-cycle noise.
    """
    return s.ewm(span=span, adjust=False).mean()


def rolling_slope(s: pd.Series, window: int) -> pd.Series:
    """Rolling least-squares slope. Positive = value is trending up.

    A drifting sensor produces a persistent nonzero slope well before its
    absolute value looks abnormal, which is useful for onset detection.
    """
    def _slope(y: np.ndarray) -> float:
        if len(y) < 2:
            return 0.0
        x = np.arange(len(y), dtype=float)
        return float(np.polyfit(x, y, 1)[0])
    return s.rolling(window, min_periods=1).apply(_slope, raw=True)


def deviation_from_baseline(
    s: pd.Series, baseline_cycles: int = 30,
) -> pd.Series:
    """(current value) - (mean of the first N cycles of this series).

    Baseline = "what this sensor looked like while the engine was healthy",
    so the feature reads directly as "how far has the reading drifted?".
    Pass a per-engine series (via groupby.transform) so each engine gets its
    own baseline.
    """
    baseline = s.iloc[:baseline_cycles].mean()
    return s - baseline
