"""Smoke tests for the per-regime normalisation helpers."""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from src.multi_regime import (
    OP_SETTING_COLS,
    fit_regime_normalisation,
    apply_regime_normalisation,
)


def _toy_df(n_per_regime=80, n_regimes=3, seed=0):
    """Synthetic data: 3 regimes with distinct op-setting centres + 5 sensors."""
    rng = np.random.default_rng(seed)
    rows = []
    centres = [(0, 0, 0), (10, 0.5, 100), (20, 1.0, 60)]
    for r, c in enumerate(centres[:n_regimes]):
        for i in range(n_per_regime):
            row = {
                "unit_id": (i % 10) + 1,
                "cycle": i + 1,
                "op_setting_1": c[0] + rng.normal(scale=0.05),
                "op_setting_2": c[1] + rng.normal(scale=0.005),
                "op_setting_3": c[2] + rng.normal(scale=0.05),
                "anomaly": int(i % 7 == 0),  # ~14% anomalous
            }
            for s in range(1, 6):
                # Sensor offset depends on regime; healthy has lower variance
                row[f"sensor_{s}"] = (
                    100 * r + rng.normal(scale=2 if row["anomaly"] == 0 else 6)
                )
            rows.append(row)
    return pd.DataFrame(rows).astype(
        {f"sensor_{s}": "float64" for s in range(1, 6)}
    )


def test_fit_regime_normalisation_clusters_correctly():
    df = _toy_df(n_regimes=3)
    sensor_cols = [f"sensor_{s}" for s in range(1, 6)]

    normed, kmeans, scalers = fit_regime_normalisation(
        df, sensor_cols, n_regimes=3
    )

    assert "regime" in normed.columns
    assert len(scalers) == 3
    # Each regime's healthy rows should now be ~standardised
    for r in range(3):
        mask = (normed["regime"] == r) & (normed["anomaly"] == 0)
        sub = normed.loc[mask, sensor_cols].values
        assert abs(sub.mean()) < 0.5
        assert abs(sub.std() - 1.0) < 0.5


def test_apply_regime_normalisation_uses_saved_artefacts():
    df = _toy_df(n_regimes=3, seed=0)
    sensor_cols = [f"sensor_{s}" for s in range(1, 6)]
    fitted, kmeans, scalers = fit_regime_normalisation(
        df, sensor_cols, n_regimes=3
    )

    # Pretend we got fresh data with the same distribution
    fresh = _toy_df(n_regimes=3, seed=1)
    applied = apply_regime_normalisation(fresh, sensor_cols, kmeans, scalers)

    assert "regime" in applied.columns
    # Should produce a regime label for every row
    assert applied["regime"].nunique() == 3
    # Normalised sensors should be on a similar scale to the fitted ones
    for r in range(3):
        mask = (applied["regime"] == r) & (applied["anomaly"] == 0)
        sub = applied.loc[mask, sensor_cols].values
        assert abs(sub.mean()) < 1.0
        assert sub.std() < 5.0


def test_op_setting_cols_constant_exists():
    assert OP_SETTING_COLS == ("op_setting_1", "op_setting_2", "op_setting_3")
