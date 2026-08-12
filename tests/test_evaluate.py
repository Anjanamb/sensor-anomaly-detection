"""Sanity checks for the onset-of-degradation evaluation helpers."""
import numpy as np
import pandas as pd

from src.evaluate import (
    label_by_rul, first_flag_lead_time, precision_recall_at_threshold,
    rmse, nasa_score,
)


def test_label_by_rul_flips_at_threshold():
    df = pd.DataFrame({"rul": [100, 31, 30, 29, 0]})
    lbl = label_by_rul(df, threshold_cycles=30)
    assert list(lbl) == [0, 0, 1, 1, 1]


def test_first_flag_lead_time_picks_earliest_flag_per_engine():
    scores = pd.DataFrame({
        "unit":  [1, 1, 1, 2, 2, 2],
        "cycle": [1, 2, 3, 1, 2, 3],
        "rul":   [50, 40, 30, 20, 10, 0],
        "score": [0.1, 0.9, 0.95, 0.2, 0.3, 0.8],
    })
    out = first_flag_lead_time(scores, threshold=0.5)
    # Engine 1 first crosses 0.5 at cycle 2 (rul=40).
    # Engine 2 first crosses 0.5 at cycle 3 (rul=0).
    assert int(out.loc[1, "lead_time_cycles"]) == 40
    assert int(out.loc[2, "lead_time_cycles"]) == 0


def test_first_flag_lead_time_omits_engines_that_never_flag():
    scores = pd.DataFrame({
        "unit":  [1, 1, 2, 2],
        "cycle": [1, 2, 1, 2],
        "rul":   [10, 0, 10, 0],
        "score": [0.1, 0.2, 0.9, 0.95],   # engine 1 never crosses
    })
    out = first_flag_lead_time(scores, threshold=0.5)
    assert 1 not in out.index
    assert 2 in out.index


def test_precision_recall_all_true_positive():
    scores = np.array([0.9, 0.8, 0.7])
    labels = np.array([1, 1, 1])
    out = precision_recall_at_threshold(scores, labels, threshold=0.5)
    assert out["precision"] == 1.0
    assert out["recall"]    == 1.0
    assert out["tp"] == 3 and out["fp"] == 0 and out["fn"] == 0


def test_rmse_zero_on_perfect_prediction():
    y = np.array([10.0, 20.0, 30.0])
    assert rmse(y, y) == 0.0


def test_rmse_matches_hand_calculation():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    # errors 2, -2, 3; squared 4, 4, 9; mean 17/3; sqrt ~2.38
    assert abs(rmse(y_true, y_pred) - np.sqrt(17.0 / 3.0)) < 1e-9


def test_nasa_score_zero_on_perfect_prediction():
    y = np.array([10.0, 20.0, 30.0])
    assert nasa_score(y, y) == 0.0


def test_nasa_score_penalises_late_more_than_early():
    """Overshoot by 10 should hurt more than undershoot by 10."""
    y_true = np.array([100.0])
    early = nasa_score(y_true, np.array([90.0]))   # d = -10
    late  = nasa_score(y_true, np.array([110.0]))  # d = +10
    assert late > early
    # Known values: early = exp(10/13)-1 ~ 1.156, late = exp(10/10)-1 ~ 1.718
    assert abs(early - (np.exp(10.0 / 13.0) - 1.0)) < 1e-9
    assert abs(late  - (np.exp(10.0 / 10.0) - 1.0)) < 1e-9
