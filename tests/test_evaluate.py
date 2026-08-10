"""Sanity checks for the onset-of-degradation evaluation helpers."""
import numpy as np
import pandas as pd

from src.evaluate import (
    label_by_rul, first_flag_lead_time, precision_recall_at_threshold,
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
