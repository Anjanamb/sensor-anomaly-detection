"""Onset-of-degradation evaluation for run-to-failure trajectories.

Framing: early-life cycles are treated as "healthy" (label 0), late-life
cycles near failure as "degraded" (label 1). The primary metric is not F1
but the lead time: how many cycles before failure did we first raise the
alarm?

Requires a ``scores_df`` with columns:
    unit, cycle, rul, score
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def label_by_rul(df: pd.DataFrame, threshold_cycles: int = 30) -> pd.Series:
    """1 if this cycle is within `threshold_cycles` of failure, else 0.

    Requires a 'rul' column (see ``data.add_rul_train``).
    """
    return (df["rul"] <= threshold_cycles).astype(int)


def first_flag_lead_time(
    scores_df: pd.DataFrame, threshold: float,
) -> pd.DataFrame:
    """Per-engine lead time between the first above-threshold flag and
    the engine's failure.

    Returns a DataFrame indexed by unit with column 'lead_time_cycles'.
    Engines that never flag are omitted.
    """
    flagged = scores_df[scores_df["score"] > threshold]
    if flagged.empty:
        empty = pd.DataFrame(columns=["unit", "lead_time_cycles"])
        return empty.set_index("unit")
    first = flagged.sort_values("cycle").drop_duplicates("unit", keep="first")
    return (
        first[["unit", "rul"]]
        .rename(columns={"rul": "lead_time_cycles"})
        .set_index("unit")
    )


def precision_recall_at_threshold(scores, labels, threshold: float) -> dict:
    """Precision / recall for a hand-picked score threshold.

    Kept dependency-free (no sklearn) so the arithmetic is auditable in the
    notebook when the reader is learning what these metrics mean.
    """
    pred = (scores > threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return {
        "precision": precision, "recall": recall,
        "tp": tp, "fp": fp, "fn": fn,
    }


def rmse(y_true, y_pred) -> float:
    """Root mean squared error, in the same units as the target (cycles)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def nasa_score(y_true, y_pred) -> float:
    """C-MAPSS PHM'08 asymmetric scoring function.

    Penalises late predictions (prognostic overshoot) more than early ones,
    reflecting the operational cost of missing a failure. Lower is better.

        d = y_pred - y_true
        score_i = exp(-d/13) - 1     if d <  0   (early)
        score_i = exp( d/10) - 1     if d >= 0   (late)
        total   = sum_i score_i
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    d = y_pred - y_true
    per_engine = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(per_engine))
