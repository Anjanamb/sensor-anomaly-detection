"""
Evaluation metrics for anomaly detection models.
Focuses on metrics suitable for imbalanced data.
"""

import logging
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""
    model_name: str
    precision: float
    recall: float
    f1: float
    auc_pr: float
    auc_roc: float
    confusion_matrix: np.ndarray
    optimal_threshold: float


def find_optimal_threshold(
    y_true: np.ndarray, scores: np.ndarray
) -> float:
    """
    Find the threshold that maximizes F1 score on the PR curve.
    """
    precisions, recalls, thresholds = precision_recall_curve(
        y_true, scores
    )
    # Avoid division by zero
    f1_scores = np.where(
        (precisions + recalls) > 0,
        2 * (precisions * recalls) / (precisions + recalls),
        0,
    )
    # thresholds has one fewer element than precisions/recalls
    best_idx = np.argmax(f1_scores[:-1])
    return thresholds[best_idx]


def evaluate_model(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> EvaluationResult:
    """
    Comprehensive evaluation of an anomaly detection model.
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc_pr = average_precision_score(y_true, scores)
    auc_roc = roc_auc_score(y_true, scores)
    cm = confusion_matrix(y_true, y_pred)
    opt_threshold = find_optimal_threshold(y_true, scores)

    result = EvaluationResult(
        model_name=model_name,
        precision=precision,
        recall=recall,
        f1=f1,
        auc_pr=auc_pr,
        auc_roc=auc_roc,
        confusion_matrix=cm,
        optimal_threshold=opt_threshold,
    )

    logger.info(
        f"{model_name}: P={precision:.3f} R={recall:.3f} "
        f"F1={f1:.3f} AUC-PR={auc_pr:.3f} AUC-ROC={auc_roc:.3f}"
    )

    return result


def compare_models(results: list[EvaluationResult]) -> None:
    """Print a comparison table of model results."""
    header = f"{'Model':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC-PR':>10} {'AUC-ROC':>10}"
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda x: x.f1, reverse=True):
        print(
            f"{r.model_name:<25} {r.precision:>10.3f} {r.recall:>10.3f} "
            f"{r.f1:>10.3f} {r.auc_pr:>10.3f} {r.auc_roc:>10.3f}"
        )
