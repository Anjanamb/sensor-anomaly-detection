"""Thin wrappers around the two anomaly detectors used in v2.

Both are unsupervised — they learn only from *healthy* engine cycles (early
in each trajectory) and score any new point by how unlike that healthy
distribution it looks.

- IsolationForest: tree-based, cheap, produces a continuous anomaly score.
- DBSCAN:         density-based, labels dense regions as clusters and sparse
                   points as noise (label -1). Noise is the natural anomaly
                   label — useful for a very different-looking second opinion.

The two detectors sit side-by-side in the notebooks so the reader can
compare a scoring approach against a clustering approach on the same features.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest


# ---- Isolation Forest -------------------------------------------------------

def fit_isolation_forest(
    X, *, contamination: float = 0.02, n_estimators: int = 200,
    random_state: int = 42,
) -> IsolationForest:
    """Train an IsolationForest on a *healthy-only* feature matrix.

    contamination — expected fraction of anomalies in training data. Since we
      train on healthy cycles, this should be tiny (0.01 – 0.05); leave a
      small non-zero so the internal offset is well-behaved.
    """
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)
    return model


def anomaly_score(model: IsolationForest, X) -> np.ndarray:
    """Higher = more anomalous.

    sklearn's ``score_samples`` returns *higher for normal*; we negate so
    the number reads intuitively as an anomaly score.
    """
    return -model.score_samples(X)


# ---- DBSCAN -----------------------------------------------------------------

def fit_predict_dbscan(X, *, eps: float, min_samples: int) -> np.ndarray:
    """Fit DBSCAN and return cluster labels. -1 == noise == anomaly.

    eps and min_samples need tuning per dataset — see the k-distance plot in
    ``notebooks/04_train_dbscan.ipynb`` for how we pick them.
    """
    return DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(X)
