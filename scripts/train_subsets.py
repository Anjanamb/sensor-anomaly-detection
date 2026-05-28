"""
Train all five detectors on each C-MAPSS subset and save per-subset
artefacts under ``models/<SUBSET>/``.

Per subset, we save:
- ``isolation_forest.pkl``, ``one_class_svm.pkl`` — sklearn detectors with
  F1-optimal thresholds in a dict
- ``autoencoder.pt``, ``lstm_autoencoder.pt``, ``transformer_autoencoder.pt``
  — torch detectors with F1-optimal thresholds in the checkpoint
- ``scaler.pkl`` — StandardScaler fit on engineered features
- ``kept_sensors.json`` — sensor names that survived constant-removal
  (varies per subset)
- ``kmeans.pkl`` and ``regime_scalers.pkl`` — multi-regime artefacts
  (FD002, FD004 only)

Run with::

    venv/Scripts/python.exe scripts/train_subsets.py [--only FD001 FD003]

Reuses the same hyperparameters as notebook 06 so the dashboard numbers
match the cross-subset comparison.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Allow `from src import ...` when run from the project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import (  # noqa: E402
    load_cmapss,
    add_rul_to_train,
    create_anomaly_labels,
    get_sensor_columns,
)
from src.preprocessing import (  # noqa: E402
    remove_constant_sensors,
    normalize_global,
    train_test_split_by_unit,
    create_sequences,
)
from src.feature_engineering import build_feature_pipeline  # noqa: E402
from src.multi_regime import fit_regime_normalisation  # noqa: E402
from src.models import (  # noqa: E402
    IsolationForestDetector,
    AutoencoderDetector,
    OneClassSVMDetector,
    LSTMAutoencoderDetector,
    TransformerAutoencoderDetector,
)
from src.evaluation import evaluate_model, find_optimal_threshold  # noqa: E402

logger = logging.getLogger(__name__)

SEQ_LEN = 30
SUBSET_REGIMES = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}


def prepare_subset(name: str, n_regimes: int):
    """Build all train / test matrices and persistence artefacts for a subset."""
    train_df, _, _ = load_cmapss(name)
    train_df = add_rul_to_train(train_df)
    train_df = create_anomaly_labels(train_df, threshold=30)

    sensor_cols = get_sensor_columns(train_df)
    train_df = train_df.astype({c: "float64" for c in sensor_cols})

    kmeans = None
    regime_scalers = None
    if n_regimes > 1:
        train_df, kmeans, regime_scalers = fit_regime_normalisation(
            train_df, sensor_cols, n_regimes=n_regimes
        )

    train_df, kept_sensors = remove_constant_sensors(train_df, sensor_cols)

    featured = build_feature_pipeline(
        train_df,
        kept_sensors,
        rolling_windows=[5, 10],
        lags=[1, 5],
        ewma_spans=[5],
    )
    exclude = ["unit_id", "cycle", "rul", "anomaly"]
    if "regime" in featured.columns:
        exclude.append("regime")
    feature_cols = [c for c in featured.columns if c not in exclude]
    raw_sensor_cols = list(kept_sensors)

    train_split, test_split = train_test_split_by_unit(
        featured, test_ratio=0.2, seed=42
    )
    train_split, scaler_all = normalize_global(
        train_split, feature_cols, method="standard"
    )
    test_split, _ = normalize_global(
        test_split, feature_cols, method="standard", scaler=scaler_all
    )

    train_healthy = train_split[train_split["anomaly"] == 0]
    X_train_healthy = np.nan_to_num(train_healthy[feature_cols].values, nan=0.0)
    X_test = np.nan_to_num(test_split[feature_cols].values, nan=0.0)
    y_test = test_split["anomaly"].values
    X_train_healthy_raw = np.nan_to_num(
        train_healthy[raw_sensor_cols].values, nan=0.0
    )
    X_test_raw = np.nan_to_num(test_split[raw_sensor_cols].values, nan=0.0)
    X_train_seq, y_train_seq = create_sequences(
        train_split, raw_sensor_cols, sequence_length=SEQ_LEN
    )
    X_test_seq, y_test_seq = create_sequences(
        test_split, raw_sensor_cols, sequence_length=SEQ_LEN
    )
    X_train_healthy_seq = X_train_seq[y_train_seq == 0]

    return {
        "name": name,
        "n_regimes": n_regimes,
        "feature_cols": feature_cols,
        "raw_sensor_cols": raw_sensor_cols,
        "kept_sensors": list(kept_sensors),
        "scaler_all": scaler_all,
        "kmeans": kmeans,
        "regime_scalers": regime_scalers,
        "X_train_healthy": X_train_healthy,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_healthy_raw": X_train_healthy_raw,
        "X_test_raw": X_test_raw,
        "X_train_healthy_seq": X_train_healthy_seq,
        "X_test_seq": X_test_seq,
        "y_test_seq": y_test_seq,
    }


def train_and_save(prep: dict, out_dir: Path) -> list[dict]:
    """Train all 5 detectors, persist them with their F1-optimal thresholds,
    and return a list of result rows for logging."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    n_sensors = len(prep["raw_sensor_cols"])

    # Isolation Forest
    iso = IsolationForestDetector(
        contamination=0.05, n_estimators=300, random_state=42
    )
    iso.fit(prep["X_train_healthy"])
    s = iso.score_samples(prep["X_test"])
    thr = find_optimal_threshold(prep["y_test"], s)
    iso.threshold = thr
    r = evaluate_model("Isolation Forest", prep["y_test"], (s > thr).astype(int), s)
    iso.save(str(out_dir / "isolation_forest.pkl"))
    results.append({"model": "Isolation Forest", **_metric_row(r)})

    # One-Class SVM
    svm = OneClassSVMDetector(kernel="rbf", gamma="scale", nu=0.05)
    svm.fit(prep["X_train_healthy"])
    s = svm.score_samples(prep["X_test"])
    thr = find_optimal_threshold(prep["y_test"], s)
    svm.threshold = thr
    r = evaluate_model("One-Class SVM", prep["y_test"], (s > thr).astype(int), s)
    svm.save(str(out_dir / "one_class_svm.pkl"))
    results.append({"model": "One-Class SVM", **_metric_row(r)})

    # Feedforward AE
    ae = AutoencoderDetector(
        input_dim=n_sensors,
        encoding_dim=8,
        epochs=150,
        batch_size=256,
        threshold_percentile=95.0,
    )
    ae.fit(prep["X_train_healthy_raw"])
    s = ae.score_samples(prep["X_test_raw"])
    thr = find_optimal_threshold(prep["y_test"], s)
    ae.threshold = thr
    r = evaluate_model(
        "Autoencoder", prep["y_test"], (s > thr).astype(int), s
    )
    ae.save(str(out_dir / "autoencoder.pt"))
    results.append({"model": "Autoencoder", **_metric_row(r)})

    # LSTM AE
    lstm = LSTMAutoencoderDetector(
        n_sensors=n_sensors,
        seq_len=SEQ_LEN,
        hidden_dim=32,
        encoding_dim=8,
        num_layers=2,
        epochs=80,
        batch_size=256,
    )
    lstm.fit(prep["X_train_healthy_seq"])
    s = lstm.score_samples(prep["X_test_seq"])
    thr = find_optimal_threshold(prep["y_test_seq"], s)
    lstm.threshold = thr
    r = evaluate_model(
        "LSTM Autoencoder", prep["y_test_seq"], (s > thr).astype(int), s
    )
    lstm.save(str(out_dir / "lstm_autoencoder.pt"))
    results.append({"model": "LSTM Autoencoder", **_metric_row(r)})

    # Transformer AE
    tfmr = TransformerAutoencoderDetector(
        n_sensors=n_sensors,
        seq_len=SEQ_LEN,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        bottleneck_dim=8,
        epochs=120,
        batch_size=256,
    )
    tfmr.fit(prep["X_train_healthy_seq"])
    s = tfmr.score_samples(prep["X_test_seq"])
    thr = find_optimal_threshold(prep["y_test_seq"], s)
    tfmr.threshold = thr
    r = evaluate_model(
        "Transformer Autoencoder", prep["y_test_seq"], (s > thr).astype(int), s
    )
    tfmr.save(str(out_dir / "transformer_autoencoder.pt"))
    results.append({"model": "Transformer Autoencoder", **_metric_row(r)})

    # Persist preprocessing artefacts
    joblib.dump(prep["scaler_all"], out_dir / "scaler.pkl")
    with open(out_dir / "kept_sensors.json", "w", encoding="utf-8") as f:
        json.dump(prep["kept_sensors"], f)
    if prep["kmeans"] is not None:
        joblib.dump(prep["kmeans"], out_dir / "kmeans.pkl")
        joblib.dump(prep["regime_scalers"], out_dir / "regime_scalers.pkl")

    return results


def _metric_row(r):
    return {
        "F1": round(r.f1, 3),
        "AUC-ROC": round(r.auc_roc, 3),
        "AUC-PR": round(r.auc_pr, 3),
        "Precision": round(r.precision, 3),
        "Recall": round(r.recall, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(SUBSET_REGIMES.keys()),
        default=None,
        help="Train only these subsets (default: all four)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    targets = args.only or list(SUBSET_REGIMES.keys())
    all_rows = []
    for name in targets:
        n_regimes = SUBSET_REGIMES[name]
        t0 = time.time()
        print(f"\n=== {name} (n_regimes={n_regimes}) ===")
        prep = prepare_subset(name, n_regimes)
        print(
            f"  features={len(prep['feature_cols'])}, "
            f"sensors_kept={len(prep['kept_sensors'])}, "
            f"healthy_train={prep['X_train_healthy'].shape[0]}, "
            f"test={prep['X_test'].shape[0]}"
        )
        out = ROOT / "models" / name
        rows = train_and_save(prep, out)
        for r in rows:
            r["subset"] = name
            all_rows.append(r)
        print(f"  done in {time.time() - t0:.0f}s; saved to {out}")
        for r in rows:
            print(
                f"    {r['model']:>30s}  F1={r['F1']:.3f}  "
                f"AUC-PR={r['AUC-PR']:.3f}  AUC-ROC={r['AUC-ROC']:.3f}"
            )

    if len(all_rows) >= 2:
        print("\n=== Summary ===")
        df = pd.DataFrame(all_rows)
        pivot = df.pivot(index="model", columns="subset", values="F1")
        print(pivot.to_string())


if __name__ == "__main__":
    main()
