"""
Generate the five CSVs the Power BI dashboard consumes.

Outputs land under ``bi/data/``. Reuses the dashboard's per-subset loaders
(``app.streamlit_app.load_and_process_data`` etc.) and the global-SHAP
helper, so the BI tables are guaranteed to match what the live Streamlit
dashboard shows.

Run with::

    venv/Scripts/python.exe bi/build_data.py
    venv/Scripts/python.exe bi/build_data.py --only FD001  # subset shortcut
    venv/Scripts/python.exe bi/build_data.py --skip-shap   # fast iteration

CSV contracts (Power BI relies on these column names — don't rename without
updating ``bi/BUILD_GUIDE.md``):

- ``cycles.csv``       — fact table, per-cycle telemetry + IF score
- ``engines.csv``      — engine dimension
- ``model_comparison.csv`` — per (subset, model) metrics
- ``feature_importance.csv`` — per (subset, sensor, rul_bucket) SHAP totals
- ``sensors.csv``      — sensor dimension (NASA descriptions)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Silence Streamlit's "No runtime found" cache warnings when called as a script
warnings.filterwarnings("ignore", module="streamlit")
logging.getLogger("streamlit").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import (  # noqa: E402
    load_cmapss,
    add_rul_to_train,
    create_anomaly_labels,
    get_sensor_columns,
)
from src.preprocessing import (  # noqa: E402
    train_test_split_by_unit,
    create_sequences,
)
from src.feature_engineering import build_feature_pipeline  # noqa: E402
from src.multi_regime import apply_regime_normalisation  # noqa: E402
from src.models import (  # noqa: E402
    IsolationForestDetector,
    AutoencoderDetector,
    OneClassSVMDetector,
    LSTMAutoencoderDetector,
    TransformerAutoencoderDetector,
)
from src.evaluation import evaluate_model  # noqa: E402
from src.sensor_descriptions import SENSOR_TABLE  # noqa: E402
from src.explainability import build_explainer, explain  # noqa: E402

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
SUBSET_REGIMES = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}
SEQ_LEN = 30
MODELS_ROOT = ROOT / "models"
OUT_DIR = Path(__file__).resolve().parent / "data"


def _prepare_subset(subset: str):
    """Replay the dashboard's preprocessing path and return everything the
    BI exports need: data frame, scaled view, kept sensors, feature cols,
    and split into train / test by unit (seed=42)."""
    train_df, _, _ = load_cmapss(subset)
    train_df = add_rul_to_train(train_df)
    train_df = create_anomaly_labels(train_df, threshold=30)
    sensor_cols = get_sensor_columns(train_df)
    train_df = train_df.astype({c: "float64" for c in sensor_cols})

    subset_dir = MODELS_ROOT / subset
    with open(subset_dir / "kept_sensors.json", "r", encoding="utf-8") as f:
        kept_sensors = json.load(f)

    if SUBSET_REGIMES[subset] > 1:
        kmeans = joblib.load(subset_dir / "kmeans.pkl")
        regime_scalers = joblib.load(subset_dir / "regime_scalers.pkl")
        train_df = apply_regime_normalisation(
            train_df, sensor_cols, kmeans, regime_scalers
        )
    else:
        train_df["regime"] = 0

    # Drop the dropped sensors so feature engineering matches training
    dropped = set(sensor_cols) - set(kept_sensors)
    if dropped:
        train_df = train_df.drop(columns=list(dropped))

    featured = build_feature_pipeline(
        train_df,
        kept_sensors,
        rolling_windows=[5, 10],
        lags=[1, 5],
        ewma_spans=[5],
    )
    exclude = {"unit_id", "cycle", "rul", "anomaly", "regime"}
    feature_cols = [c for c in featured.columns if c not in exclude]

    scaler = joblib.load(subset_dir / "scaler.pkl")
    scaled = featured.copy()
    scaled[feature_cols] = scaler.transform(featured[feature_cols])

    train_split, test_split = train_test_split_by_unit(
        scaled, test_ratio=0.2, seed=42
    )

    return {
        "subset": subset,
        "featured": featured,                # un-scaled, has raw sensors + cycle_norm
        "scaled": scaled,                    # scaled, used for model scoring
        "kept_sensors": kept_sensors,
        "feature_cols": feature_cols,
        "train_split": train_split,
        "test_split": test_split,
    }


def _load_detectors(subset: str, n_sensors: int) -> dict:
    sd = MODELS_ROOT / subset
    models = {}
    iso = IsolationForestDetector()
    iso.load(str(sd / "isolation_forest.pkl"))
    models["Isolation Forest"] = iso

    svm = OneClassSVMDetector()
    svm.load(str(sd / "one_class_svm.pkl"))
    models["One-Class SVM"] = svm

    ae = AutoencoderDetector(input_dim=n_sensors)
    ae.load(str(sd / "autoencoder.pt"))
    models["Autoencoder"] = ae

    lstm = LSTMAutoencoderDetector(n_sensors=n_sensors, seq_len=SEQ_LEN)
    lstm.load(str(sd / "lstm_autoencoder.pt"))
    models["LSTM Autoencoder"] = lstm

    tfmr = TransformerAutoencoderDetector(n_sensors=n_sensors, seq_len=SEQ_LEN)
    tfmr.load(str(sd / "transformer_autoencoder.pt"))
    models["Transformer Autoencoder"] = tfmr
    return models


# ── CSV builders ────────────────────────────────────────────────────────


def build_cycles_csv(preps: dict[str, dict]) -> pd.DataFrame:
    """Long-format per-cycle telemetry with the IF anomaly score attached."""
    out_rows = []
    for subset, prep in preps.items():
        featured = prep["featured"]                  # un-scaled
        scaled = prep["scaled"]
        kept = prep["kept_sensors"]
        feature_cols = prep["feature_cols"]

        # Score every cycle with the IF (the headline detector for the dashboard)
        iso = IsolationForestDetector()
        iso.load(str(MODELS_ROOT / subset / "isolation_forest.pkl"))
        X = np.nan_to_num(scaled[feature_cols].values, nan=0.0)
        scores = iso.score_samples(X)

        df = featured[["unit_id", "cycle", "rul", "anomaly", "regime"]].copy()
        df["subset"] = subset
        df["anomaly_score_iso"] = scores
        # Carry the kept raw sensors so Power BI can build sparklines
        for s in kept:
            df[s] = featured[s].values
        out_rows.append(df)
    cycles = pd.concat(out_rows, ignore_index=True)
    # Put metadata columns first
    leading = ["subset", "unit_id", "cycle", "rul", "anomaly",
               "anomaly_score_iso", "regime"]
    other = [c for c in cycles.columns if c not in leading]
    return cycles[leading + sorted(other)]


def build_engines_csv(preps: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for subset, prep in preps.items():
        featured = prep["featured"]
        gb = featured.groupby("unit_id")
        per_engine = pd.DataFrame({
            "subset": subset,
            "unit_id": gb.size().index,
            "max_cycle": gb["cycle"].max().values,
            "anomaly_cycle_count": gb["anomaly"].sum().values,
        })
        # Severity bucket on anomaly density, useful for slicers
        per_engine["fault_severity_bucket"] = pd.cut(
            per_engine["anomaly_cycle_count"],
            bins=[-1, 20, 35, 60, 9999],
            labels=["mild (<=20)", "moderate (21-35)", "high (36-60)", "very high (>60)"],
        ).astype(str)
        rows.append(per_engine)
    return pd.concat(rows, ignore_index=True)


def build_model_comparison_csv(preps: dict[str, dict]) -> pd.DataFrame:
    """Re-evaluate every detector on its test split — guarantees the numbers
    in the BI match the live dashboard exactly."""
    rows = []
    for subset, prep in preps.items():
        test = prep["test_split"]
        kept = prep["kept_sensors"]
        feature_cols = prep["feature_cols"]
        models = _load_detectors(subset, n_sensors=len(kept))

        X_test = np.nan_to_num(test[feature_cols].values, nan=0.0)
        X_test_raw = np.nan_to_num(test[kept].values, nan=0.0)
        y_test = test["anomaly"].values

        # Sequence models need windows
        X_test_seq, y_test_seq = create_sequences(
            test, kept, sequence_length=SEQ_LEN
        )

        for name, model in models.items():
            if name == "Autoencoder":
                X_for_model, y_for_model = X_test_raw, y_test
            elif name in ("LSTM Autoencoder", "Transformer Autoencoder"):
                X_for_model, y_for_model = X_test_seq, y_test_seq
            else:
                X_for_model, y_for_model = X_test, y_test
            scores = model.score_samples(X_for_model)
            preds = model.predict(X_for_model)
            r = evaluate_model(name, y_for_model, preds, scores)
            rows.append({
                "subset": subset,
                "model": name,
                "F1": round(r.f1, 3),
                "AUC_ROC": round(r.auc_roc, 3),
                "AUC_PR": round(r.auc_pr, 3),
                "Precision": round(r.precision, 3),
                "Recall": round(r.recall, 3),
            })
    return pd.DataFrame(rows)


def _bucket_rul(r: float) -> str:
    if r > 80:
        return "mid-life (RUL > 80)"
    if r > 30:
        return "pre-warning (30 < RUL <= 80)"
    return "warning zone (RUL <= 30)"


def build_feature_importance_csv(preps: dict[str, dict]) -> pd.DataFrame:
    """Per (subset, sensor, rul_bucket) SHAP totals + subsystem metadata.

    Computes SHAP from scratch here (no Streamlit caching dependency).
    """
    rows = []
    for subset, prep in preps.items():
        feature_cols = prep["feature_cols"]
        train_split = prep["train_split"]
        test_split = prep["test_split"]

        iso = IsolationForestDetector()
        iso.load(str(MODELS_ROOT / subset / "isolation_forest.pkl"))

        healthy_train = train_split[train_split["anomaly"] == 0]
        bg = healthy_train[feature_cols].sample(
            min(200, len(healthy_train)), random_state=42
        ).values
        background = np.nan_to_num(bg, nan=0.0)

        X_test = np.nan_to_num(test_split[feature_cols].values, nan=0.0)
        rul = test_split["rul"].values

        explainer = build_explainer(iso, background, max_background=200)
        explanation = explain(iso, X_test, feature_cols, explainer=explainer)

        # Map each feature → base sensor
        per_feature = pd.DataFrame({
            "feature": feature_cols,
            "shap_matrix_col": np.arange(len(feature_cols)),
        })
        per_feature["base_sensor"] = per_feature["feature"].apply(
            lambda f: (f.split("_roll_")[0].split("_lag_")[0]
                       .split("_diff_")[0].split("_ewma_")[0]
                       .split("_skew_")[0].split("_kurt_")[0])
            if "sensor_" in f else f
        )
        # Overall + per RUL bucket
        bucket_labels = pd.Series(rul).apply(_bucket_rul)
        scenarios = [("overall", np.arange(len(rul)))] + [
            (label, np.array(idx_list))
            for label, idx_list in bucket_labels.groupby(bucket_labels).groups.items()
        ]

        for label, idx in scenarios:
            mean_abs = np.abs(explanation.shap_values[idx]).mean(axis=0)
            per_feature["abs_shap"] = mean_abs
            agg = (
                per_feature.groupby("base_sensor")["abs_shap"].sum()
                .reset_index()
                .sort_values("abs_shap", ascending=False)
            )
            agg = agg[agg["base_sensor"].str.startswith("sensor_")]
            agg = agg.merge(
                SENSOR_TABLE.reset_index().rename(columns={"sensor": "base_sensor"}),
                on="base_sensor", how="left",
            )
            agg["subset"] = subset
            agg["rul_bucket"] = label
            agg["rank"] = range(1, len(agg) + 1)
            rows.append(agg[
                ["subset", "rul_bucket", "rank", "base_sensor",
                 "symbol", "quantity", "units", "subsystem", "abs_shap"]
            ])
    return pd.concat(rows, ignore_index=True).rename(
        columns={"abs_shap": "total_abs_shap", "base_sensor": "sensor_id"}
    )


def build_sensors_csv() -> pd.DataFrame:
    df = SENSOR_TABLE.reset_index().rename(columns={"sensor": "sensor_id"})
    return df


# ── Main ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", nargs="+", choices=SUBSETS, default=None,
        help="Only export these subsets (default: all four)",
    )
    parser.add_argument(
        "--skip-shap", action="store_true",
        help="Skip the per-subset SHAP computation (fast iteration)",
    )
    args = parser.parse_args()

    targets = args.only or SUBSETS
    logging.basicConfig(level=logging.WARNING)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Preparing subsets: {targets}")
    preps = {s: _prepare_subset(s) for s in targets}
    print(f"  Loaded {len(preps)} subsets")

    print("Building cycles.csv ...")
    cycles = build_cycles_csv(preps)
    cycles.to_csv(OUT_DIR / "cycles.csv", index=False)
    print(f"  {len(cycles):,} rows")

    print("Building engines.csv ...")
    engines = build_engines_csv(preps)
    engines.to_csv(OUT_DIR / "engines.csv", index=False)
    print(f"  {len(engines):,} rows")

    print("Building model_comparison.csv ...")
    mc = build_model_comparison_csv(preps)
    mc.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    print(f"  {len(mc):,} rows")

    if not args.skip_shap:
        print("Building feature_importance.csv (running SHAP per subset) ...")
        fi = build_feature_importance_csv(preps)
        fi.to_csv(OUT_DIR / "feature_importance.csv", index=False)
        print(f"  {len(fi):,} rows")
    else:
        print("Skipping feature_importance.csv (--skip-shap)")

    print("Building sensors.csv ...")
    sensors = build_sensors_csv()
    sensors.to_csv(OUT_DIR / "sensors.csv", index=False)
    print(f"  {len(sensors):,} rows")

    print()
    print(f"All CSVs written to {OUT_DIR}")
    for p in sorted(OUT_DIR.glob("*.csv")):
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name:<30s}  {size_kb:>8.1f} KB")


if __name__ == "__main__":
    main()
