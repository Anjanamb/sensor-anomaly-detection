"""
Industrial Sensor Anomaly Detection Dashboard
Main Streamlit application.
"""

import os
import sys
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
from pathlib import Path

from src.data_loader import load_cmapss, add_rul_to_train, create_anomaly_labels, get_sensor_columns
from src.preprocessing import train_test_split_by_unit, create_sequences
from src.feature_engineering import build_feature_pipeline
from src.models import (
    IsolationForestDetector, AutoencoderDetector, OneClassSVMDetector,
)
from src.evaluation import evaluate_model
from src.multi_regime import apply_regime_normalisation

# All four C-MAPSS subsets. Multi-regime ones need per-regime sensor
# normalisation (saved KMeans + regime scalers); single-regime ones skip it.
SUBSETS = ['FD001', 'FD002', 'FD003', 'FD004']
SUBSET_REGIMES = {'FD001': 1, 'FD002': 6, 'FD003': 1, 'FD004': 6}
MODELS_ROOT = Path(__file__).resolve().parent.parent / 'models'

# The LSTM detector pulls in extra torch surface (LSTM cells, weights_only
# kwarg in newer torch.load) that has occasionally tripped Streamlit Cloud
# wheels. Import it defensively so a missing-deps failure on the LSTM never
# takes down the whole dashboard — the other three detectors stay usable.
try:
    from src.models import LSTMAutoencoderDetector, TransformerAutoencoderDetector
    _SEQ_IMPORT_ERROR: str | None = None
except Exception as _e:  # pragma: no cover - environment-dependent
    LSTMAutoencoderDetector = None  # type: ignore[assignment,misc]
    TransformerAutoencoderDetector = None  # type: ignore[assignment,misc]
    _SEQ_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"

# The explainability layer imports `shap`, which drags in numba + llvmlite —
# wheels that frequently fail to resolve on Streamlit Cloud's Python. Import
# defensively so the SHAP panel becomes optional rather than fatal. None of
# these symbols are needed outside the SHAP panel.
try:
    from src.explainability import (
        build_explainer, explain, top_features_for_sample,
        pretty_feature_label, feature_glossary, narrate,
    )
    _SHAP_IMPORT_ERROR: str | None = None
except Exception as _e:  # pragma: no cover - environment-dependent
    build_explainer = explain = top_features_for_sample = None  # type: ignore
    pretty_feature_label = feature_glossary = narrate = None  # type: ignore
    _SHAP_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"

LSTM_SEQ_LEN = 30

# ── Page config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sensor Anomaly Detection",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ──────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading + featurising subset…")
def load_and_process_data(subset: str):
    """Load a C-MAPSS subset and run the same preprocessing as training.

    Returns ``(featured_df, kept_sensors)``. For multi-regime subsets,
    raw sensor values are first standardised per regime using the saved
    KMeans + regime scalers (so feature engineering downstream sees the
    same inputs the model was trained on).
    """
    train_df, _test_df, _rul_df = load_cmapss(subset)
    train_df = add_rul_to_train(train_df)
    train_df = create_anomaly_labels(train_df, threshold=30)

    # Use the *saved* kept-sensor list from training so feature engineering
    # operates on exactly the same columns the model was trained on.
    subset_dir = MODELS_ROOT / subset
    with open(subset_dir / 'kept_sensors.json', 'r', encoding='utf-8') as f:
        kept_sensors = json.load(f)

    sensor_cols = get_sensor_columns(train_df)
    train_df = train_df.astype({c: 'float64' for c in sensor_cols})

    if SUBSET_REGIMES[subset] > 1:
        kmeans = joblib.load(subset_dir / 'kmeans.pkl')
        regime_scalers = joblib.load(subset_dir / 'regime_scalers.pkl')
        train_df = apply_regime_normalisation(
            train_df, sensor_cols, kmeans, regime_scalers
        )

    # Match training: drop the sensors that were constant (and hence dropped)
    # at training time. Otherwise the saved StandardScaler complains about
    # unseen feature names downstream.
    dropped = set(sensor_cols) - set(kept_sensors)
    if dropped:
        train_df = train_df.drop(columns=list(dropped))

    featured_df = build_feature_pipeline(
        train_df, kept_sensors,
        rolling_windows=[5, 10], lags=[1, 5], ewma_spans=[5]
    )
    return featured_df, kept_sensors


@st.cache_resource(show_spinner="Loading models for subset…")
def load_models(subset: str, raw_sensor_dim: int):
    """Load the saved detectors for ``subset`` from ``models/<subset>/``.

    Sequence models and torch-based detectors are loaded best-effort: if a
    Cloud-side dependency issue stops one from loading, the others stay
    usable rather than crashing the whole app.
    """
    subset_dir = MODELS_ROOT / subset
    models = {}

    iso = IsolationForestDetector()
    iso.load(str(subset_dir / 'isolation_forest.pkl'))
    models['Isolation Forest'] = iso

    ae = AutoencoderDetector(input_dim=raw_sensor_dim)
    ae.load(str(subset_dir / 'autoencoder.pt'))
    models['Autoencoder'] = ae

    svm = OneClassSVMDetector()
    svm.load(str(subset_dir / 'one_class_svm.pkl'))
    models['One-Class SVM'] = svm

    if LSTMAutoencoderDetector is not None:
        try:
            lstm = LSTMAutoencoderDetector(
                n_sensors=raw_sensor_dim, seq_len=LSTM_SEQ_LEN
            )
            lstm.load(str(subset_dir / 'lstm_autoencoder.pt'))
            models['LSTM Autoencoder'] = lstm
        except Exception as e:  # pragma: no cover - environment-dependent
            st.warning(
                f"LSTM Autoencoder could not be loaded for {subset} "
                f"({type(e).__name__}: {e})."
            )

    if TransformerAutoencoderDetector is not None:
        try:
            tfmr = TransformerAutoencoderDetector(
                n_sensors=raw_sensor_dim, seq_len=LSTM_SEQ_LEN
            )
            tfmr.load(str(subset_dir / 'transformer_autoencoder.pt'))
            models['Transformer Autoencoder'] = tfmr
        except Exception as e:  # pragma: no cover - environment-dependent
            st.warning(
                f"Transformer Autoencoder could not be loaded for {subset} "
                f"({type(e).__name__}: {e})."
            )

    return models


@st.cache_resource
def load_scaler(subset: str):
    """Load the StandardScaler fit during training for this subset."""
    return joblib.load(MODELS_ROOT / subset / 'scaler.pkl')


@st.cache_data
def apply_scaler(_scaler, df, feature_cols):
    """Return a copy of df with `feature_cols` standardised by the saved scaler."""
    out = df.copy()
    out[feature_cols] = _scaler.transform(df[feature_cols])
    return out


@st.cache_resource
def load_if_explainer(_iso_detector, background_matrix):
    """Build the TreeExplainer once per session for the Isolation Forest."""
    return build_explainer(_iso_detector, background_matrix, max_background=200)


def get_all_feature_columns(df):
    """Return all engineered + raw feature columns (for IF and SVM)."""
    exclude = {'unit_id', 'cycle', 'rul', 'anomaly'}
    return [c for c in df.columns if c not in exclude]


def get_raw_sensor_columns(df, kept_sensors):
    """Return raw sensor column names only (for Autoencoder)."""
    return list(kept_sensors)


# Sequence detectors consume sliding windows of raw sensors; the others
# consume per-cycle feature rows.
SEQUENCE_MODELS = {"LSTM Autoencoder", "Transformer Autoencoder"}


def get_model_features(model_name, engine_data, all_feature_cols, raw_sensor_cols):
    """Return the right feature matrix for each model type."""
    if model_name == "Autoencoder":
        X = engine_data[raw_sensor_cols].values
    elif model_name in SEQUENCE_MODELS:
        # Build sliding windows of raw sensors per engine, length T = LSTM_SEQ_LEN.
        # Returns shape (n_windows, T, n_sensors). Caller aligns scores to the
        # last cycle of each window.
        values = engine_data[raw_sensor_cols].values
        n_cycles = values.shape[0]
        if n_cycles < LSTM_SEQ_LEN:
            return np.empty((0, LSTM_SEQ_LEN, len(raw_sensor_cols)))
        windows = np.stack([
            values[i - LSTM_SEQ_LEN : i]
            for i in range(LSTM_SEQ_LEN, n_cycles + 1)
        ])
        return np.nan_to_num(windows, nan=0.0)
    else:
        X = engine_data[all_feature_cols].values
    return np.nan_to_num(X, nan=0.0)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    # ── Sidebar (subset selector first; everything downstream is keyed on it)
    st.sidebar.title("🔧 Configuration")

    # Only offer subsets that actually have saved artefacts (graceful degrade
    # if the training script hasn't been run for some subset).
    available_subsets = [
        s for s in SUBSETS if (MODELS_ROOT / s / 'isolation_forest.pkl').exists()
    ]
    if not available_subsets:
        st.error(
            "No trained subsets found under `models/<SUBSET>/`. "
            "Run `python scripts/train_subsets.py` first."
        )
        st.stop()

    subset = st.sidebar.selectbox(
        "C-MAPSS Subset",
        available_subsets,
        format_func=lambda s: (
            f"{s} ({SUBSET_REGIMES[s]} regime"
            f"{'s' if SUBSET_REGIMES[s] > 1 else ''}, "
            f"{'1 fault' if s in ('FD001', 'FD002') else '2 faults'})"
        ),
        help=(
            "FD001: 1 regime / 1 fault. FD002: 6 regimes / 1 fault. "
            "FD003: 1 regime / 2 faults. FD004: 6 regimes / 2 faults. "
            "See notebooks/06 for the cross-subset comparison."
        ),
    )

    # Load subset-specific data and models. Cached by subset key, so
    # switching is fast after the first visit.
    data, kept_sensors = load_and_process_data(subset)
    all_feature_cols = get_all_feature_columns(data)
    raw_sensor_cols = get_raw_sensor_columns(data, kept_sensors)
    models = load_models(subset, raw_sensor_dim=len(raw_sensor_cols))
    scaler = load_scaler(subset)
    scaled_data = apply_scaler(scaler, data, all_feature_cols)

    # Only offer the detectors that actually loaded
    available_models = [
        name for name in (
            "Isolation Forest", "Autoencoder", "One-Class SVM",
            "LSTM Autoencoder", "Transformer Autoencoder",
        ) if name in models
    ]
    model_name = st.sidebar.selectbox(
        "Anomaly Detection Model",
        available_models,
    )

    if _SEQ_IMPORT_ERROR is not None:
        st.sidebar.caption(
            f"_Sequence models (LSTM / Transformer) unavailable on this "
            f"deploy: {_SEQ_IMPORT_ERROR}_"
        )

    threshold = st.sidebar.slider(
        "Anomaly Score Threshold", 0.0, 1.0, 0.5, 0.01
    )

    engine_ids = sorted(data["unit_id"].unique())
    selected_engine = st.sidebar.selectbox(
        "Select Engine Unit",
        engine_ids,
        format_func=lambda x: f"Engine #{x}",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**Subset:** {subset} "
        f"({SUBSET_REGIMES[subset]} regime"
        f"{'s' if SUBSET_REGIMES[subset] > 1 else ''}, "
        f"{len(raw_sensor_cols)} sensors kept)\n\n"
        "**Dataset:** NASA C-MAPSS Turbofan\n\n"
        "**Author:** Anjana Bandara\n\n"
        "MSc AI & Data Science"
    )

    # ── Prepare engine data ─────────────────────────────────────────────
    engine_mask = data["unit_id"] == selected_engine
    engine_data = data[engine_mask].copy().sort_values("cycle").reset_index(drop=True)
    scaled_engine_data = (
        scaled_data[engine_mask].copy().sort_values("cycle").reset_index(drop=True)
    )

    # Get correct feature matrix for selected model (always from scaled view)
    X_engine = get_model_features(
        model_name, scaled_engine_data, all_feature_cols, raw_sensor_cols
    )

    # Run selected model
    selected_model = models[model_name]

    # Sequence models are windowed: one score per window, positioned at the
    # last cycle. The first (LSTM_SEQ_LEN - 1) cycles have no score (NaN).
    if model_name in SEQUENCE_MODELS:
        n_cycles = len(engine_data)
        per_cycle_scores = np.full(n_cycles, np.nan)
        if X_engine.shape[0] > 0:
            window_scores = selected_model.score_samples(X_engine)
            # Score at cycle i = score of the window ending at cycle i (i ≥ T-1)
            per_cycle_scores[LSTM_SEQ_LEN - 1 :] = window_scores
        raw_scores = per_cycle_scores
        finite = raw_scores[~np.isnan(raw_scores)]
        if finite.size > 0 and finite.max() > finite.min():
            norm_scores = (raw_scores - finite.min()) / (finite.max() - finite.min())
        else:
            norm_scores = np.zeros_like(raw_scores)
        engine_data["anomaly_score"] = norm_scores
        # NaN > threshold is False, so cold-start cycles never flag — correct.
        engine_data["predicted_anomaly"] = (
            (norm_scores > threshold) & ~np.isnan(norm_scores)
        ).astype(int)
    else:
        raw_scores = selected_model.score_samples(X_engine)
        score_min, score_max = raw_scores.min(), raw_scores.max()
        if score_max > score_min:
            norm_scores = (raw_scores - score_min) / (score_max - score_min)
        else:
            norm_scores = np.zeros_like(raw_scores)
        engine_data["anomaly_score"] = norm_scores
        engine_data["predicted_anomaly"] = (norm_scores > threshold).astype(int)

    # ── Header ──────────────────────────────────────────────────────────
    st.title("🔧 Industrial Sensor Anomaly Detection")
    st.markdown(
        f"Real-time monitoring and anomaly detection for turbofan engine "
        f"sensors — **{subset}** subset"
    )

    # ── KPI cards ───────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    total_cycles = len(engine_data)
    anomaly_count = int(engine_data["predicted_anomaly"].sum())
    anomaly_rate = anomaly_count / total_cycles

    with col1:
        st.metric("Total Cycles", total_cycles)
    with col2:
        st.metric("Anomalies Detected", anomaly_count)
    with col3:
        st.metric("Anomaly Rate", f"{anomaly_rate:.1%}")

    st.markdown("---")

    # ── Sensor time-series with anomaly overlay ─────────────────────────
    st.subheader(f"📈 Sensor Readings — Engine #{selected_engine}")

    # Only show raw sensor columns
    display_sensor_cols = [c for c in engine_data.columns if c.startswith("sensor_")
                           and "_roll_" not in c and "_lag_" not in c
                           and "_diff_" not in c and "_ewma_" not in c
                           and "_skew_" not in c and "_kurt_" not in c]

    selected_sensors = st.multiselect(
        "Select Sensors",
        display_sensor_cols,
        default=display_sensor_cols[:3],
    )

    if selected_sensors:
        fig = make_subplots(
            rows=len(selected_sensors),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            subplot_titles=selected_sensors,
        )

        anomaly_mask = engine_data["predicted_anomaly"] == 1

        for i, sensor in enumerate(selected_sensors):
            fig.add_trace(
                go.Scatter(
                    x=engine_data["cycle"],
                    y=engine_data[sensor],
                    mode="lines",
                    name=sensor,
                    line=dict(color="#4fc3f7", width=1.5),
                    showlegend=(i == 0),
                ),
                row=i + 1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=engine_data.loc[anomaly_mask, "cycle"],
                    y=engine_data.loc[anomaly_mask, sensor],
                    mode="markers",
                    name="Anomaly" if i == 0 else None,
                    marker=dict(color="#ef5350", size=4, symbol="x"),
                    showlegend=(i == 0),
                ),
                row=i + 1, col=1,
            )

        fig.update_layout(
            height=200 * len(selected_sensors),
            template="plotly_dark",
            margin=dict(l=60, r=20, t=40, b=40),
        )
        st.plotly_chart(fig, width='stretch')

    # ── Anomaly score timeline ──────────────────────────────────────────
    st.subheader("🎯 Anomaly Score Timeline")

    fig_score = go.Figure()
    fig_score.add_trace(
        go.Scatter(
            x=engine_data["cycle"],
            y=engine_data["anomaly_score"],
            mode="lines",
            fill="tozeroy",
            line=dict(color="#ab47bc"),
            fillcolor="rgba(171, 71, 188, 0.2)",
            name="Anomaly Score",
        )
    )
    fig_score.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#ef5350",
        annotation_text=f"Threshold ({threshold:.2f})",
    )
    fig_score.update_layout(
        height=300,
        template="plotly_dark",
        xaxis_title="Cycle",
        yaxis_title="Anomaly Score (normalized)",
        margin=dict(l=60, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_score, width='stretch')

    # ── Model comparison ────────────────────────────────────────────────
    st.subheader(f"🏆 Model Comparison — {subset}")
    if SUBSET_REGIMES[subset] > 1:
        _regime_note = " (and per-regime normalised before that)"
    else:
        _regime_note = ""
    st.caption(
        "Evaluated on the held-out 20%-engines test split (seed=42), "
        "matching the per-subset training protocol. Inputs are standardised "
        f"with the saved StandardScaler{_regime_note}; each detector uses "
        "its F1-optimal threshold from training."
    )

    # Same 80/20 unit-level split as the training notebook
    _train_split, test_split = train_test_split_by_unit(
        scaled_data, test_ratio=0.2, seed=42
    )
    X_test = np.nan_to_num(test_split[all_feature_cols].values, nan=0.0)
    X_test_raw = np.nan_to_num(test_split[raw_sensor_cols].values, nan=0.0)
    y_test = test_split["anomaly"].values

    # Sequence models use a windowed test population (slightly fewer samples
    # — the first T-1 cycles of each engine drop out). Only build the windows
    # when at least one sequence model loaded.
    if any(name in models for name in SEQUENCE_MODELS):
        X_test_seq, y_test_seq = create_sequences(
            test_split, raw_sensor_cols, sequence_length=LSTM_SEQ_LEN
        )
    else:
        X_test_seq = y_test_seq = None

    comparison_rows = []
    for name, model in models.items():
        if name == "Autoencoder":
            X_for_model, y_for_model = X_test_raw, y_test
        elif name in SEQUENCE_MODELS:
            X_for_model, y_for_model = X_test_seq, y_test_seq
        else:
            X_for_model, y_for_model = X_test, y_test
        scores = model.score_samples(X_for_model)
        preds = model.predict(X_for_model)
        result = evaluate_model(name, y_for_model, preds, scores)
        comparison_rows.append({
            "Model": name,
            "Precision": round(result.precision, 3),
            "Recall": round(result.recall, 3),
            "F1 Score": round(result.f1, 3),
            "AUC-PR": round(result.auc_pr, 3),
        })

    comparison_df = pd.DataFrame(comparison_rows)

    fig_comp = go.Figure()
    for metric in ["Precision", "Recall", "F1 Score", "AUC-PR"]:
        fig_comp.add_trace(
            go.Bar(
                name=metric,
                x=comparison_df["Model"],
                y=comparison_df[metric],
                text=comparison_df[metric],
                textposition="outside",
            )
        )
    fig_comp.update_layout(
        barmode="group",
        height=400,
        template="plotly_dark",
        yaxis_range=[0, 1.05],
        margin=dict(l=60, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_comp, width='stretch')

    st.dataframe(comparison_df.set_index("Model"), width='stretch')

    # ── Explainability (Isolation Forest) ───────────────────────────────
    st.markdown("---")
    st.subheader("🔍 Why was this flagged? — Isolation Forest SHAP")
    st.caption(
        "SHAP values attribute the Isolation Forest's anomaly score back to "
        "individual engineered features. Positive bars push a sample towards "
        "anomalous; values are aggregated across the cycles of the selected "
        "engine. Available for Isolation Forest only — multi-model SHAP is "
        "planned as future work."
    )

    if _SHAP_IMPORT_ERROR is not None:
        st.info(
            "The SHAP panel is unavailable on this deploy because the `shap` "
            "library failed to import "
            f"({_SHAP_IMPORT_ERROR}). It runs locally — see "
            "`notebooks/05_shap_narratives.ipynb` for the full walkthrough."
        )
    elif st.toggle("Compute SHAP for this engine", value=False):
        # Background and explain set must both be in the model's training
        # scale (StandardScaler-normalised), same as everywhere else.
        healthy_mask = scaled_data["anomaly"] == 0
        background = np.nan_to_num(
            scaled_data.loc[healthy_mask, all_feature_cols].values, nan=0.0
        )
        X_engine_all = np.nan_to_num(
            scaled_engine_data[all_feature_cols].values, nan=0.0
        )

        with st.spinner("Computing SHAP values…"):
            iso_detector = models["Isolation Forest"]
            explainer = load_if_explainer(iso_detector, background)
            explanation = explain(
                iso_detector,
                X_engine_all,
                all_feature_cols,
                explainer=explainer,
            )

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Top base sensors by total |SHAP|**")
            sensor_top = explanation.per_sensor.head(10)
            fig_sensor = go.Figure(
                go.Bar(
                    x=sensor_top["total_abs_shap"][::-1],
                    y=sensor_top["base_sensor"][::-1],
                    orientation="h",
                    marker_color="#4fc3f7",
                )
            )
            fig_sensor.update_layout(
                height=320,
                template="plotly_dark",
                margin=dict(l=80, r=20, t=20, b=40),
                xaxis_title="Σ |SHAP|",
            )
            st.plotly_chart(fig_sensor, width='stretch')

        with col_b:
            st.markdown("**Contribution by feature family**")
            fam = explanation.per_family
            fig_fam = go.Figure(
                go.Bar(
                    x=fam["total_abs_shap"],
                    y=fam["family"],
                    orientation="h",
                    marker_color="#ab47bc",
                )
            )
            fig_fam.update_layout(
                height=320,
                template="plotly_dark",
                margin=dict(l=80, r=20, t=20, b=40),
                xaxis_title="Σ |SHAP|",
            )
            st.plotly_chart(fig_fam, width='stretch')

        # Per-cycle drill-down
        st.markdown("**Per-cycle drill-down**")
        cycle_choice = st.slider(
            "Cycle",
            min_value=int(engine_data["cycle"].min()),
            max_value=int(engine_data["cycle"].max()),
            value=int(engine_data["cycle"].iloc[-1]),
        )
        sample_idx = int(
            (engine_data["cycle"] == cycle_choice).idxmax()
            - engine_data.index[0]
        )

        # Plain-English narrative for this cycle, rendered above the chart
        st.info(narrate(explanation, sample_idx, k=5))

        top_k = top_features_for_sample(explanation, sample_idx, k=10)
        colors = [
            "#ef5350" if v > 0 else "#66bb6a" for v in top_k["shap_value"]
        ]
        # Pretty labels on the y-axis instead of raw feature names
        pretty_labels = [pretty_feature_label(f) for f in top_k["feature"]]
        fig_sample = go.Figure(
            go.Bar(
                x=top_k["shap_value"],
                y=pretty_labels,
                orientation="h",
                marker_color=colors,
                hovertext=top_k["feature"],
                hovertemplate="%{hovertext}<br>SHAP: %{x:.4f}<extra></extra>",
            )
        )
        fig_sample.update_layout(
            height=400,
            template="plotly_dark",
            margin=dict(l=240, r=20, t=20, b=40),
            xaxis_title="SHAP value (→ anomaly)",
        )
        st.plotly_chart(fig_sample, width='stretch')

        # Feature glossary — collapsed by default
        with st.expander("📖 Feature glossary — what do these names mean?"):
            st.caption(
                "The model sees 184 engineered features built from 15 raw "
                "sensors. Each name follows one of these patterns:"
            )
            st.dataframe(feature_glossary(), width='stretch', hide_index=True)


if __name__ == "__main__":
    main()
