"""
Industrial Sensor Anomaly Detection Dashboard
Main Streamlit application.
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_cmapss, add_rul_to_train, create_anomaly_labels, get_sensor_columns
from src.preprocessing import remove_constant_sensors, normalize_global
from src.feature_engineering import build_feature_pipeline
from src.models import IsolationForestDetector, AutoencoderDetector, OneClassSVMDetector
from src.evaluation import evaluate_model

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

@st.cache_data
def load_and_process_data():
    """Load C-MAPSS, add RUL/anomaly labels, engineer features."""
    train_df, _test_df, _rul_df = load_cmapss('FD001')
    train_df = add_rul_to_train(train_df)
    train_df = create_anomaly_labels(train_df, threshold=30)

    sensor_cols = get_sensor_columns(train_df)
    train_df, kept_sensors = remove_constant_sensors(train_df, sensor_cols)

    featured_df = build_feature_pipeline(
        train_df, kept_sensors,
        rolling_windows=[5, 10], lags=[1, 5], ewma_spans=[5]
    )
    return featured_df, kept_sensors


@st.cache_resource
def load_models(all_feature_dim, raw_sensor_dim):
    """Load all three saved models from disk."""
    models = {}

    iso = IsolationForestDetector()
    iso.load('models/isolation_forest.pkl')
    models['Isolation Forest'] = iso

    ae = AutoencoderDetector(input_dim=raw_sensor_dim)
    ae.load('models/autoencoder.pt')
    models['Autoencoder'] = ae

    svm = OneClassSVMDetector()
    svm.load('models/one_class_svm.pkl')
    models['One-Class SVM'] = svm

    return models


def get_all_feature_columns(df):
    """Return all engineered + raw feature columns (for IF and SVM)."""
    exclude = {'unit_id', 'cycle', 'rul', 'anomaly'}
    return [c for c in df.columns if c not in exclude]


def get_raw_sensor_columns(df, kept_sensors):
    """Return raw sensor column names only (for Autoencoder)."""
    return list(kept_sensors)


def get_model_features(model_name, engine_data, all_feature_cols, raw_sensor_cols):
    """Return the right feature matrix for each model type."""
    if model_name == "Autoencoder":
        X = engine_data[raw_sensor_cols].values
    else:
        X = engine_data[all_feature_cols].values
    return np.nan_to_num(X, 0)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    # Load real data and models
    data, kept_sensors = load_and_process_data()
    all_feature_cols = get_all_feature_columns(data)
    raw_sensor_cols = get_raw_sensor_columns(data, kept_sensors)
    models = load_models(
        all_feature_dim=len(all_feature_cols),
        raw_sensor_dim=len(raw_sensor_cols)
    )

    # ── Sidebar ─────────────────────────────────────────────────────────
    st.sidebar.title("🔧 Configuration")

    model_name = st.sidebar.selectbox(
        "Anomaly Detection Model",
        ["Isolation Forest", "Autoencoder", "One-Class SVM"],
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
        "**Dataset:** NASA C-MAPSS Turbofan\n\n"
        "**Author:** Anjana Bandara\n\n"
        "MSc AI & Data Science"
    )

    # ── Prepare engine data ─────────────────────────────────────────────
    engine_data = data[data["unit_id"] == selected_engine].copy()
    engine_data = engine_data.sort_values("cycle").reset_index(drop=True)

    # Get correct feature matrix for selected model
    X_engine = get_model_features(model_name, engine_data, all_feature_cols, raw_sensor_cols)

    # Run selected model
    selected_model = models[model_name]
    raw_scores = selected_model.score_samples(X_engine)

    # Normalize scores to 0-1 range for the threshold slider
    score_min, score_max = raw_scores.min(), raw_scores.max()
    if score_max > score_min:
        norm_scores = (raw_scores - score_min) / (score_max - score_min)
    else:
        norm_scores = np.zeros_like(raw_scores)

    engine_data["anomaly_score"] = norm_scores
    engine_data["predicted_anomaly"] = (norm_scores > threshold).astype(int)

    # ── Header ──────────────────────────────────────────────────────────
    st.title("🔧 Industrial Sensor Anomaly Detection")
    st.markdown("Real-time monitoring and anomaly detection for turbofan engine sensors")

    # ── KPI cards ───────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    total_cycles = len(engine_data)
    anomaly_count = int(engine_data["predicted_anomaly"].sum())
    anomaly_rate = anomaly_count / total_cycles
    current_rul = int(engine_data["rul"].iloc[-1])

    with col1:
        st.metric("Total Cycles", total_cycles)
    with col2:
        st.metric("Anomalies Detected", anomaly_count)
    with col3:
        st.metric("Anomaly Rate", f"{anomaly_rate:.1%}")
    with col4:
        st.metric(
            "Est. RUL",
            f"{current_rul} cycles",
            delta="⚠️ Critical" if current_rul < 30 else "✅ Healthy",
        )

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
    st.subheader("🏆 Model Comparison")
    st.caption("Evaluated on all engine data with ground-truth anomaly labels (RUL ≤ 30)")

    X_all = np.nan_to_num(data[all_feature_cols].values, 0)
    X_all_raw = np.nan_to_num(data[raw_sensor_cols].values, 0)
    y_all = data["anomaly"].values

    comparison_rows = []
    for name, model in models.items():
        if name == "Autoencoder":
            scores = model.score_samples(X_all_raw)
        else:
            scores = model.score_samples(X_all)
        preds = model.predict(X_all_raw if name == "Autoencoder" else X_all)
        result = evaluate_model(name, y_all, preds, scores)
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


if __name__ == "__main__":
    main()
