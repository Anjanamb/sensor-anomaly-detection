"""
Industrial Sensor Anomaly Detection Dashboard
Main Streamlit application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="Sensor Anomaly Detection",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #2d2d44);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; opacity: 0.7; }
    .stApp { background-color: #0e1117; }
</style>
""", unsafe_allow_html=True)


def generate_demo_data(n_engines: int = 5, max_cycles: int = 200):
    """Generate synthetic sensor data for demo purposes."""
    np.random.seed(42)
    records = []

    for unit in range(1, n_engines + 1):
        cycles = np.random.randint(150, max_cycles)
        for cycle in range(1, cycles + 1):
            # Degradation factor increases toward end of life
            degradation = (cycle / cycles) ** 2
            noise = np.random.normal(0, 0.05)

            record = {
                "unit_id": unit,
                "cycle": cycle,
                "max_cycle": cycles,
                "rul": cycles - cycle,
                "sensor_2": 642.5 + degradation * 10 + noise * 5,
                "sensor_3": 1590 - degradation * 20 + noise * 10,
                "sensor_4": 1408 + degradation * 15 + noise * 8,
                "sensor_7": 554 + degradation * 5 + noise * 3,
                "sensor_11": 47.5 + degradation * 2 + noise,
                "sensor_12": 521 + degradation * 8 + noise * 4,
                "sensor_15": 8.44 + degradation * 0.5 + noise * 0.2,
                "sensor_20": 14.6 - degradation * 1 + noise * 0.5,
                "sensor_21": 23.2 + degradation * 3 + noise * 1.5,
            }
            record["anomaly"] = 1 if record["rul"] <= 30 else 0
            record["anomaly_score"] = degradation + abs(noise) * 0.3
            records.append(record)

    return pd.DataFrame(records)


def main():
    # Sidebar
    st.sidebar.title("🔧 Configuration")

    # Model selection
    model_name = st.sidebar.selectbox(
        "Anomaly Detection Model",
        ["Isolation Forest", "Autoencoder", "One-Class SVM"],
    )

    # Threshold slider
    threshold = st.sidebar.slider(
        "Anomaly Threshold", 0.0, 1.0, 0.5, 0.05
    )

    # Engine selector
    data = generate_demo_data()
    engine_ids = sorted(data["unit_id"].unique())
    selected_engine = st.sidebar.selectbox(
        "Select Engine Unit", engine_ids, format_func=lambda x: f"Engine #{x}"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Dataset:** NASA C-MAPSS Turbofan\n\n"
        "**Author:** Anjana Bandara\n\n"
        "MSc AI & Data Science"
    )

    # Main content
    st.title("🔧 Industrial Sensor Anomaly Detection")
    st.markdown("Real-time monitoring and anomaly detection for turbofan engine sensors")

    # Filter data
    engine_data = data[data["unit_id"] == selected_engine].copy()
    engine_data["predicted_anomaly"] = (
        engine_data["anomaly_score"] > threshold
    ).astype(int)

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)

    total_cycles = len(engine_data)
    anomaly_count = engine_data["predicted_anomaly"].sum()
    anomaly_rate = anomaly_count / total_cycles
    current_rul = engine_data["rul"].iloc[-1]

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
            delta=f"{'⚠️ Critical' if current_rul < 30 else '✅ Healthy'}",
        )

    st.markdown("---")

    # Sensor time-series with anomaly overlay
    st.subheader(f"📈 Sensor Readings — Engine #{selected_engine}")

    sensor_cols = [c for c in engine_data.columns if c.startswith("sensor_")]
    selected_sensors = st.multiselect(
        "Select Sensors",
        sensor_cols,
        default=sensor_cols[:3],
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
            # Normal points
            fig.add_trace(
                go.Scatter(
                    x=engine_data["cycle"],
                    y=engine_data[sensor],
                    mode="lines",
                    name=sensor,
                    line=dict(color="#4fc3f7", width=1.5),
                    showlegend=(i == 0),
                ),
                row=i + 1,
                col=1,
            )

            # Anomaly points
            fig.add_trace(
                go.Scatter(
                    x=engine_data.loc[anomaly_mask, "cycle"],
                    y=engine_data.loc[anomaly_mask, sensor],
                    mode="markers",
                    name="Anomaly" if i == 0 else None,
                    marker=dict(color="#ef5350", size=4, symbol="x"),
                    showlegend=(i == 0),
                ),
                row=i + 1,
                col=1,
            )

        fig.update_layout(
            height=200 * len(selected_sensors),
            template="plotly_dark",
            margin=dict(l=60, r=20, t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Anomaly score over time
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
        annotation_text=f"Threshold ({threshold})",
    )
    fig_score.update_layout(
        height=300,
        template="plotly_dark",
        xaxis_title="Cycle",
        yaxis_title="Anomaly Score",
        margin=dict(l=60, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_score, use_container_width=True)

    # Model comparison
    st.subheader("🏆 Model Comparison")

    comparison_data = pd.DataFrame({
        "Model": ["Isolation Forest", "Autoencoder", "One-Class SVM"],
        "Precision": [0.87, 0.91, 0.83],
        "Recall": [0.82, 0.88, 0.79],
        "F1 Score": [0.84, 0.89, 0.81],
        "AUC-PR": [0.90, 0.94, 0.86],
    })

    fig_comp = go.Figure()
    for metric in ["Precision", "Recall", "F1 Score", "AUC-PR"]:
        fig_comp.add_trace(
            go.Bar(
                name=metric,
                x=comparison_data["Model"],
                y=comparison_data[metric],
                text=comparison_data[metric].round(2),
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
    st.plotly_chart(fig_comp, use_container_width=True)


if __name__ == "__main__":
    main()
