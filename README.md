# 🔧 Industrial Sensor Anomaly Detection Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sensor-anomaly-detection-aj.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end machine learning pipeline for detecting anomalies in turbofan engine sensor data, with an interactive Streamlit dashboard for real-time monitoring and multi-model comparison.

> **[🚀 Live Demo](https://sensor-anomaly-detection-aj.streamlit.app/)**

<!-- Uncomment and update path once you add a screenshot:
![Dashboard Screenshot](assets/dashboard_screenshot.png)
-->

---

## Overview

Predictive maintenance is critical in industrial settings where unexpected equipment failure leads to costly downtime. This project demonstrates a complete anomaly detection workflow — from raw sensor ingestion to an interactive monitoring dashboard — using the NASA C-MAPSS turbofan engine degradation dataset.

The pipeline compares three fundamentally different anomaly detection approaches, all trained exclusively on healthy engine data so they learn what "normal" looks like and flag deviations:

| Model | Type | Approach |
|-------|------|----------|
| **Isolation Forest** | Tree-based | Isolates anomalies by recursive random partitioning |
| **Autoencoder** | Neural network (PyTorch) | Learns to reconstruct normal sensor patterns; high reconstruction error signals anomalies |
| **One-Class SVM** | Kernel-based | Fits a decision boundary around normal data in high-dimensional feature space |

### Results

Evaluated on a held-out set of 20 engine units (4,291 samples, 14.4% anomalous):

| Model | Precision | Recall | F1 | AUC-PR | AUC-ROC |
|-------|-----------|--------|-----|--------|---------|
| **Isolation Forest** | **0.734** | **0.826** | **0.777** | **0.701** | **0.957** |
| One-Class SVM | 0.609 | 0.773 | 0.681 | 0.650 | 0.926 |
| Autoencoder | 0.362 | 0.485 | 0.415 | 0.384 | 0.766 |

Isolation Forest achieves the best overall performance. The Autoencoder, trained on raw sensor readings (15 features) rather than the full engineered feature set (184 features), captures nonlinear patterns but is limited by its feedforward architecture — a sequence-based LSTM autoencoder would be a natural next step.

---

## Features

- **300+ engineered time-series features** — rolling statistics, exponentially weighted moving averages, lag/diff features, skewness, and kurtosis computed per engine unit
- **Three anomaly detection models** trained on healthy data only, with optimal thresholds selected via PR curve analysis
- **Interactive Streamlit dashboard** with per-engine sensor visualisation, adjustable anomaly thresholds, anomaly score timelines, and a live model comparison chart
- **Modular, testable codebase** — separate modules for data loading, preprocessing, feature engineering, models, and evaluation
- **Dockerised** for reproducible deployment

---

## Project Structure

```
sensor-anomaly-detection/
├── app/
│   └── streamlit_app.py          # Interactive dashboard
├── data/
│   ├── README.md                 # Dataset download instructions
│   ├── train_FD001.txt           # Training data (run-to-failure)
│   ├── test_FD001.txt            # Test data
│   └── RUL_FD001.txt             # Ground-truth RUL
├── models/
│   ├── isolation_forest.pkl      # Trained Isolation Forest
│   ├── autoencoder.pt            # Trained Autoencoder weights
│   └── one_class_svm.pkl         # Trained One-Class SVM
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_comparison.ipynb # Training, evaluation, PR curves
├── src/
│   ├── data_loader.py            # C-MAPSS ingestion & RUL labelling
│   ├── preprocessing.py          # Normalisation, splitting, cleaning
│   ├── feature_engineering.py    # Rolling, lag, EWMA, statistical features
│   ├── evaluation.py             # Precision, Recall, F1, AUC-PR, AUC-ROC
│   └── models/
│       ├── isolation_forest.py
│       ├── autoencoder.py
│       └── one_class_svm.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_models.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone and set up

```bash
git clone https://github.com/Anjanamb/sensor-anomaly-detection.git
cd sensor-anomaly-detection

python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

### 2. Explore the notebooks

```bash
jupyter notebook notebooks/
```

Run in order: `01_eda.ipynb` → `02_feature_engineering.ipynb` → `03_model_comparison.ipynb`

### 3. Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

### 4. Run tests

```bash
pytest tests/ -v
```

---

## Dataset

**NASA C-MAPSS Turbofan Engine Degradation Simulation**

Run-to-failure sensor recordings from 100 turbofan engines (FD001 subset), each with 21 sensor channels and 3 operational settings. Engines operate normally at first, then develop faults that progress until failure.

- **Source:** [NASA Open Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
- **Reference:** A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM08, Denver CO, 2008.

---

## Approach

### Preprocessing

- Removed 6 constant-variance sensors (no degradation signal)
- Normalised readings globally with StandardScaler
- Split data by engine unit (not by row) to prevent data leakage
- Trained models on **healthy data only** (RUL > 30) so they learn what "normal" looks like

### Feature Engineering

From 15 active sensors, generated 180+ features per time step:

- **Rolling statistics** (mean, std) at windows of 5 and 10 cycles
- **Lag and difference features** capturing rate-of-change at 1 and 5 cycle offsets
- **Exponentially weighted moving averages** (span of 5) for trend detection
- **Higher-order statistics** (skewness, kurtosis) over 20-cycle windows to detect distributional shifts
- **Normalised cycle position** (0→1) representing lifecycle progress

### Model Design Choices

- **Isolation Forest & One-Class SVM** use all 184 engineered features — tree and kernel methods handle correlated features well
- **Autoencoder** uses 15 raw sensor columns only — neural networks struggle with redundant engineered features, so raw sensors give a cleaner learning signal
- **Thresholds** optimised via Precision-Recall curve analysis (maximising F1) rather than default model thresholds

### Anomaly Labelling

Binary labels derived from Remaining Useful Life: samples with RUL ≤ 30 cycles are labelled as anomalous (~15% of the dataset), reflecting the degradation zone approaching failure.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| **ML / Data** | Python, PyTorch, scikit-learn, pandas, NumPy, SciPy |
| **Visualisation** | Plotly, Matplotlib, Seaborn |
| **Dashboard** | Streamlit |
| **Testing** | pytest |
| **Deployment** | Docker, Streamlit Cloud |

---

## Docker

```bash
docker build -t sensor-anomaly .
docker run -p 8501:8501 sensor-anomaly
```

Then open `http://localhost:8501`.

---

## Author

**Anjana Bandara**
MSc Artificial Intelligence & Data Science — Heinrich Heine University Düsseldorf

[![LinkedIn](https://img.shields.io/badge/LinkedIn-anjana--b-blue?logo=linkedin)](https://linkedin.com/in/anjana-b-)
[![GitHub](https://img.shields.io/badge/GitHub-Anjanamb-181717?logo=github)](https://github.com/Anjanamb)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.