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

Predictive maintenance is critical in industrial settings where unexpected equipment failure leads to costly downtime. This project tackles the problem of detecting early signs of degradation in turbofan jet engines using unsupervised anomaly detection — identifying when sensor readings start deviating from healthy operating patterns before a failure occurs.

The pipeline compares three fundamentally different anomaly detection approaches, all trained exclusively on healthy engine data:

| Model | F1 | AUC-ROC | Precision | Recall |
|-------|-----|---------|-----------|--------|
| **Isolation Forest** | **0.777** | **0.957** | 0.734 | 0.826 |
| One-Class SVM | 0.681 | 0.926 | 0.609 | 0.773 |
| Autoencoder | 0.415 | 0.766 | 0.362 | 0.485 |

Evaluated on a held-out set of 20 engine units (4,291 samples, 14.4% anomalous).

---

## Dataset

**NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)**

The dataset contains run-to-failure recordings from 100 simulated turbofan engines (FD001 subset). Each engine starts in a healthy state and gradually degrades until failure, with 21 sensors recording measurements like temperature, pressure, fan speed, and fuel flow at each operating cycle.

| Property | Value |
|----------|-------|
| Engines | 100 (80 train / 20 test) |
| Sensor channels | 21 (6 constant, 15 informative) |
| Operational settings | 3 |
| Samples | 20,631 total |
| Anomaly definition | RUL ≤ 30 cycles (~15% of data) |

**What counts as an anomaly?** Each engine has a Remaining Useful Life (RUL) — the number of cycles until failure. Samples where RUL ≤ 30 are labelled as anomalous, representing the degradation zone where sensor patterns visibly shift from normal operating behaviour. This threshold reflects the practical window where maintenance intervention would be needed.

**Source:** [NASA Prognostics Data Repository](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
**Reference:** Saxena et al., "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM08.

---

## Approach

### Preprocessing

- Removed 6 constant-variance sensors (sensors 1, 5, 6, 10, 16, 18, 19 — no degradation signal)
- Normalised readings globally with StandardScaler
- Split data **by engine unit** (not by row) to prevent data leakage — no engine appears in both train and test sets
- Trained all models on **healthy data only** (RUL > 30), so they learn what "normal" looks like and flag deviations

### Feature Engineering

From 15 active sensors, generated 184 features per time step:

- **Rolling statistics** (mean, std) at windows of 5 and 10 cycles — capture short-term trends
- **Lag and difference features** at 1 and 5 cycle offsets — capture rate-of-change and sudden shifts
- **Exponentially weighted moving averages** (span 5) — react faster to recent changes than simple rolling means
- **Higher-order statistics** (skewness, kurtosis) over 20-cycle windows — detect distributional shifts that precede failure
- **Normalised cycle position** (0→1) — represents lifecycle progress

### Models

#### Isolation Forest — Best overall (F1: 0.777)

A tree-based ensemble that detects anomalies by measuring how easily a data point can be isolated through random partitioning. Normal points require many splits; anomalies are isolated quickly.

**Why it works well here:** Handles the 184 correlated engineered features natively — tree splits are unaffected by feature correlation, and the ensemble (300 trees) provides stable anomaly scores. Trained with `contamination=0.05` since the training set contains only healthy data with minimal noise.

#### One-Class SVM — Strong runner-up (F1: 0.681)

A kernel-based method that learns a decision boundary enclosing normal data in a high-dimensional feature space (via RBF kernel). Points outside the boundary are flagged as anomalous.

**Why it works here:** The RBF kernel captures nonlinear relationships between engineered features. Subsampled to 10,000 training points due to O(n²) memory complexity. Trained with `nu=0.05` (upper bound on the fraction of outliers expected in the clean training set).

#### Autoencoder — Complementary approach (F1: 0.415)

A feedforward neural network (PyTorch) trained to compress and reconstruct sensor readings. The bottleneck forces the model to learn a compact representation of normal patterns. When it encounters degraded readings, reconstruction error spikes — signalling an anomaly.

**Architecture (auto-scaled to input size):**

```
Input (15) → Dense(32) → BN → LeakyReLU → Dropout(0.2)
          → Dense(16) → BN → LeakyReLU
          → Dense(8)  [bottleneck]
          → Dense(16) → BN → LeakyReLU
          → Dense(32) → BN → LeakyReLU → Dropout(0.2)
          → Dense(15) [reconstruction]
```

**Key design choice:** The autoencoder is trained on **15 raw sensor columns only**, not the full 184 engineered features. Autoencoders learn by reconstructing inputs — when features are highly correlated (e.g., rolling mean of sensor 2 and EWMA of sensor 2), the reconstruction error becomes noisy and uninformative. Raw sensors provide a cleaner learning signal.

**Why it underperforms:** A feedforward autoencoder treats each time step independently and cannot capture temporal dependencies across cycles. The degradation pattern in C-MAPSS is a gradual, sequential shift — exactly what sequence models excel at. An LSTM or Transformer-based autoencoder operating on sliding windows of raw sensor sequences would be a natural improvement.

### Threshold Selection

Rather than using each model's default decision boundary, thresholds are optimised by finding the point on the Precision-Recall curve that maximises the F1 score. This is important for imbalanced data (only ~15% anomalous) where accuracy would be misleading.

---

## Features

- **184 engineered time-series features** with rolling statistics, EWMA, lag/diff, skewness, and kurtosis
- **Three anomaly detection models** trained on healthy data only, with PR-curve-optimised thresholds
- **Interactive Streamlit dashboard** with per-engine sensor visualisation, adjustable thresholds, anomaly score timelines, and live model comparison
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
│       ├── autoencoder.py        # PyTorch autoencoder with auto-scaling layers
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

## Limitations & Future Work

- **Feedforward autoencoder ignores temporal order** — each cycle is scored independently. An LSTM or Temporal Convolutional autoencoder on sliding windows would capture sequential degradation patterns.
- **Single operating condition (FD001)** — the pipeline currently uses the simplest C-MAPSS subset. Extending to FD002–FD004 (multiple operating conditions, multiple fault modes) would test generalisation.
- **Fixed anomaly threshold (RUL ≤ 30)** — the binary labelling is a simplification. A regression approach predicting RUL directly would provide more granular prognostics.
- **No online learning** — models are trained offline. A production system would need incremental updates as new engine data arrives.

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