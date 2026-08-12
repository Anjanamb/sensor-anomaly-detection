# Sensor anomaly detection on C-MAPSS FD004

Onset-of-degradation detection on NASA's C-MAPSS turbofan simulator
(FD004 subset). Two unsupervised detectors trained on the healthy portion
of each engine's life only, then scored across the full run-to-failure
trajectory.

## Headline results

On the 249 training engines (trained on healthy-only cycles, scored on all):

| Detector         | Engines flagged | Median lead time | Precision | Recall |
| ---------------- | --------------: | ---------------: | --------: | -----: |
| Isolation Forest |       249 / 249 |   **210 cycles** |      0.43 |   1.00 |
| DBSCAN           |        90 / 249 |      13.5 cycles |      0.75 |   0.06 |

Isolation Forest flags every failing engine well before failure at the
cost of many cycles outside the RUL <= 30 window. DBSCAN is stricter and
fires only when its density criterion is clearly violated; when it does
fire, Isolation Forest agreed with it on 614 out of 617 cycles (99.5%).

### RUL regression on the FD004 test set (for external comparison)

The same 20 features are also used as input to two regressors evaluated
on the FD004 test set against `RUL_FD004.txt`, with RUL clipped at 125
(Heimes 2008 convention):

| Regressor     | Test RMSE | NASA Score   |
| ------------- | --------: | -----------: |
| Ridge         |     20.04 |        2,207 |
| Random Forest | **16.50** |    **1,443** |

Both land in the "strong deep model" band typical for FD004 in review
articles (Vollert & Theissler 2021), without windowed inputs or
hyperparameter search. Notebook 07 walks through the pipeline and the
comparison with published methods.

## What's here

- **FD004 only.** 249 training engines, 6 operating regimes, 2 failure
  modes. The hardest of the four C-MAPSS subsets.
- **20 features on 8 sensors.** EWMA (smoothed level), deviation from a
  per-engine healthy baseline, and rolling slope on the sensors that
  actually shift between early and late life. Each family is justified in
  notebook 02 with a hypothesis, the math, and a validation plot.
- **Two detectors.** Isolation Forest (scoring) and DBSCAN (density
  clustering; noise = anomaly). Both trained unsupervised on the healthy
  portion of engine lives.
- **Lead time as the headline metric**, not F1. Notebook 05 explains why.

## Quick start

```bash
git clone https://github.com/Anjanamb/sensor-anomaly-detection.git
cd sensor-anomaly-detection
python -m venv venv
source venv/Scripts/activate     # Windows Git Bash; on macOS/Linux use bin/activate
pip install -r requirements.txt
jupyter lab
```

Then open `notebooks/00_intro.ipynb` and work through the numbered
notebooks in order.

## Notebooks

| #  | File                                                                        | What it does                                                                                                              |
| -- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| 00 | [`00_intro.ipynb`](notebooks/00_intro.ipynb)                                | The problem, dataset, framing (onset of degradation), roadmap                                                             |
| 01 | [`01_load_and_eda.ipynb`](notebooks/01_load_and_eda.ipynb)                  | Load FD004, describe engines and cycles, identify the 6 operating regimes, rank sensors by degradation signal             |
| 02 | [`02_feature_engineering.ipynb`](notebooks/02_feature_engineering.ipynb)    | Drop dead sensors per regime, per-regime z-score, build 20 features (EWMA / deviation / rolling slope)                    |
| 03 | [`03_train_iforest.ipynb`](notebooks/03_train_iforest.ipynb)                | Split into healthy vs everything, fit IF, score, pick a threshold via 95th percentile of healthy scores                   |
| 04 | [`04_train_dbscan.ipynb`](notebooks/04_train_dbscan.ipynb)                  | Feature-warmup exclusion, standardise, pick min_samples and eps via k-distance plot, fit, PCA projection, compare with IF |
| 05 | [`05_evaluate.ipynb`](notebooks/05_evaluate.ipynb)                          | Why lead time is the headline (not F1), per-detector distributions, per-engine agreement analysis                         |
| 06 | [`06_visuals.ipynb`](notebooks/06_visuals.ipynb)                            | Four summary plots: hero score-vs-cycle grid, PCA colored by RUL, lead-time reverse CDF, per-engine detection heatmap     |
| 07 | [`07_rul_regression.ipynb`](notebooks/07_rul_regression.ipynb)              | Ridge + Random Forest on the same 20 features, scored on the FD004 test set (RMSE + NASA Score) for external benchmark    |

## Repository layout

```text
sensor-anomaly-detection/
├── data/                                  FD004 raw files + generated CSVs (gitignored)
│   ├── train_FD004.txt
│   ├── test_FD004.txt
│   └── RUL_FD004.txt
├── notebooks/                             The seven notebooks
├── src/                                   Thin reusable modules
│   ├── data.py         Loader + RUL helpers
│   ├── features.py     Rolling primitives (mean, std, ewma, slope, deviation)
│   ├── models.py       IF and DBSCAN wrappers
│   └── evaluate.py     Lead-time and precision/recall helpers
├── tests/                                 Sanity checks for src/
├── requirements.txt                       numpy, pandas, sklearn, matplotlib, seaborn, jupyter, pytest
├── Dockerfile                             Reproducible env for anyone who wants a container
└── LICENSE                                MIT
```

## Running the tests

```bash
python -m pytest tests/ -q
```

## Running in Docker

```bash
docker build -t sensor-anomaly-detection .
docker run --rm -p 8888:8888 -v "$(pwd)":/app sensor-anomaly-detection
```

Then open the Jupyter Lab URL printed in the terminal.

## License

MIT. See [`LICENSE`](LICENSE).
