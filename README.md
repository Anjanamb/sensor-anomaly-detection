# Sensor anomaly detection on C-MAPSS FD004

Onset-of-degradation detection on NASA's C-MAPSS turbofan simulator, done as
a learning-first walkthrough. Seven Jupyter notebooks take a reader from
"never touched this dataset" to "I understand why each feature exists and
what each detector is doing".

> **Looking for the *why* behind the choices?** See
> [`JOURNEY.md`](JOURNEY.md). It documents the v1 kitchen-sink version and
> the reasoning behind the v2 refine.

---

## What this project does

- Loads C-MAPSS **FD004** (249 training engines, run-to-failure, 21 sensors
  across 6 operating conditions and 2 failure modes).
- Engineers 20 features on 8 sensors that carry a degradation signal. Every
  feature is justified with a hypothesis, math, and a validation plot.
- Trains two unsupervised detectors on the *healthy* portion of engine
  lives only:
  - **Isolation Forest** (tree-based, scoring)
  - **DBSCAN** (density-based, clustering; noise = anomaly)
- Evaluates by *lead time*: how many cycles before failure did the first
  above-threshold flag fire?

## Headline results

On the 249 training engines (fitted on healthy-only cycles, scored on all):

| Detector         | Engines flagged | Median lead time | Precision | Recall |
| ---------------- | --------------: | ---------------: | --------: | -----: |
| Isolation Forest |       249 / 249 |   **210 cycles** |      0.43 |   1.00 |
| DBSCAN           |        90 / 249 |      13.5 cycles |      0.75 |   0.06 |

The trade-off is by design. Isolation Forest is a broad early-warning
system: it flags every failing engine with substantial warning, at the cost
of many "false positives" that are really just early warnings falling
outside the RUL <= 30 window. DBSCAN is a strict late confirmer: when it
fires, Isolation Forest agreed with it 99.5% of the time (614 out of 617
DBSCAN noise cycles were also IF-flagged).

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

Work through them in order. Each opens with a "What you will learn" list
and closes with a "Takeaways for the next notebook" section, so the flow
is continuous.

| #  | File                                                                        | What it does                                                                                                              |
| -- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| 00 | [`00_intro.ipynb`](notebooks/00_intro.ipynb)                                | The problem, dataset, framing (onset of degradation), roadmap                                                             |
| 01 | [`01_load_and_eda.ipynb`](notebooks/01_load_and_eda.ipynb)                  | Load FD004, describe engines and cycles, identify the 6 operating regimes, rank sensors by degradation signal             |
| 02 | [`02_feature_engineering.ipynb`](notebooks/02_feature_engineering.ipynb)    | Drop dead sensors per regime, per-regime z-score, build 20 hypothesis-driven features (EWMA / deviation / rolling slope)  |
| 03 | [`03_train_iforest.ipynb`](notebooks/03_train_iforest.ipynb)                | Split into healthy vs everything, fit IF, score, pick a threshold via 95th percentile of healthy scores                   |
| 04 | [`04_train_dbscan.ipynb`](notebooks/04_train_dbscan.ipynb)                  | Feature-warmup exclusion, standardise, pick min_samples and eps via k-distance plot, fit, PCA projection, compare with IF |
| 05 | [`05_evaluate.ipynb`](notebooks/05_evaluate.ipynb)                          | Why lead time is the headline (not F1), per-detector distributions, per-engine agreement analysis                         |
| 06 | [`06_visuals.ipynb`](notebooks/06_visuals.ipynb)                            | Four summary plots: hero score-vs-cycle grid, PCA colored by RUL, lead-time reverse CDF, per-engine detection heatmap     |

## Repository layout

```text
sensor-anomaly-detection/
├── data/                                  FD004 raw files + generated CSVs (gitignored)
│   ├── train_FD004.txt
│   ├── test_FD004.txt
│   └── RUL_FD004.txt
├── notebooks/                             The seven learning notebooks
│   └── ...
├── src/                                   Thin reusable modules
│   ├── data.py         Loader + RUL helpers
│   ├── features.py     Rolling primitives (mean, std, ewma, slope, deviation)
│   ├── models.py       IF and DBSCAN wrappers
│   └── evaluate.py     Lead-time and precision/recall helpers
├── tests/                                 Sanity checks for src/
│   ├── test_data.py
│   ├── test_features.py
│   └── test_evaluate.py
├── requirements.txt                       numpy, pandas, sklearn, matplotlib, seaborn, jupyter, pytest
├── Dockerfile                             Reproducible env for anyone who wants a container
├── JOURNEY.md                             Design decisions and the v1 -> v2 story
└── LICENSE                                MIT
```

## What v2 does not include (deliberately)

- **No hyperparameter search.** Threshold, contamination, min_samples, eps
  are picked by defensible rules of thumb explained in the notebooks. A
  cross-validated grid search over these knobs is a natural follow-up.
- **No neural networks.** V1 shipped feedforward, LSTM, and Transformer
  autoencoders. On FD004 with well-designed features they did not beat
  Isolation Forest and were much harder to explain. See `JOURNEY.md`.
- **No Streamlit dashboard.** V1 had one; it added complexity without
  adding teaching value. The notebooks are the user interface here.
- **No FD001 / FD002 / FD003.** FD004 is the hardest subset; if the
  approach works there it works everywhere. See `JOURNEY.md` for why we
  focused down.
- **No SHAP / explainability layer.** Isolation Forest scores are already
  interpretable enough for this analysis. If you want per-cycle
  attributions on a specific engine, add SHAP tree explainer on top of
  the fitted IF model.

## Running the tests

```bash
python -m pytest tests/ -q
```

Expects ~15 assertions across the three files. All should pass.

## Version history

- **v2** (current) - one subset, two detectors, twenty features, seven
  learning-first notebooks. About one fifth the code of v1, easier to
  follow, same qualitative conclusion.
- **v1** - all four FD subsets, five detectors including three
  autoencoders, 184 engineered features, live Streamlit dashboard. Tagged
  as [`v1.0.0`](https://github.com/Anjanamb/sensor-anomaly-detection/releases/tag/v1.0.0).
  Recover with `git checkout v1.0.0`.

## License

MIT. See [`LICENSE`](LICENSE).
