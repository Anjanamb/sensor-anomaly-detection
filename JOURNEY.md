# Project journey — decisions, challenges, lessons

> A first-person guide to the *why* behind this project, written for
> recruiters / interviewers who want the reasoning behind the code
> rather than the F1 numbers. **5-minute read.**

The technical README documents what's in the repo. This document
documents what's in my head: the decisions I weighed, the bugs I
hit, what I learned about jet engines / ML / shipping, and the
tools I picked + why.

---

## 1. Project journey

I started with the simplest thing: FD001, an Isolation Forest, the 21
raw sensors, a single F1 number. That worked surprisingly well
(F1 ≈ 0.78), so the question quickly became *why*. Looking at the IF's
splits and the SHAP attributions made it clear the model was reading
HPC degradation through downstream effects (bypass-duct pressure, LPT
temperature) more than through the direct HPC sensors. That meant the
data and the engine physics were doing most of the work — the model
was just confirming what the sensors already showed.

So the next question became: can a richer model do *better* than the
IF, and on what? I added a One-Class SVM (kernel-based decision
boundary), a feedforward autoencoder (reconstruction error from raw
sensors), an LSTM autoencoder (windowed reconstruction), and a
Transformer autoencoder (windowed reconstruction with self-attention).
The autoencoders all came in under the IF on FD001. That non-result
turned out to be more interesting than a win would have been — it
forced me to articulate exactly *why* the IF was hard to beat on FD001
(few engines, single fault mode, single regime, hand-engineered
temporal features), and *where* sequence models should help (FD002 /
FD004 with multiple regimes and fault modes).

The cross-subset experiment in [notebook 06](notebooks/06_fd_subset_comparison.ipynb)
ran all five detectors on all four subsets with per-regime normalisation
for FD002/FD004. The IF still won 3 of 4, but the *gap* closed
dramatically on the multi-regime subsets — the LSTM and Transformer
closed from −0.29 F1 vs the IF on FD001 to −0.05 on FD004. And on
FD003 (single regime, two fault modes) the *feedforward AE* actually
beat the IF, because reconstruction error catches multiple fault modes
uniformly while tree splits struggle to carve two separate anomaly
regions on ~100 engines.

That experiment made the project's narrative honest: not "IF
dominates", but "the right detector depends on which kind of
complexity you have, and the IF is hard to beat with neural methods
on simple tabular sensor data even before you get to
explainability".

Explainability was the other big track. [SHAP](notebooks/05_shap_narratives.ipynb)
turned `sensor_9_diff_5` into "Sensor 9 — change over 5 cycles"; the
[global feature-importance analysis](notebooks/07_feature_importance.ipynb)
found that the *top* sensor on FD001 isn't at the HPC (where the
fault is) but in the Fan section (bypass-duct pressure) — and that
the model has implicitly learned a *temporal progression* of which
sensors carry signal at which lifecycle stage. That was the most
satisfying single finding of the project.

The dashboard ([Streamlit live](https://sensor-anomaly-detection-aj.streamlit.app/),
[Power BI executive view](bi/BUILD_GUIDE.md)) is what stays
demoable in interviews — same data, complementary surfaces.

---

## 2. Key decisions

I picked **Isolation Forest** as the primary detector over XGBoost /
LightGBM and ensembled IF variants. *Why:* unsupervised anomaly
detection is the right framing (very few failure examples; want to
catch novel modes), the IF gives the strongest unsupervised baseline
for tabular sensor data, and `TreeExplainer` gives exact SHAP for free
(KernelExplainer for OC-SVM is slow and approximate; DeepExplainer
for the AE is messier on sequence inputs).

I picked **MSE** for the autoencoder reconstruction loss over MAE /
Huber. *Why:* MSE biases the model toward reconstructing the bulk of
the distribution well rather than the tails — appropriate when the
training set is healthy-only and the goal is "fail to reconstruct
when something looks weird". MAE would give more weight to outliers,
which we don't want during training on healthy data.

I picked a **30-cycle window** for the sequence models. *Why:* matches
the RUL ≤ 30 anomaly threshold (so the windowed input covers exactly
the warning-zone span). Tried 50 and 100 informally; both increased
training time and reduced the number of usable windows per engine
without lifting F1.

I picked **per-engine feature computation** over a cross-engine
approach. *Why:* prevents information leakage. A cross-engine rolling
mean would let an engine's features depend on data from other engines
that might end up in the test split.

I picked **train on healthy only (RUL > 30)** for the unsupervised
detectors rather than training on labels. *Why:* (a) honest framing
of the problem (you wouldn't have failure labels at deployment), (b)
the model learns "this is what healthy looks like" rather than a
binary classifier with very imbalanced classes, (c) downstream
anomaly attribution naturally describes "how does this differ from
healthy".

I picked **per-regime KMeans(k=6)** for FD002 / FD004 over a single
global StandardScaler. *Why:* the operating-settings space has six
clean discrete clusters; without per-regime normalisation the
multi-regime variance drowns out the fault signal (verified in
[notebook 01 section B5](notebooks/01_eda.ipynb)).

I picked **Streamlit + Power BI** instead of one or the other.
*Why:* they serve different audiences. Streamlit is the live
interactive demo (recruiters click around, screen-share friendly);
Power BI is the stakeholder report you'd actually share in a
maintenance ops review. Same data, complementary presentations.

I picked **PyTorch CUDA 12.6 wheel** locally and **CPU wheel**
on Streamlit Cloud. *Why:* GPU for training (saves ~10× over CPU
for the LSTM + Transformer); CPU for inference where torch
falls back automatically via the existing
`device = "cuda" if torch.cuda.is_available() else "cpu"` pattern.

---

## 3. Challenges I faced

**"Feature names should match those that were passed during fit"**
on Streamlit Cloud for FD002 / FD004. The dashboard's `apply_scaler`
was passing an extra `regime` column the saved StandardScaler had
never seen. Fix: add `'regime'` to the exclude set in
`get_all_feature_columns`. *Lesson:* any preprocessing step that
*adds* a column at training time needs that column in the
dashboard's exclude set, and local smoke tests need to actually run
the dashboard pipeline (not a notebook copy of it) to catch it.

**Streamlit Cloud Python 3.13 → shap → numba → llvmlite → no
compatible wheel.** First-time deploy died on the import. Fix: set
the Cloud app's Python version to 3.12 in the Advanced settings
panel; pin `statsmodels` and other heavy deps explicitly. *Lesson:*
heavy Python ML stacks lag Python releases by 6-12 months. Stay one
Python minor version behind the latest on managed hosting.

**Per-engine min-max threshold normalisation flagging 99% of cycles.**
The dashboard normalises each model's anomaly score to [0, 1] *per
engine* so the threshold slider behaves uniformly. On strongly-
degrading engines the IF scores cluster near the high end, and the
default 0.5 slider value caught almost everything. Fix: bumped the
default slider to 0.85 and added a "Computed across N test cycles"
caption so users know what they're looking at. *Lesson:* per-row
normalisation interacts badly with global thresholds; either
threshold on raw scores or normalise globally.

**LSTM training plateau on FD001.** The LSTM AE's reconstruction
loss barely moved (0.29 → 0.26 over 80 epochs) while the feedforward
AE went 0.19 → 0.13. *Diagnosis:* the LSTM's hidden state was 16-dim,
the bottleneck was 8-dim, and the decoder was repeating the
bottleneck as the input across all T steps — a known-weak design for
sequence reconstruction. *Fix:* replaced with the Transformer AE
(notebook 03), where attention + per-timestep bottleneck got
reconstruction loss to 0.035. *Lesson:* an LSTM AE's decoder design
matters more than I'd internalised. Either use teacher forcing or
use attention.

**Sensor 6 missed an interpretation paragraph.** When I first ran
the global SHAP analysis on FD001 the top sensor was `sensor_6`
(bypass-duct total pressure) — but my `INTERPRETATION` dict in
`src/sensor_descriptions.py` only covered the obvious HPC sensors.
*Fix:* added a paragraph for `sensor_6` and re-ran. *Lesson:* the
sensors you expect to be informative aren't always the ones the
model picks. Cover them all upfront.

**The "baseline vs engineered" comparison didn't go my way.**
[Notebook 02 C2](notebooks/02_feature_engineering.ipynb) showed
that the IF on 15 raw sensors *outperforms* the IF on 184
engineered features (F1 0.818 vs 0.777). Surprising but real. The
honest fix is to write up *why* — tree models are scale-invariant,
correlation hurts the feature scoring, curse of dimensionality at
this training-set size — and to note where engineered features
still earn their place (autoencoders, SHAP narrative,
multi-regime). *Lesson:* run the experiment that could falsify
your hypothesis, and report the result.

---

## 4. What I learned about

### Jet engines / domain knowledge
- HPC (high-pressure compressor) degradation propagates downstream
  through the gas path — bypass-duct pressure and LPT outlet
  temperature pick it up indirectly before the HPC sensors
  themselves show clear drift. *That's why the SHAP top-sensor on
  FD001 sits in the Fan section, not the HPC.*
- "Corrected" speeds (NRf, NRc) are operating-condition-normalised;
  *physical* speeds (Nf, Nc) are the raw RPM. On FD001 (single
  regime) the corrected speeds carry the signal; on FD004 (six
  regimes with per-regime normalisation upstream), the physical
  speeds win because the operating-condition adjustment is already
  removed.
- The C-MAPSS sensor mapping (NASA's PHM08 paper) turns
  every SHAP feature name into a real engine quantity. *This single
  reference table changed every interview answer I gave from
  jargon to narrative.*

### ML for time-series anomaly detection
- Tree-based unsupervised models (Isolation Forest, One-Class SVM)
  are a *very* strong baseline on tabular sensor data — hard to
  beat with neural methods unless the data is large and richly
  multi-dimensional.
- Sequence models matter when **per-cycle marginal distributions
  lose meaning** (multi-regime data) or when the failure pattern is
  *only* visible as a sequential progression. On simple subsets
  with monotonic degradation they don't add much value.
- **AUC-PR is the threshold-free metric to lead with on imbalanced
  data**, not F1 (which depends on a threshold you pick) and not
  AUC-ROC (which doesn't properly weight the rare class).
- **SHAP is model-specific.** TreeExplainer for trees, KernelExplainer
  for SVMs, DeepExplainer for NNs — each gives a different
  attribution and they're not directly comparable. Cross-model SHAP
  agreement on top features is a stronger anomaly signal than
  majority voting on predictions.
- **Multiple SHAP views answer different questions.** Bar plots rank;
  beeswarms reveal *spread* and direction per feature; dependence
  plots expose *threshold effects* and auto-detected interactions.
  Recruiters who do ML themselves notice when only one view is
  shown — that flag costs nothing to fix.
- **Cross-check explainability with a model-agnostic method.**
  Permutation importance shuffles each feature and measures the F1
  drop — different mechanism from SHAP entirely. When both rankings
  agree on the top sensors, that's a credibility argument; when they
  disagree (a feature high on SHAP but low on permutation), the gap
  usually reveals a correlated proxy carrying the signal.
- **Feature engineering doesn't always help.** See notebook 02 C2.
  Don't add a feature unless you can show empirically that it
  improves the metric.

### Engineering / shipping ML
- **Defensive imports** for optional ML dependencies (shap, LSTM,
  Transformer): wrap the import in try/except and surface a clear
  user-facing error rather than crashing the whole dashboard.
- **Per-subset caching** in Streamlit: every cached function takes
  the subset as a key argument, so flipping subsets is fast after
  the first load.
- **Deploying torch + shap on a constrained Cloud tier** means
  matching the Python version to what has wheels, accepting the
  ~5-10s import cost, and budgeting memory carefully (each model
  ~5-200 MB on disk; the IF pickle is the largest).
- **Notebook tests** (extract code cells → run headless with Agg
  backend) catch ~80% of regressions a CI pipeline would.
- **Composite-key joins in Power BI** need a calculated
  `EngineKey = subset & "::" & unit_id` column on both fact and dim
  tables. Native composite-key relationships are unsupported.

---

## 5. Tools used and why

| Tool | Purpose | Why this one | Notes |
|---|---|---|---|
| **scikit-learn** | Isolation Forest, One-Class SVM, KMeans, StandardScaler | Battle-tested, deterministic with `random_state`, fast `TreeExplainer` for SHAP | Default contamination=0.05 worked; tuned via F1-optimal threshold post-hoc |
| **PyTorch** (CUDA 12.6 local, CPU on Cloud) | Feedforward AE, LSTM AE, Transformer AE | Flexibility for custom architectures; `.to(device)` makes GPU/CPU portable | Pin wheel index URL for CUDA build |
| **SHAP** | Global + per-cycle attribution for IF; beeswarm + dependence plots | `TreeExplainer` is exact and ~100× faster than KernelExplainer; beeswarm/dependence are the canonical views recruiters expect | Negate against `score_samples` so positive = pushes toward anomaly |
| **scikit-learn `permutation_importance`** | Model-agnostic cross-check on the SHAP ranking | Different mechanism (shuffle a feature, measure F1 drop) gives an independent ranking — agreement = credibility, disagreement = informative | Wrap the IF as a `BaseEstimator + ClassifierMixin` so the F1 scorer accepts it |
| **Streamlit** | Live interactive dashboard | Single-file deploy, simple state model, free Cloud hosting | Cache by subset; defensive imports for optional deps |
| **Power BI Desktop** | Stakeholder-style executive dashboard | Industry-standard BI tool for non-technical audiences; cleaner KPI/slicer/drillthrough UX than Plotly | Build guide + committed CSVs + screenshots in `bi/` |
| **pytest** | Unit testing | Fast, simple, all 36 tests run in ~10 s | Includes notebook tests for the analytics work |
| **pandas / NumPy / SciPy / statsmodels** | Data manipulation, statistical tests | Standard scientific stack | `statsmodels` for ADF stationarity tests in notebook 01 |
| **Plotly / Matplotlib / Seaborn** | Visualisation | Plotly for dashboard; Matplotlib + Seaborn for notebook plots | Plotly's hover-rich UI works in Streamlit; static for committed plots |

---

## 6. What I'd do differently next time

1. **Start with the cross-subset experiment, not FD001 alone.** Running
   FD004 alongside FD001 from the start would have shaped the model
   selection differently (sequence models look much better when their
   competition isn't all running on the easy subset).

2. **Pin Python version explicitly from day one** — adding `runtime.txt`
   or a Cloud Python pin would have avoided the shap/numba/llvmlite
   wheel-resolution hour.

3. **Run the baseline-vs-engineered comparison first**, before pouring
   time into the feature engineering pipeline. Notebook 02 section C2
   would have changed the order of investment.

4. **Skip the LSTM AE** and go straight to the Transformer. The LSTM's
   reconstruction-loss plateau wasn't surprising in hindsight — the
   repeat-bottleneck decoder is a known-weak design — and the LSTM's
   only narrative role is to make the Transformer's improvement
   look bigger.

5. **Invest in domain knowledge earlier.** The sensor → physical-
   quantity mapping (notebook 07, `src/sensor_descriptions.py`) was
   added late but changed every part of my interview narrative. Doing
   that on day one would have shaped the feature engineering and the
   SHAP analysis from the start.

---

If you want to talk through any of these in more detail, I'm happy
to walk through specific code paths during the conversation.
[GitHub repo](https://github.com/Anjanamb/sensor-anomaly-detection)

---

## 8. The v2 refine (2026-08-11): why less, and what I kept

By early August the v1 repo was doing too much. Five detectors, four
subsets, 184 features, a Streamlit dashboard, seven notebooks totalling
~11k lines. It read like a portfolio piece rather than a project someone
could learn from. So I refined it in place, tagged the pre-refine state
as `v1.0.0`, and rebuilt the working tree around a much narrower goal.

### The five constraints

I wrote them down before touching any code:

1. FD004 only. It is the hardest of the four subsets. If the approach
   works there, it generalises.
2. Isolation Forest as the primary detector. DBSCAN as a companion
   specifically to get the two-methods-agreeing story and the density-based
   plots. Skip One-Class SVM and every autoencoder.
3. Every engineered feature justified with a hypothesis, math, and a
   validation plot. Anything I could not justify in that template was cut.
4. Notebook-first workflow. Thin `src/` modules, heavy notebooks. The
   notebook itself is the tutorial; nothing important should require
   jumping to a file.
5. No Streamlit. The dashboard added surface area without adding teaching
   value.

### What survived

- The notebook-first philosophy from v1, but sharpened. The v2 notebooks
  each open with a "What you will learn" list and close with a "Takeaways
  for the next notebook" bridge so the flow is continuous.
- Per-regime normalisation as the first non-obvious step. In v1 I found
  it late; in v2 it is the first thing notebook 02 does.
- Lead-time as the headline metric. In v1 I reported F1 because everyone
  reports F1. In v2 I opened notebook 05 with why F1 is actively wrong for
  this problem and why lead time matches the operational question.

### What was cut

- FD001, FD002, FD003 (and their trained models and RUL files).
- One-Class SVM, feedforward autoencoder, LSTM autoencoder, Transformer
  autoencoder, and every supporting notebook.
- SHAP explanations and the whole `explainability.py` module. Isolation
  Forest scores are already interpretable enough at this scope.
- Streamlit app, Power BI dashboard, live demo URL. The notebooks with
  baked outputs are the interface now.
- 164 of the 184 features. Kept 20 (8 EWMA, 8 deviation, 4 slope) on the
  8 sensors that pass the early-vs-late shift test.

### Two gotchas I did not see coming

1. **Dead sensors are per-regime on FD004, not global.** On FD001 (single
   regime) a sensor is dead if its std is near zero. On FD004 the same
   sensor may vary substantially between regimes while being constant
   within each. My first-pass global-std threshold in notebook 02 found
   zero dead sensors when there are actually four (s1, s5, s18, s19).
   Switched to "max std across regimes is below the threshold" and the
   right sensors dropped out.

2. **DBSCAN is confused by feature warmup.** My rolling primitives
   (EWMA span 15, slope window 15, deviation baseline 30 cycles) have a
   warmup phase where the values are not on the same manifold as the rest
   of the engine's life. Isolation Forest handled this because those
   warmup cycles were in its training set and it learned them as normal.
   DBSCAN treated them as low-density outliers and labelled them as
   anomalies. First-pass result: 69 % of DBSCAN noise was in cycles 1-30,
   the wrong direction (noise should be *late* in life, not early). Fix:
   drop the first 30 cycles per engine before fitting DBSCAN. After the
   fix, median RUL of noise is 12 cycles vs 109 for clustered.

Both of these are the sort of thing you only find by running the pipeline
end-to-end and looking at the outputs. The v1 architecture masked them
because it had so many moving parts that anomalies in any single detector
got averaged out by the ensemble comparison.

### What v2 actually shows

The story landed cleanly:

- Isolation Forest catches all 249 training engines with a median lead
  time of 210 cycles.
- DBSCAN catches 90/249, mostly late (median lead 13 cycles), but with
  much higher per-flag precision.
- Every DBSCAN-flagged cycle was also IF-flagged. Zero disagreement in
  the direction "DBSCAN said yes, IF said no". IF flags many cycles DBSCAN
  does not, which is IF being more sensitive rather than wrong.

Two very different detection mechanisms confirming each other on the same
cycles is a much better answer than a single F1 leaderboard, and it took
about a fifth of the code to get there.

### The v1.0.0 tag

Everything from before the refine lives at
[`v1.0.0`](https://github.com/Anjanamb/sensor-anomaly-detection/releases/tag/v1.0.0).
Nothing was lost; the tag is browseable on GitHub and any file recoverable
with `git checkout v1.0.0 -- <path>`. If you're reading this JOURNEY as a
recruiter you might want to skim v1.0.0 first (the five-detector version
with the F1 leaderboard is the more portfolio-style artefact) and then
come back to v2 to see how I would actually approach the problem now.
