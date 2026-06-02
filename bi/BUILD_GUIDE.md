# Power BI executive dashboard — build guide

This guide walks through assembling `bi/dashboard.pbix` from the CSVs in `bi/data/`. The Streamlit app is the live, interactive demo; this is the stakeholder-style report you'd actually share in a maintenance ops review — pre-aggregated, paginated, with KPIs up top.

**Prerequisites**

- Power BI Desktop (free, Windows). Download from <https://aka.ms/pbidesktopstore>.
- `bi/data/*.csv` regenerated and current. **`cycles.csv` is gitignored** (it's ~48 MB across all four subsets), so it must be built locally before the .pbix can be assembled:

  ```bash
  python bi/build_data.py            # all four subsets, ~12 min
  python bi/build_data.py --skip-shap  # ~30 s if feature_importance.csv already exists
  ```

  The other four CSVs (`engines.csv`, `model_comparison.csv`, `feature_importance.csv`, `sensors.csv`) are committed and kept in sync via the script. Re-run the script after any model retraining.

Estimated build time: **45 minutes** the first time, ~10 minutes for subsequent rebuilds once muscle memory kicks in.

---

## 1 — Import the CSVs

1. Open Power BI Desktop → **Home → Get Data → Text/CSV**.
2. Import the five files in this order (lets Power BI guess types correctly):
   1. `sensors.csv` (smallest, sets up the dimension first)
   2. `engines.csv`
   3. `model_comparison.csv`
   4. `feature_importance.csv`
   5. `cycles.csv` (largest, ~10 MB total across all subsets)
3. For each import: click **Transform Data**, verify column types (text for `subset`, `model`, `subsystem`, `symbol`, `rul_bucket`, `fault_severity_bucket`; integer for `unit_id`, `cycle`, `rank`, `max_cycle`, `anomaly`; decimal for `rul`, `anomaly_score_iso`, `F1`, `AUC_ROC`, `AUC_PR`, `Precision`, `Recall`, `total_abs_shap`, raw sensor columns).
4. **Apply** → tables land in the Fields pane on the right.

---

## 2 — Build the data model (star schema)

Switch to the **Model** view (left sidebar, third icon down). Drag to create these relationships:

```
        ┌──────────┐
        │ sensors  │ (sensor_id)
        └────┬─────┘
             │ 1
             │
             │ *
┌──────────────────────┐         ┌─────────────────────┐
│ feature_importance   │         │  model_comparison   │
│  (sensor_id, subset, │         │  (subset, model,    │
│   rul_bucket, ...)   │         │   F1, AUC_PR, ...)  │
└──────────────────────┘         └──────────┬──────────┘
                                            │
                                            │ subset filter only
                                            │ (no FK)
                ┌─────────────┐    1   *   ┌─┴────────────┐
                │  engines    │◄───────────│   cycles     │
                │ (subset,    │            │ (subset,     │
                │  unit_id)   │            │  unit_id,    │
                └─────────────┘            │  cycle, ...) │
                                           └──────────────┘
```

Relationships to create (drag from the second-listed table's column to the first):

| From | To | Cardinality | Cross-filter |
|---|---|---|---|
| `feature_importance[sensor_id]` | `sensors[sensor_id]` | many-to-one | single (sensors → fi) |
| `cycles[(subset, unit_id)]` (composite) | `engines[(subset, unit_id)]` | many-to-one | single |

Power BI doesn't natively support composite key relationships out of the box, so the **simplest** workaround is to create a calculated key column on both `cycles` and `engines`:

```dax
EngineKey = cycles[subset] & "::" & FORMAT(cycles[unit_id], "000")
```

Add the same column on `engines`, then relate `cycles[EngineKey]` to `engines[EngineKey]`.

For `subset` itself, leave it as a free-floating filter — pages use a slicer on whichever table is most visible.

---

## 3 — Add DAX measures

Right-click the `model_comparison` table → **New measure**. Paste each block and save.

```dax
Selected F1 =
    SELECTEDVALUE ( model_comparison[F1], BLANK () )

Selected AUC-PR =
    SELECTEDVALUE ( model_comparison[AUC_PR], BLANK () )

Top Model F1 =
    MAXX ( ALLSELECTED ( model_comparison ), model_comparison[F1] )

Top Model Name =
    CALCULATE (
        SELECTEDVALUE ( model_comparison[model] ),
        TOPN ( 1, ALLSELECTED ( model_comparison ), model_comparison[F1], DESC )
    )
```

On `cycles`:

```dax
Anomaly Rate =
    DIVIDE (
        CALCULATE ( SUM ( cycles[anomaly] ) ),
        COUNTROWS ( cycles )
    )

% Captured Failures =
    -- Of all true anomalies, the fraction the IF flags above 0.5 normalised score
    VAR _flagged =
        CALCULATE (
            COUNTROWS ( cycles ),
            cycles[anomaly] = 1,
            cycles[anomaly_score_iso] > 0
        )
    VAR _total =
        CALCULATE ( COUNTROWS ( cycles ), cycles[anomaly] = 1 )
    RETURN DIVIDE ( _flagged, _total )

Engine Count = DISTINCTCOUNT ( cycles[EngineKey] )

Cycles Observed = COUNTROWS ( cycles )
```

On `engines`:

```dax
Avg Engine Lifetime = AVERAGE ( engines[max_cycle] )

High Severity Engines =
    CALCULATE (
        DISTINCTCOUNT ( engines[EngineKey] ),
        engines[fault_severity_bucket] IN { "high (36-60)", "very high (>60)" }
    )
```

On `feature_importance`:

```dax
Top Subsystem by SHAP =
    CALCULATE (
        SELECTEDVALUE ( feature_importance[subsystem] ),
        TOPN ( 1,
            SUMMARIZE ( ALLSELECTED ( feature_importance ),
                        feature_importance[subsystem],
                        "S", SUM ( feature_importance[total_abs_shap] ) ),
            [S], DESC
        )
    )
```

---

## 4 — Page layout

### Page 1 — Executive Summary

Layout (16:9):

```
┌──────────────────────────────────────────────────────────────────┐
│ TITLE: "Industrial Sensor Anomaly Detection — Executive Summary" │
├───────────────┬───────────────┬───────────────┬──────────────────┤
│ Engine Count  │ Anomaly Rate  │ Top Model F1  │ Top Model Name   │
│  (KPI card)   │  (KPI card)   │  (KPI card)   │  (text + icon)   │
├───────────────┴───────────────┴───────────────┴──────────────────┤
│ Model comparison bar chart                                       │
│ X-axis: model · Y-axis: F1 · Colour by subset                    │
├──────────────────────────────────────────────────────────────────┤
│ Subset slicer (FD001 / FD002 / FD003 / FD004)  [horizontal pill] │
└──────────────────────────────────────────────────────────────────┘
```

KPI cards: use the Card visual; set the data label to the measure (e.g. `Anomaly Rate`); format the colour to white text on dark.

### Page 2 — Engine Fleet

```
┌──────────────────────────────────────────────────────────────────┐
│ Slicer: subset                                                   │
├──────────────────────────────────────────────────────────────────┤
│ Scatter plot                                                     │
│ X = max_cycle (engine lifetime)                                  │
│ Y = anomaly_cycle_count                                          │
│ Colour by fault_severity_bucket                                  │
│ Size = constant                                                  │
│ Tooltip: unit_id, subset                                         │
├──────────────────────────────────────────────────────────────────┤
│ Table: top-10 most-anomalous engines                             │
│ Columns: unit_id · max_cycle · anomaly_cycle_count · severity    │
│ Sort: anomaly_cycle_count desc                                   │
└──────────────────────────────────────────────────────────────────┘
```

Right-click an engine bubble → **Drill through → Single Engine** (set up in Page 4).

### Page 3 — Sensor Diagnostics

```
┌──────────────────────────────────────────────────────────────────┐
│ Slicer: subset                  Slicer: rul_bucket (multi-select)│
├────────────────────────────┬─────────────────────────────────────┤
│ Top-10 sensors bar         │ Subsystem treemap                   │
│ Y: symbol (Ps30, T30, ...) │ Size: sum(total_abs_shap)           │
│ X: total_abs_shap          │ Colour by subsystem                 │
│ Colour by subsystem        │                                     │
├────────────────────────────┴─────────────────────────────────────┤
│ Card: "Top Subsystem by SHAP"  (measure)                         │
└──────────────────────────────────────────────────────────────────┘
```

### Page 4 — Single Engine Drill-Through

```
┌──────────────────────────────────────────────────────────────────┐
│ Drill-through filter: EngineKey (added at the page level)        │
├──────────────────────────────────────────────────────────────────┤
│ Line chart: anomaly_score_iso vs cycle, colour by anomaly label  │
├──────────────────────────────────────────────────────────────────┤
│ Small multiples: sensor_2, sensor_3, sensor_4, sensor_11 over cycle │
└──────────────────────────────────────────────────────────────────┘
```

To enable drill-through: select the page → **Visualizations pane → bottom → Drill-through filters → Add field `EngineKey`**. Pages 2 + 3 will now show "right-click → drill through" on any visual with `EngineKey` in scope.

---

## 5 — Theme

**Home → Themes → Browse for themes →** import this JSON (save as `bi/theme.json` if you want it tracked):

```json
{
  "name": "Sensor Anomaly Detection",
  "dataColors": [
    "#ef5350", "#4fc3f7", "#ffca28", "#66bb6a",
    "#ab47bc", "#26a69a", "#ff7043", "#bc6c25"
  ],
  "background": "#0e1117",
  "foreground": "#ffffff",
  "tableAccent": "#4fc3f7"
}
```

These colours match the Streamlit panel's subsystem palette (HPC = red, Fan = light blue, LPT = amber, etc.) so a recruiter viewing both side-by-side sees one consistent visual language.

---

## 6 — Save and screenshot

1. **File → Save As → `bi/dashboard.pbix`**.
2. On each page, hit **PrintScreen** or use the Windows **Snipping Tool** → save as `bi/screenshots/page_1.png` … `page_4.png` (1920×1080 ideal).
3. `git add bi/dashboard.pbix bi/screenshots/*.png && git commit -m "feat(bi): assembled Power BI dashboard"`.

Once those land, the README's "Power BI executive dashboard" section will display the screenshots and the .pbix is downloadable from the repo for recruiters who want to open it.

---

## 7 — Regenerating data

Whenever models change (new training run, swapped subset, new sensor descriptions):

```bash
python bi/build_data.py            # all four subsets, ~12 min
python bi/build_data.py --only FD001 --skip-shap   # fast iteration
```

The CSVs are regenerated in place. Re-open `dashboard.pbix` → **Home → Refresh** and visuals pick up the new data automatically.

---

## Reuse notes

- `cycles.csv` is keyed on `(subset, unit_id, cycle)` — composite. Power BI handles this via the `EngineKey` calculated column from §2.
- `feature_importance.csv` has `rul_bucket = "overall"` rows in addition to the three lifecycle buckets — useful for the "overall" view on Page 3 without union-ing.
- Anomaly score column is the **un-normalised IF score** (`anomaly_score_iso = -model.score_samples(X)`). If you want a 0-1 normalised version for Page 4, create a measure: `Normalised Score = ([anomaly_score_iso] - MIN([anomaly_score_iso])) / (MAX([anomaly_score_iso]) - MIN([anomaly_score_iso]))`.
