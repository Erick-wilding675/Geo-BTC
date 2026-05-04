# `results/`

Tabular outputs of the pipeline — CSV artefacts referenced directly by
the paper.

## Phase 1 — quantitative

| File | Produced by |
| ---- | ----------- |
| `grid_search_results.csv` | `scripts/run_grid_search.py` |
| `analysis_residuals_2015.csv` | `scripts/run_residual_analysis.py` |
| `investigation_periods.csv` | `scripts/run_residual_analysis.py` |

## Phase 2 — causal inference

| File | Produced by |
| ---- | ----------- |
| `inference_table.csv` | `scripts/run_inference_merge.py` |
| `explaining_ratio.csv` | `scripts/run_explaining_ratio.py` |

### `analysis_residuals_2015.csv`

| Column | Description |
| ------ | ----------- |
| `Date` | Inference day |
| `Actual_Price` | Ground-truth BTC/USD |
| `Predicted_Price` | Model forecast |
| `Abs_Error` | `|actual − predicted|` |
| `Pct_Error` | `Abs_Error / Actual_Price × 100` |
| `Directional_Error` | `actual − predicted` |
| `Is_Outlier` | `True` if outside ±2σ directional band |
| `Outlier_Type` | `"Positive Spike"` / `"Negative Drop"` / `"Normal"` |

### `investigation_periods.csv`

| Column | Description |
| ------ | ----------- |
| `Event_Date` | Day the outlier occurred |
| `Actual_Price_USD` | BTC/USD on `Event_Date` |
| `Error_Value_USD` | Absolute residual |
| `Type` | `"Peak"` (actual > predicted) or `"Drop"` |
| `Start_Date` | `Event_Date − 3 days` |
| `End_Date` | `Event_Date + 7 days` |

### `inference_table.csv`

| Column | Description |
| ------ | ----------- |
| `Event_Date` | Inference day for the outlier |
| `Data_Window` | `Start_Date → End_Date` (string) |
| `Real_Price` | BTC/USD on `Event_Date` |
| `Absolute_Error` | LSTM absolute residual (USD) |
| `Fluctuation` | `Up` / `Down` from the qualitative database |
| `Category` | Event category (NaN if no matching window) |
| `Event_Macro` | Macro-level narrative |
| `Breakdown_Micro` | Proximate market mechanism |
| `Price_Movement` | Verbatim window-level price move |

### `explaining_ratio.csv`

| Column | Description |
| ------ | ----------- |
| `Event Category` | Aggregation key |
| `Frequence` | Number of explained outliers in the category |
| `Mean Absolute Error (MAE)` | Average residual (USD) for the category |
| `Explaining Ratio (ER)` | Share of total identified outliers (%) |

Summing the `Explaining Ratio (ER)` column reproduces the global
Explaining Ratio.
