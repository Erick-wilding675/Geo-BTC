# `results/`

Tabular outputs of the pipeline — CSV artefacts referenced directly
by the paper and by Phase 2 qualitative analysis.

| File | Produced by |
| ---- | ----------- |
| `grid_search_results.csv` | `scripts/run_grid_search.py` |
| `analysis_residuals_2015.csv` | `scripts/run_residual_analysis.py` |
| `investigation_periods.csv` | `scripts/run_residual_analysis.py` |

`analysis_residuals_2015.csv` columns:

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

`investigation_periods.csv` columns:

| Column | Description |
| ------ | ----------- |
| `Event_Date` | Day the outlier occurred |
| `Actual_Price_USD` | BTC/USD on `Event_Date` |
| `Error_Value_USD` | Absolute residual |
| `Type` | `"Peak"` (actual > predicted) or `"Drop"` |
| `Start_Date` | `Event_Date − 3 days` |
| `End_Date` | `Event_Date + 7 days` |
