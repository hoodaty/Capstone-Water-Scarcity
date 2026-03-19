# Capstone Project - Streamflow Prediction

## Setup

1.  Install `uv`.
2.  Clone and `cd` into repo.
3.  Run `uv sync`.

## Running Models

To run the models with different calibration strategies:

### No Calibration
```bash
for m in lgbm_cqr qrf_cqr catboost_cqr; do
  uv run run.py --model $m --name ${m}_none
done
```

### Temporal Calibration
```bash
for m in lgbm_cqr qrf_cqr catboost_cqr; do
  uv run run.py --model $m --calib-temp --name ${m}_calib_temp
done
```

### Mixed (Temporal + Spatio-Temporal) Calibration
```bash
for m in lgbm_cqr qrf_cqr catboost_cqr; do
  uv run run.py --model $m --calib-stemp --name ${m}_calib_mixed
done
```

## Interpretability & Error Analysis

Once a model is trained, use these scripts to analyze its behavior on the test set.

### 1. Advanced Interpretability (Grouped SHAP + ALE)
Analyze which feature clusters drive the model using correlation-aware clustering.

*   **Uncertainty Mode** (Why is the model unsure?):
    ```bash
    uv run interp.py --results-dir results/lgbm_cqr_calib_mixed/ --week 0 --mode width
    ```
*   **Median Mode** (How is streamflow magnitude affected?):
    ```bash
    uv run interp.py --results-dir results/lgbm_cqr_calib_mixed/ --week 0 --mode median
    ```

### 2. Worst-Station Analysis
Identify and plot the top $K$ stations with the highest RMSE on unseen (spatio-temporal) data to debug spatial generalization.
```bash
uv run worst-analysis.py --results-dir results/lgbm_cqr_calib_mixed/ --week 0 --top-k 3
```

## Generating Global Plots

To compare the results and generate figures across all experiments:

### Per-model Comparison
```bash
uv run plots.py --results-dir results/ --group-by model --output figures/modelwise
```

### All Experiments Comparison
```bash
uv run plots.py --results-dir results/ --group-by all --output figures/comparison
```
