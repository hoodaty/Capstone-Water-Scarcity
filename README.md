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

## Generating Plots

To compare the results and generate figures:

### Per-model Comparison
Compare calibration strategies side by side for each model (default):
```bash
uv run plots.py --results-dir results/ --group-by model --output figures/modelwise
```

### All Experiments Comparison
Include all experiments in one set of plots:
```bash
uv run plots.py --results-dir results/ --group-by all --output figures/comparison
```
