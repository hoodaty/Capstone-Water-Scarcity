import argparse
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models.registry import get_model_spec, list_model_ids
from src.utils.model import (
    # split_dataset,
    get_station_stats,
    summarize_metrics,
    standardize_prediction_intervals,
    standardize_values,
    winkler_score,
    wis_score,
)

# --- CONFIGURATION ---
BASE_DIR_CONF = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR_CONF, "data", "input", "dataset_baseline.csv")
TIME_VALIDATION = "2000-01-01 00:00:00"
TRAIN_STATION_FRACTION = 0.75
NUMBER_OF_WEEKS = 4
ALPHA = 0.1
USE_EWMA = False

METEO_LAG_WINDOWS = []
METEO_LAG_COLUMNS = [
    "precipitations", "temperatures", "soil_moisture", "evaporation",
    "precipitation_region", "temperature_region", "soil_moisture_region", "evaporation_region",
    "precipitation_zone", "temperature_zone", "soil_moisture_zone", "evaporation_zone",
    "precipitation_sector", "temperature_sector", "soil_moisture_sector", "evaporation_sector",
    "precipitation_sub_sector", "temperature_sub_sector", "soil_moisture_sub_sector", "evaporation_sub_sector",
]

TARGET_COLS = [f"water_flow_week{i+1}" for i in range(NUMBER_OF_WEEKS)]
FORBIDDEN_COLS = ["water_flow_lag_1w", "water_flow_lag_2w"]
FEATURE_DROP_COLS = ["station_code"] + TARGET_COLS + FORBIDDEN_COLS

def load_dataset(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    df = pd.read_csv(filepath, index_col=0)
    return df

def build_features(df: pd.DataFrame, reference_columns: list[str] | None = None) -> pd.DataFrame:
    """Build features with consistent dummy encoding."""
    if df.empty:
        if reference_columns:
            return pd.DataFrame(columns=reference_columns)
        return pd.DataFrame()

    X = df.drop(columns=FEATURE_DROP_COLS, errors="ignore")
    # One-hot encode potential categorical columns if any
    X = pd.get_dummies(X, drop_first=True)
    
    if reference_columns is not None:
        # Add missing columns (filled with 0)
        for col in reference_columns:
            if col not in X.columns:
                X[col] = 0
        # Remove extra columns not in reference
        X = X[[col for col in reference_columns if col in X.columns]]
        # Ensure same order
        X = X.reindex(columns=reference_columns, fill_value=0)
    
    return X

def add_meteo_lags(df: pd.DataFrame, columns: list[str], windows: list[int], use_ewm: bool = False) -> pd.DataFrame:
    df = df.copy()
    if df.index.name is None:
        df.index.name = "ObsDate"
    df = df.sort_values(["station_code", df.index.name])
    available_cols = [c for c in columns if c in df.columns]
    
    new_features = []
    for col in available_cols:
        agg = "sum" if "precip" in col.lower() or col == "tp" else "mean"
        for w in windows:
            if use_ewm:
                feature_name = f"{col}_ewm_{w}w_mean"
                series = df.groupby("station_code")[col].transform(
                    lambda s: s.ewm(span=w, min_periods=1).mean()
                )
            else:
                feature_name = f"{col}_roll_{w}w_{agg}"
                series = df.groupby("station_code")[col].transform(
                    lambda s: s.rolling(w, min_periods=1).agg(agg)
                )
            series.name = feature_name
            new_features.append(series)
            
    if new_features:
        df = pd.concat([df] + new_features, axis=1)
        
    return df

def compute_per_station_metrics_robust(
    y_true_std: np.ndarray,
    y_pred_std: np.ndarray,
    stations: np.ndarray,
    y_pred_lower_std: np.ndarray,
    y_pred_upper_std: np.ndarray,
    alpha: float,
) -> pd.DataFrame:
    """Robust computation of station metrics with sigma clamping."""
    station_list = np.unique(stations)
    records = []

    for s in station_list:
        idx = stations == s
        if np.sum(idx) == 0:
            continue
            
        y_true_s = y_true_std[idx]
        y_pred_s = y_pred_std[idx]
        y_lower_s = y_pred_lower_std[idx]
        y_upper_s = y_pred_upper_std[idx]

        rmse_s = sqrt(mean_squared_error(y_true_s, y_pred_s))
        mae_s = mean_absolute_error(y_true_s, y_pred_s)

        # Robust Sigma Estimation
        sigma_s = (y_upper_s - y_lower_s) / 3.29
        sigma_s = np.maximum(sigma_s, 0.5) # Clamp sigma

        log_term = np.log(sigma_s)
        resid_term = np.abs(y_true_s - y_pred_s) / (2 * sigma_s)
        nll_s = np.mean(log_term + resid_term)

        coverage_s = np.mean((y_true_s >= y_lower_s) & (y_true_s <= y_upper_s))
        interval_size_s = np.mean(y_upper_s - y_lower_s)
        winkler_s = np.mean(winkler_score(y_true_s, y_lower_s, y_upper_s, alpha))
        wis_s = np.mean(wis_score(y_true_s, y_lower_s, y_upper_s, y_pred_s, alpha))
        coverage_gap_s = coverage_s - (1 - alpha)

        records.append({
            "station_code": s,
            "scaled_rmse": rmse_s,
            "scaled_mae": mae_s,
            "coverage": coverage_s,
            "scaled_interval_size": interval_size_s,
            "log_likelihood": nll_s,
            "winkler": winkler_s,
            "wis": wis_s,
            "coverage_gap": coverage_gap_s,
        })

    return pd.DataFrame(records)

def analyze_collinearity(X, threshold=0.9):
    print("\n--- 1. Pre-Analysis: Collinearity Detection ---")
    X_sample = X.sample(n=min(5000, len(X)), random_state=42)
    corr_matrix = X_sample.corr(method='spearman').abs()
    distance_matrix = 1 - corr_matrix
    distance_matrix = distance_matrix.clip(lower=0)
    dist_array = distance_matrix.values.copy()
    dist_array = (dist_array + dist_array.T) / 2
    np.fill_diagonal(dist_array, 0)
    dist_linkage = squareform(dist_array)
    Z = hierarchy.linkage(dist_linkage, method='ward')
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FIG_DIR = os.path.join(BASE_DIR, "figures", "models")
    os.makedirs(FIG_DIR, exist_ok=True)
    
    plt.figure(figsize=(32, 8))
    _ = hierarchy.dendrogram(
        Z, labels=X.columns, leaf_rotation=90, leaf_font_size=8
    )
    plt.title("Hierarchical Clustering Dendrogram (Spearman Correlation)")
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, "collinearity_dendrogram.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Collinearity dendrogram saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Train quantile model with CQR calibration")
    parser.add_argument("--calib-temp", action="store_true", help="Enable calibration using the Temporal Train split.")
    parser.add_argument("--calib-stemp", action="store_true", help="Enable calibration using BOTH Temporal and Spatio-Temporal Train splits.")
    parser.add_argument(
        "--calib-spatio-only",
        action="store_true",
        help="Enable calibration using ONLY the Spatio-Temporal Train split.",
    )
    parser.add_argument("--name", type=str, default="", help="Custom name tag for the experiment (appended to timestamp)")
    parser.add_argument(
        "--model",
        type=str,
        default="lgbm_cqr",
        help=f"Model id. Options: {', '.join(list_model_ids())}",
    )
    args = parser.parse_args()

    if args.calib_spatio_only and (args.calib_temp or args.calib_stemp):
        raise ValueError("Choose one calibration mode: temp, mixed, or spatio-only.")

    # Logic: --calib-stemp implies mixed calibration (both)
    if args.calib_stemp:
        args.calib_temp = True

    calib_spatio_only = args.calib_spatio_only

    model_spec = get_model_spec(args.model)
    is_chronos_model = model_spec.id == "chronos_2"
    print(
        "Configuration: "
        f"Model={model_spec.id}, "
        f"Temporal Calib={args.calib_temp}, "
        f"SpatioTemporal Calib={args.calib_stemp}, "
        f"SpatioOnly Calib={calib_spatio_only}"
    )

    # --- 0. Setup Experiment Directory ---
    import datetime
    import json
    import re
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.name:
        clean_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', args.name)
        folder_name = clean_name
    else:
        folder_name = timestamp
        
    RESULTS_DIR = os.path.join(BASE_DIR_CONF, "results", folder_name)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Results will be saved to: {RESULTS_DIR}")
    
    MODEL_NAME = model_spec.display_name
    
    # Save Config
    config = {
        "experiment_name": args.name,
        "calib_temp": args.calib_temp,
        "calib_stemp": args.calib_stemp,
        "calib_spatio_only": calib_spatio_only,
        "alpha": ALPHA,
        "weeks": NUMBER_OF_WEEKS,
        "model": MODEL_NAME,
        "model_id": model_spec.id,
        "train_station_fraction": TRAIN_STATION_FRACTION,
        "time_validation": TIME_VALIDATION
    }
    # Initial save of config
    with open(os.path.join(RESULTS_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # --- 1. Loading Data ---
    PROCESSED_DIR = os.path.join(BASE_DIR_CONF, "data", "processed")
    if not os.path.exists(PROCESSED_DIR):
        raise FileNotFoundError(f"Processed data not found in {PROCESSED_DIR}. Run data.py first.")

    print("Loading pre-processed splits...")
    
    # Helper to load and preprocess a split
    def load_split(name):
        df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{name}.csv"), index_col=0)
        return add_meteo_lags(df, METEO_LAG_COLUMNS, METEO_LAG_WINDOWS, use_ewm=USE_EWMA)

    train_df = load_split("main_train")
    
    # Calibration Sets (Conditional Load)
    calib_temp_df = load_split("temporal_train") if args.calib_temp else pd.DataFrame()
    calib_spatio_enabled = args.calib_stemp or calib_spatio_only
    calib_spatio_df = load_split("spatiotemporal_train") if calib_spatio_enabled else pd.DataFrame()
    
    # Evaluation Sets (Always Load)
    eval_temp_df = load_split("temporal_test")
    eval_spatio_df = load_split("spatiotemporal_test")

    # Compute GLOBAL station stats (Using full raw input JUST for stats metadata)
    dataset_full = load_dataset(INPUT_FILE)
    global_stats_by_week = {
        target_col: get_station_stats(
            dataset_full[target_col].values, dataset_full["station_code"].values
        )
        for target_col in TARGET_COLS
    }

    print(f"Split sizes:")
    print(f"  Train: {len(train_df)}")
    print(f"  Calib Temporal: {len(calib_temp_df)} (Enabled: {args.calib_temp})")
    print(f"  Calib SpatioTemp: {len(calib_spatio_df)} (Enabled: {calib_spatio_enabled})")
    print(f"  Eval Temporal: {len(eval_temp_df)}")
    print(f"  Eval SpatioTemp: {len(eval_spatio_df)}")

    X_train = build_features(train_df)
    train_columns = X_train.columns.tolist()
    
    # Pre-process Calibration Features
    X_calib_temp = build_features(calib_temp_df, reference_columns=train_columns)
    X_calib_spatio = build_features(calib_spatio_df, reference_columns=train_columns)
    
    # Pre-process Evaluation Features
    X_eval_temp = build_features(eval_temp_df, reference_columns=train_columns)
    X_eval_spatio = build_features(eval_spatio_df, reference_columns=train_columns)
    
    stations_eval_temp = eval_temp_df["station_code"].values
    stations_eval_spatio = eval_spatio_df["station_code"].values
    
    MODEL_NAME = model_spec.display_name
    
    # --- 2. Training & Calibration ---
    models = {}
    calibrated_any = False
    
    for i, target_col in enumerate(TARGET_COLS):
        print(f"\nProcessing {target_col}...")
        
        # A. Training
        model = model_spec.cls(alpha=ALPHA, **model_spec.init_kwargs)
        if is_chronos_model:
            model.fit(train_df=train_df, target_col=target_col)
        else:
            model.fit(X_train.values, train_df[target_col].values)
        models[i] = model
        
        # B. Calibration
        # Construct X_calib and y_calib based on flags
        Xs_calib = []
        ys_calib = []
        
        if args.calib_temp:
            Xs_calib.append(calib_temp_df if is_chronos_model else X_calib_temp)
            ys_calib.append(calib_temp_df[target_col].values)
            
        if args.calib_stemp or calib_spatio_only:
            Xs_calib.append(calib_spatio_df if is_chronos_model else X_calib_spatio)
            ys_calib.append(calib_spatio_df[target_col].values)
            
        if Xs_calib:
            X_calib_final = pd.concat(Xs_calib)
            y_calib_final = np.concatenate(ys_calib)
            if is_chronos_model:
                model.calibrate(X_calib_final, y_calib_final, target_col=target_col)
            else:
                model.calibrate(X_calib_final.values, y_calib_final)
            calibrated_any = True
            print(f"  Calibrated on {len(X_calib_final)} samples. q_score={model.q_score:.3f}")
        else:
            print("  Skipping calibration.")

        # Save Model for this week
        if "moment" in args.model.lower():
            import torch
            save_path = os.path.join(RESULTS_DIR, f"model_week{i}.pt")
            # Save strictly the learned weights of the projection head
            # The frozen MOMENT backbone does not need to be saved
            torch.save(model.head.state_dict(), save_path)
        else:
            save_path = os.path.join(RESULTS_DIR, f"model_week{i}.joblib")
            joblib.dump(model, save_path)
        print(f"  Week {i} model saved to {save_path}")

    if calibrated_any:
        MODEL_NAME = model_spec.calibrated_name
        print(f"\nModel updated to: {MODEL_NAME} (Calibration Active)")

    # Update Config with final model name
    config["model"] = MODEL_NAME
    with open(os.path.join(RESULTS_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # --- 3. Evaluation ---
    print("\n--- Evaluation ---")
    
    all_raw_metrics = []
    all_summaries = []

    def evaluate_set(X, stations, dataset_source_name, indices, source_df):
        print(f"\nEvaluating {dataset_source_name}...")

        for i, target_col in enumerate(TARGET_COLS):
            # Use GLOBAL stats for stats reindexing (per target week)
            station_stats = (
                global_stats_by_week[target_col]
                .reindex(np.unique(stations))
                .fillna(1.0)
            )

            y_true = source_df[target_col].values[indices]
            
            # Predictions
            if is_chronos_model:
                pred_df = source_df.iloc[indices].copy()
                y_pred = models[i].predict(pred_df, quantiles="mean")
                y_quantiles = models[i].predict(pred_df, quantiles=[ALPHA / 2, 1 - ALPHA / 2])
            else:
                y_pred = models[i].predict(X.values, quantiles="mean")
                y_quantiles = models[i].predict(X.values, quantiles=[ALPHA/2, 1-ALPHA/2])
            y_lower = y_quantiles[:, 0]
            y_upper = y_quantiles[:, 1]
            
            # Standardize
            y_true_std = standardize_values(y_true, stations, station_stats)
            y_pred_std = standardize_values(y_pred, stations, station_stats)
            y_lower_std, y_upper_std = standardize_prediction_intervals(
                np.column_stack([y_lower, y_upper]), stations, station_stats
            )
            
            # Compute Raw Metrics (Per Station)
            metrics_df = compute_per_station_metrics_robust(
                y_true_std, y_pred_std, stations, y_lower_std, y_upper_std, ALPHA
            )
            
            # Add Context Columns
            metrics_df["dataset"] = dataset_source_name
            metrics_df["week"] = i
            metrics_df["target_col"] = target_col
            all_raw_metrics.append(metrics_df)
            
            # Compute Summary
            summary_df = summarize_metrics(
                metrics_df, MODEL_NAME, f"{dataset_source_name}_wk{i}", alpha=ALPHA
            )
            summary_df["week"] = i
            summary_df["dataset"] = dataset_source_name
            all_summaries.append(summary_df)
            
            print(summary_df.to_string(index=False))

    # Evaluate on Eval Splits
    evaluate_set(X_eval_temp, stations_eval_temp, "Eval_Temporal", np.arange(len(X_eval_temp)), eval_temp_df)
    evaluate_set(X_eval_spatio, stations_eval_spatio, "Eval_SpatioTemporal", np.arange(len(X_eval_spatio)), eval_spatio_df)

    # Compile and Save CSVs
    if all_raw_metrics:
        final_raw = pd.concat(all_raw_metrics, ignore_index=True)
        final_raw.to_csv(os.path.join(RESULTS_DIR, "metrics_per_station.csv"), index=False)
        print(f"\nSaved raw station metrics to {RESULTS_DIR}/metrics_per_station.csv")
    
    if all_summaries:
        final_summary = pd.concat(all_summaries, ignore_index=True)
        final_summary.to_csv(os.path.join(RESULTS_DIR, "metrics_summary.csv"), index=False)
        print(f"Saved summary metrics to {RESULTS_DIR}/metrics_summary.csv")

if __name__ == "__main__":
    main()
