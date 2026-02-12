import os
from math import sqrt

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from src.utils.model import (
    split_dataset,
    # compute_per_station_metrics, # using a more robust version below
    get_station_stats,
    summarize_metrics,
    standardize_prediction_intervals,
    standardize_values,
)

# --- CONFIGURATION ---
BASE_DIR_CONF = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR_CONF, "data", "input", "dataset_baseline.csv")
TIME_VALIDATION = "2000-01-01 00:00:00"
TRAIN_STATION_FRACTION = 0.75
NUMBER_OF_WEEKS = 4
ALPHA = 0.1
CALIBRATE_INTERVALS = True
# If True, we use Conformalized Quantile Regression (CQR) style calibration
# If False, we depend on raw model quantiles

# Toggle for mixed calibration strategy:
# True  = calibrate on a held-out portion of test_spatio_temporal (unseen stations)
# False = calibrate on test_temporal only (seen stations, future time)
USE_MIXED_CALIBRATION = True
MIXED_CALIB_FRACTION = 0.2  # fraction of test_spatio_temporal used for calibration
USE_EWMA = False

METEO_LAG_WINDOWS = [1, 2, 4, 8, 12, 16]
METEO_LAG_COLUMNS = [
    "precipitations", "temperatures", "soil_moisture", "evaporation",
    "precipitation_region", "temperature_region", "soil_moisture_region", "evaporation_region",
    "precipitation_zone", "temperature_zone", "soil_moisture_zone", "evaporation_zone",
    "precipitation_sector", "temperature_sector", "soil_moisture_sector", "evaporation_sector",
    "precipitation_sub_sector", "temperature_sub_sector", "soil_moisture_sub_sector", "evaporation_sub_sector",
]

TARGET_COLS = [f"water_flow_week{i+1}" for i in range(NUMBER_OF_WEEKS)]

FORBIDDEN_COLS = [
    "water_flow_lag_1w",
    "water_flow_lag_2w",
] 

FEATURE_DROP_COLS = ["station_code"] + TARGET_COLS + FORBIDDEN_COLS

class LightGBMQuantileRegressor(RegressorMixin, BaseEstimator):
    """Wrapper to train Lower, Median, and Upper quantile models."""
    def __init__(self, alpha=0.1, **kwargs):
        self.alpha = alpha
        self.kwargs = kwargs
        self.models = {}
        
    def fit(self, X, y):
        # Train Lower Bound
        self.models['lower'] = lgb.LGBMRegressor(
            objective='quantile', alpha=self.alpha/2,
            verbose=-1, **self.kwargs
        ).fit(X, y)
        
        # Train Median (Point Prediction)
        self.models['median'] = lgb.LGBMRegressor(
            objective='quantile', alpha=0.5,
            verbose=-1, **self.kwargs
        ).fit(X, y)
        
        # Train Upper Bound
        self.models['upper'] = lgb.LGBMRegressor(
            objective='quantile', alpha=1.0 - self.alpha/2,
            verbose=-1, **self.kwargs
        ).fit(X, y)
        return self

    def predict(self, X, quantiles=None):
        if quantiles == "mean":
            return self.models['median'].predict(X)
        elif isinstance(quantiles, list):
            # Return shape (N, 2) for [lower, upper]
            lower = self.models['lower'].predict(X)
            upper = self.models['upper'].predict(X)
            return np.column_stack([lower, upper])
        return self.models['median'].predict(X)

def load_dataset(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    df = pd.read_csv(filepath, index_col=0)
    return df

def build_features(df: pd.DataFrame, reference_columns: list[str] | None = None) -> pd.DataFrame:
    """Build features with consistent dummy encoding.
    
    Args:
        df: Input dataframe
        reference_columns: If provided, align output columns to this list (from training set)
    """
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
) -> pd.DataFrame:
    """Robust computation of station metrics with sigma clamping."""
    station_list = np.unique(stations)
    records = []

    for s in station_list:
        idx = stations == s
        y_true_s = y_true_std[idx]
        y_pred_s = y_pred_std[idx]
        y_lower_s = y_pred_lower_std[idx]
        y_upper_s = y_pred_upper_std[idx]

        rmse_s = sqrt(mean_squared_error(y_true_s, y_pred_s))
        mae_s = mean_absolute_error(y_true_s, y_pred_s)

        # Robust Sigma Estimation
        # 1. Estimate sigma from 95% CI (width / 3.29)
        sigma_s = (y_upper_s - y_lower_s) / 3.29
        
        # 2. CLAMPING: Prevent sigma from being too close to zero.
        # Since data is standardized [0, 100], a sigma < 0.1 implies extremely high confidence.
        # We clamp sigma to prevent log(0) exploding to infinity.
        sigma_s = np.maximum(sigma_s, 0.5)

        # Compute Gaussian NLL (proper formula)
        # NLL = 0.5 * log(2 * pi * sigma^2) + (y - y_hat)^2 / (2 * sigma^2)
        #     = 0.5 * log(2 * pi) + log(sigma) + (y - y_hat)^2 / (2 * sigma^2)
        # residuals = y_true_s - y_pred_s
        # nll_s = np.mean(
        #     0.5 * np.log(2 * np.pi) + np.log(sigma_s) + (residuals ** 2) / (2 * sigma_s ** 2)
        # )
        
        # OLD Laplace-ish NLL
        log_term = np.log(sigma_s)
        resid_term = np.abs(y_true_s - y_pred_s) / (2 * sigma_s)
        nll_s = np.mean(log_term + resid_term)

        coverage_s = np.mean((y_true_s >= y_lower_s) & (y_true_s <= y_upper_s))
        interval_size_s = np.mean(y_upper_s - y_lower_s)

        records.append({
            "station_code": s,
            "scaled_rmse": rmse_s,
            "scaled_mae": mae_s,
            "coverage": coverage_s,
            "scaled_interval_size": interval_size_s,
            "log_likelihood": nll_s,
        })

    return pd.DataFrame(records)

def analyze_collinearity(X, threshold=0.9):
    """
    Analyzes collinearity using Hierarchical Clustering on Spearman Correlation.
    """
    print("\n--- 1. Pre-Analysis: Collinearity Detection ---")
    
    # 1. Compute Spearman Correlation
    # We use a subset if X is too large for speed
    X_sample = X.sample(n=min(5000, len(X)), random_state=42)
    corr_matrix = X_sample.corr(method='spearman').abs()
    
    # 2. Hierarchical Clustering
    # Convert correlation to distance
    distance_matrix = 1 - corr_matrix
    # handle floating point errors where distance < 0
    distance_matrix = distance_matrix.clip(lower=0)
    
    # Work with numpy array to avoid read-only issues
    dist_array = distance_matrix.values.copy()
    
    # Ensure symmetric
    dist_array = (dist_array + dist_array.T) / 2
    np.fill_diagonal(dist_array, 0)
    
    # Condensed distance matrix for hierarchy.linkage
    dist_linkage = squareform(dist_array)
    Z = hierarchy.linkage(dist_linkage, method='ward')
    
    # 3. Plot Dendrogram
    
    # Absolute paths
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
    
    # 4. Identify Clusters
    # fcluster returns cluster IDs. Criterion 'distance' with threshold corresponds to 1 - corr_threshold
    cluster_ids = hierarchy.fcluster(Z, t=1-threshold, criterion='distance')
    
    cluster_mapping = {}
    for feature, cluster_id in zip(X.columns, cluster_ids):
        cluster_mapping.setdefault(cluster_id, []).append(feature)
        
    print(f"\nFeature Clusters (Correlation > {threshold}):")
    found_collinear = False
    for cluster_id, features in cluster_mapping.items():
        if len(features) > 1:
            print(f"  Cluster {cluster_id}: {features}")
            found_collinear = True
    
    if not found_collinear:
        print("  No highly collinear features found.")

def train_multiweek():
    # --- 1. Loading Data ---
    dataset = load_dataset(INPUT_FILE)
    dataset = add_meteo_lags(dataset, METEO_LAG_COLUMNS, METEO_LAG_WINDOWS, use_ewm=USE_EWMA)
    
    # Compute GLOBAL station stats for consistent standardization
    # We use the full dataset min/max to define the "scale" of the river
    # This prevents division by zero if the test set has constant flow
    # Use week1 as proxy for general flow magnitude
    global_stats = get_station_stats(dataset["water_flow_week1"].values, dataset["station_code"].values)

    train, test_spatio_temporal, test_temporal = split_dataset(
        dataset, p=TRAIN_STATION_FRACTION, time=TIME_VALIDATION
    )

    X_train = build_features(train)
    train_columns = X_train.columns.tolist()  # Reference columns for consistent encoding
    
    # --- Pre-Analysis ---
    analyze_collinearity(X_train)
    
    X_test = build_features(test_spatio_temporal, reference_columns=train_columns)
    test_stations = test_spatio_temporal["station_code"].values
    
    # Validation set for Calibration (Conformal Prediction)
    # We use the temporal split (Seen Stations, Future Time) as calibration
    X_calib = build_features(test_temporal, reference_columns=train_columns)
    
    print(f"Train shapes: X={X_train.shape}")
    
    print("\n--- 2. Training LightGBM per week ---")
    models = {}
    for i, target_col in enumerate(TARGET_COLS):
        print(f"Training week {i} ({target_col})")
        
        # Default parameters (Aggressive but accurate)
        model = LightGBMQuantileRegressor(
            alpha=ALPHA, 
            n_estimators=500, 
            learning_rate=0.05, 
            num_leaves=31
        )
        model.fit(X_train, train[target_col])
        models[i] = model
        
        # Save Week 0 Model
        if i == 0:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            MODEL_DIR = os.path.join(BASE_DIR, "models", "final")
            os.makedirs(MODEL_DIR, exist_ok=True)
            
            save_path = os.path.join(MODEL_DIR, "lgbm_week0.joblib")
            joblib.dump(model, save_path)
            print(f"Week 0 model saved to {save_path}")

    print("\n--- 3. Evaluation ---")
    
    # --- MIXED CALIBRATION SETUP (toggle with USE_MIXED_CALIBRATION) ---
    if USE_MIXED_CALIBRATION:
        # Split test_spatio_temporal into calibration and final test
        calib_idx, final_test_idx = train_test_split(
            np.arange(len(test_spatio_temporal)),
            test_size=1 - MIXED_CALIB_FRACTION,
            random_state=42
        )
        X_calib_mixed = X_test.iloc[calib_idx]
        X_final_test = X_test.iloc[final_test_idx]
        test_stations_final = test_stations[final_test_idx]
        print(f"Mixed calibration; {MIXED_CALIB_FRACTION:.2%} of spatiotemporal split.")
        print(f"{len(calib_idx)} calib, {len(final_test_idx)} test samples.")
    else:
        # Original behavior: use test_temporal for calibration, full test_spatio_temporal for eval
        X_calib_mixed = X_calib  # This is test_temporal
        X_final_test = X_test
        test_stations_final = test_stations
    # --- END MIXED CALIBRATION SETUP ---
    
    for i, target_col in enumerate(TARGET_COLS):
        
        # Select appropriate data based on calibration mode
        if USE_MIXED_CALIBRATION:
            y_true = test_spatio_temporal[target_col].values[final_test_idx]
            y_calib_target = test_spatio_temporal[target_col].values[calib_idx]
        else:
            y_true = test_spatio_temporal[target_col].values
            y_calib_target = test_temporal[target_col].values
        
        # Raw Quantiles on final test set
        y_pred = models[i].predict(X_final_test, quantiles="mean")
        y_quantiles = models[i].predict(X_final_test, quantiles=[ALPHA/2, 1-ALPHA/2])
        y_lower = y_quantiles[:, 0]
        y_upper = y_quantiles[:, 1]
        
        # Conformal Calibration (CQR-style)
        if CALIBRATE_INTERVALS:
            # 1. Compute scores on Calibration Set
            y_calib_quantiles = models[i].predict(X_calib_mixed, quantiles=[ALPHA/2, 1-ALPHA/2])
            
            low_calib = y_calib_quantiles[:, 0]
            high_calib = y_calib_quantiles[:, 1]
            
            # CQR Score: max(low - y, y - high)
            # We want y \in [low - q, high + q]
            scores = np.maximum(low_calib - y_calib_target, y_calib_target - high_calib)
            
            # Compute (1 - alpha) quantile of scores
            q_score = np.quantile(scores, 1 - ALPHA)
        else:
            q_score = 0.0
        
        # Adjust intervals
        y_lower_adj = y_lower - q_score
        y_upper_adj = y_upper + q_score
        
        # Force non-negative lower bound for discharge
        y_lower_adj = np.maximum(0, y_lower_adj)

        # --- CONSISTENCY CHECK ---
        # Ensure Median is within [Lower, Upper]
        # If Median < Lower, lower the Lower bound.
        # If Median > Upper, raise the Upper bound.
        y_lower_adj = np.minimum(y_lower_adj, y_pred)
        y_upper_adj = np.maximum(y_upper_adj, y_pred)

        # Force Upper >= Lower + small epsilon to avoid log(0) or log(neg) in metrics
        y_upper_adj = np.maximum(y_upper_adj, y_lower_adj + 1e-4)

        print(f"Week {i} Conformal Correction (q_score): {q_score:.3f}")
        
        # Compute Metrics
        # Use GLOBAL stats to avoid "division by tiny range" issues in local test sets
        station_stats = global_stats.reindex(np.unique(test_stations_final)).fillna(1.0)
        
        y_true_std = standardize_values(y_true, test_stations_final, station_stats)
        y_pred_std = standardize_values(y_pred, test_stations_final, station_stats)
        y_lower_std, y_upper_std = standardize_prediction_intervals(
            np.column_stack([y_lower_adj, y_upper_adj]), test_stations_final, station_stats
        )
        
        station_metrics = compute_per_station_metrics_robust(
            y_true_std, y_pred_std, test_stations_final, y_lower_std, y_upper_std
        )
        
        summary = summarize_metrics(station_metrics, "LGBM+CQR", "test")
        print(summary.to_string(index=False))

if __name__ == "__main__":
    train_multiweek()
