import os

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error

from src.utils.model import split_dataset
from src.utils.plots import plot_water_flow_predictions

# --- CONFIGURATION (Must match lgbm-cqr.py) ---
INPUT_FILE = "./data/input/dataset_baseline.csv"
TIME_VALIDATION = "2000-01-01 00:00:00"
TRAIN_STATION_FRACTION = 0.75
NUMBER_OF_WEEKS = 4
ALPHA = 0.1
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
FORBIDDEN_COLS = ["water_flow_lag_1w", "water_flow_lag_2w"]
FEATURE_DROP_COLS = ["station_code"] + TARGET_COLS + FORBIDDEN_COLS

# --- Helper Functions (Duplicated from lgbm-cqr.py) ---

class LightGBMQuantileRegressor(RegressorMixin, BaseEstimator):
    """Wrapper to train Lower, Median, and Upper quantile models."""
    def __init__(self, alpha=0.1, **kwargs):
        self.alpha = alpha
        self.kwargs = kwargs
        self.models = {}
        
    def fit(self, X, y):
        self.models['lower'] = lgb.LGBMRegressor(
            objective='quantile', alpha=self.alpha/2, verbose=-1, **self.kwargs
        ).fit(X, y)
        self.models['median'] = lgb.LGBMRegressor(
            objective='quantile', alpha=0.5, verbose=-1, **self.kwargs
        ).fit(X, y)
        self.models['upper'] = lgb.LGBMRegressor(
            objective='quantile', alpha=1.0 - self.alpha/2, verbose=-1, **self.kwargs
        ).fit(X, y)
        return self

    def predict(self, X, quantiles=None):
        if quantiles == "mean":
            return self.models['median'].predict(X)
        elif isinstance(quantiles, list):
            lower = self.models['lower'].predict(X)
            upper = self.models['upper'].predict(X)
            return np.column_stack([lower, upper])
        return self.models['median'].predict(X)

def load_dataset(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath, index_col=0)

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

def build_features(df: pd.DataFrame, reference_columns: list[str] | None = None) -> pd.DataFrame:
    X = df.drop(columns=FEATURE_DROP_COLS, errors="ignore")
    X = pd.get_dummies(X, drop_first=True)
    if reference_columns is not None:
        for col in reference_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[[col for col in reference_columns if col in X.columns]]
        X = X.reindex(columns=reference_columns, fill_value=0)
    return X

# --- Analysis Logic ---

def run_analysis():
    print("--- 1. Loading Data & Model ---")
    rng = np.random.default_rng(42)
    dataset = load_dataset(INPUT_FILE)
    dataset = add_meteo_lags(dataset, METEO_LAG_COLUMNS, METEO_LAG_WINDOWS, use_ewm=USE_EWMA)
    
    # Split
    train, test_spatio_temporal, _ = split_dataset(dataset, p=TRAIN_STATION_FRACTION, time=TIME_VALIDATION)
    
    # Features
    X_train = build_features(train)
    train_columns = X_train.columns.tolist()
    X_test = build_features(test_spatio_temporal, reference_columns=train_columns)
    
    # Target (Week 0)
    y_test = test_spatio_temporal[TARGET_COLS[0]].values
    
    # Load Model
    model_path = "./models/final/lgbm_week0.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run lgbm-cqr.py first.")
    
    print(f"Loading model from {model_path}...")
    wrapper = joblib.load(model_path)
    median_model = wrapper.models['median'] # LGBMRegressor
    
    os.makedirs("./figures/models/", exist_ok=True)

    print("\n--- 2. Sanity Check: LGBM Gain Importance ---")
    plt.figure(figsize=(10, 12))
    lgb.plot_importance(
        median_model,
        importance_type='gain',
        max_num_features=20,
        figsize=(10, 12),
        title="LGBM Gain Importance (Top 20)",
    )
    plt.tight_layout()
    plt.savefig("./figures/models/week0_01_gain_importance.png")
    plt.close()
    print("Saved: week0_01_gain_importance.png")

    print("\n--- 3. Robustness Check: Permutation Importance ---")
    # We run this on a sample of the test set for speed
    print("Computing permutation importance (this may take a moment)...")
    
    sample_indices = rng.choice(len(X_test), size=min(2000, len(X_test)), replace=False)
    X_perm = X_test.iloc[sample_indices]
    y_perm = y_test[sample_indices]
    
    result = permutation_importance(
        median_model,
        X_perm,
        y_perm,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
        scoring='neg_root_mean_squared_error',
    )
    
    perm_sorted_idx = result.importances_mean.argsort()[-20:]  # Top 20
    
    plt.figure(figsize=(10, 12))
    plt.boxplot(
        result.importances[perm_sorted_idx].T,
        vert=False,
        labels=X_perm.columns[perm_sorted_idx]
    )
    plt.title("Permutation Importance (Test Set)")
    plt.tight_layout()
    plt.savefig("./figures/models/week0_02_permutation_importance.png")
    plt.close()
    print("Saved: week0_02_permutation_importance.png")

    print("\n--- 4. Deep Dive: SHAP Summary ---")
    print("Calculating SHAP values...")
    # Use TreeExplainer
    explainer = shap.TreeExplainer(median_model)
    # Use a larger sample for SHAP summary
    
    # Reuse indices if we want, or resample. Let's resample to be robust.
    shap_indices = rng.choice(len(X_test), size=min(2000, len(X_test)), replace=False)
    X_shap = X_test.iloc[shap_indices]
    shap_values = explainer.shap_values(X_shap)
    
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_shap, show=False)
    plt.title("SHAP Summary Plot (Week 0)")
    plt.tight_layout()
    plt.savefig("./figures/models/week0_03_shap_summary.png")
    plt.close()
    print("Saved: week0_03_shap_summary.png")
    
    # Identify Top 5 Features by absolute SHAP mean
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-5:][::-1]
    top_features = X_shap.columns[top_indices]
    
    print(f"\nTop 5 SHAP Features: {top_features.tolist()}")

    print("\n--- 5. Non-Linearity: SHAP Dependence Plots ---")
    for feature in top_features:
        print(f"Generating dependence plot for {feature}...")
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            feature, shap_values, X_shap, display_features=X_shap, show=False
        )
        plt.title(f"SHAP Dependence: {feature}")
        plt.tight_layout()
        clean_name = feature.replace("/", "_").replace(" ", "_")
        plt.savefig(f"./figures/models/week0_04_shap_dependence_{clean_name}.png")
        plt.close()
        print(f"Saved dependence plot for {feature}")

    print("\n--- 6. Error Analysis: Worst Stations ---")
    print("Generating predictions for full test set...")
    
    # 1. Predictions
    y_pred_median = wrapper.predict(X_test, quantiles="mean")
    y_pred_intervals = wrapper.predict(X_test, quantiles=[ALPHA/2, 1-ALPHA/2]) # Shape (N, 2)
    
    # 2. Compute RMSE per Station
    # Reconstruct DataFrame for easy grouping
    results_df = test_spatio_temporal.copy()
    results_df['y_true'] = results_df[TARGET_COLS[0]]
    results_df['y_pred'] = y_pred_median
    
    station_rmse = results_df.groupby('station_code').apply(
        lambda x: np.sqrt(mean_squared_error(x['y_true'], x['y_pred']))
    ).sort_values(ascending=False)
    
    worst_stations = station_rmse.head(3).index.tolist()
    print(f"Top 3 Worst Stations (RMSE): {worst_stations}")
    print(station_rmse.head(3))
    
    # 3. Plot
    for station in worst_stations:
        print(f"Plotting predictions for station {station}...")
        
        # Filter data for this station
        mask = results_df['station_code'] == station
        station_data = results_df[mask].copy()
        
        # Ensure 'water_flow_week1' is present as expected by plot_water_flow_predictions
        # (It is, because results_df is a copy of test_spatio_temporal which has it)
        
        station_pred = y_pred_median[mask]
        station_pis = y_pred_intervals[mask]
        
        # Reset index if ObsDate is in index, as plot function expects ObsDate column
        if 'ObsDate' not in station_data.columns and station_data.index.name == 'ObsDate':
            station_data = station_data.reset_index()
            
        try:
            plot_water_flow_predictions(
                ground_truth=station_data,
                prediction=station_pred,
                y_pis=station_pis,
                prefixe=f"week0_worst_{station}",
                save=True,
                display=False
            )
        except Exception as e:
            print(f"Error plotting station {station}: {e}")

    print("\nAnalysis Complete.")

if __name__ == "__main__":
    run_analysis()
