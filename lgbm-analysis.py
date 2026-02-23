import argparse
import os

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.inspection import permutation_importance

from analysis import ALPHA, get_point_model, prepare_analysis_data, run_error_analysis


def run_lgbm_analysis():
    parser = argparse.ArgumentParser(description="Analyze LightGBM model (specific)")
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Path to the results directory (containing config.json and model)",
    )
    parser.add_argument(
        "--week",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Week index to analyze (0-3). Default: 0",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    week_idx = args.week
    rng = np.random.default_rng(42)

    config, wrapper, X_test, y_test, test_eval, target_col_name, model_path = (
        prepare_analysis_data(results_dir, week_idx)
    )

    calib_temp = config.get("calib_temp", False)
    calib_stemp = config.get("calib_stemp", False)
    if "calib_temp_ratio" in config:
        calib_temp = float(config["calib_temp_ratio"]) > 0
    if "calib_stemp_ratio" in config:
        calib_stemp = float(config["calib_stemp_ratio"]) > 0

    model_id = config.get("model_id")
    model_name = config.get("model", model_id or "Unknown")

    print(f"Loaded configuration from {results_dir}")
    if model_id:
        print(f"Model: {model_name} (id: {model_id})")
    else:
        print(f"Model: {model_name}")
    print(f"Targeting Week: {week_idx} (water_flow_week{week_idx+1})")
    print(f"Calibration: Temp={calib_temp}, SpatioTemp={calib_stemp}")
    print(f"Loading model from {model_path}...")
    print(f"Analyzing on Spatio-Temporal Test Set ({len(X_test)} samples).")

    point_model = get_point_model(wrapper)
    if not isinstance(point_model, lgb.LGBMRegressor):
        raise TypeError("lgbm-analysis.py requires a LightGBM model.")

    fig_dir = os.path.join(results_dir, "figures", f"week{week_idx}")
    os.makedirs(fig_dir, exist_ok=True)

    print("\n--- 1. LGBM Gain Importance ---")
    plt.figure(figsize=(10, 12))
    lgb.plot_importance(
        point_model,
        importance_type="gain",
        max_num_features=20,
        figsize=(10, 12),
        title=f"LGBM Gain Importance (Week {week_idx})",
    )
    plt.tight_layout()
    save_path = os.path.join(fig_dir, f"week{week_idx}_01_gain_importance.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

    print("\n--- 2. Permutation Importance ---")
    print("Computing permutation importance (this may take a moment)...")

    sample_indices = rng.choice(len(X_test), size=min(2000, len(X_test)), replace=False)
    X_perm = X_test.iloc[sample_indices]
    y_perm = y_test[sample_indices]

    result = permutation_importance(
        point_model,
        X_perm,
        y_perm,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
        scoring="neg_root_mean_squared_error",
    )

    perm_sorted_idx = result.importances_mean.argsort()[-20:]
    plt.figure(figsize=(10, 12))
    plt.boxplot(
        result.importances[perm_sorted_idx].T,
        vert=False,
        labels=X_perm.columns[perm_sorted_idx],
    )
    plt.title(f"Permutation Importance (Week {week_idx})")
    plt.tight_layout()
    save_path = os.path.join(fig_dir, f"week{week_idx}_02_permutation_importance.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

    print("\n--- 3. SHAP Summary ---")
    print("Calculating SHAP values...")
    shap_indices = rng.choice(len(X_test), size=min(2000, len(X_test)), replace=False)
    X_shap = X_test.iloc[shap_indices]

    explainer = shap.TreeExplainer(point_model)
    shap_values = explainer.shap_values(X_shap)
    if isinstance(shap_values, list):
        shap_array = shap_values[0]
    else:
        shap_array = shap_values

    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_array, X_shap, show=False)
    plt.title(f"SHAP Summary Plot (Week {week_idx})")
    plt.tight_layout()
    save_path = os.path.join(fig_dir, f"week{week_idx}_03_shap_summary.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

    mean_abs_shap = np.abs(shap_array).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-5:][::-1]
    top_features = X_shap.columns[top_indices]
    print(f"\nTop 5 SHAP Features: {top_features.tolist()}")

    print("\n--- 4. SHAP Dependence Plots ---")
    for feature in top_features:
        print(f"Generating dependence plot for {feature}...")
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            feature, shap_array, X_shap, display_features=X_shap, show=False
        )
        plt.title(f"SHAP Dependence: {feature} (Week {week_idx})")
        plt.tight_layout()
        clean_name = feature.replace("/", "_").replace(" ", "_")
        save_path = os.path.join(
            fig_dir, f"week{week_idx}_04_shap_dependence_{clean_name}.png"
        )
        plt.savefig(save_path)
        plt.close()
        print(f"Saved dependence plot for {feature}")

    run_error_analysis(
        wrapper=wrapper,
        X_test=X_test,
        test_eval=test_eval,
        target_col_name=target_col_name,
        fig_dir=fig_dir,
        week_idx=week_idx,
        alpha=ALPHA,
    )

    print("\nAnalysis Complete.")


if __name__ == "__main__":
    run_lgbm_analysis()
