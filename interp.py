import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.cluster.hierarchy import complete, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import mean_squared_error

from analysis import load_config, prepare_analysis_data
from pipeline import ALPHA, NUMBER_OF_WEEKS, build_features, load_split
from src.utils.plots import COLORS, apply_style

# Global constants for Phase 2/3/4
CORR_THRESHOLD_TAU = 0.25  # Distance threshold for hierarchical clustering
N_TOP_CLUSTERS = 6
MAX_EVAL_ROWS = 500  # For SHAP and ALE computational efficiency
RANDOM_SEED = 42


@dataclass
class FeatureCluster:
    cluster_id: int
    features: list[str]
    medoid: str
    importance: float = 0.0


def get_calibration_data(config: dict, train_cols: list[str]) -> pd.DataFrame:
    """Load the calibration data used during training based on config."""
    calib_dfs = []
    if config.get("calib_temp"):
        calib_dfs.append(load_split("temporal_calib"))
    if config.get("calib_stemp") or config.get("calib_spatio_only"):
        calib_dfs.append(load_split("spatiotemporal_calib"))

    if not calib_dfs:
        calib_dfs.append(load_split("temporal_calib"))

    df_cal = pd.concat(calib_dfs)
    X_cal = build_features(df_cal, reference_columns=train_cols)
    return X_cal


def compute_ale(
    model_func: Callable[[pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    feature: str,
    bins: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase 4: Manual ALE implementation."""
    x_val = X[feature].values
    q = np.unique(np.quantile(x_val, np.linspace(0, 1, bins + 1)))
    if len(q) < 2:
        return np.array([x_val.min(), x_val.max()]), np.array([0.0, 0.0])

    bin_indices = np.digitize(x_val, q, right=True)
    bin_indices[bin_indices == 0] = 1
    bin_indices[bin_indices > len(q) - 1] = len(q) - 1

    ale = np.zeros(len(q) - 1)
    for i in range(1, len(q)):
        idx = bin_indices == i
        if not np.any(idx):
            continue
        X_low = X.loc[idx].copy()
        X_high = X.loc[idx].copy()
        X_low.loc[:, feature] = q[i - 1]
        X_high.loc[:, feature] = q[i]

        diff = model_func(X_high) - model_func(X_low)
        ale[i - 1] = np.mean(diff)

    ale = np.cumsum(ale)
    ale = np.concatenate([[0], ale])

    hist, _ = np.histogram(x_val, bins=q)
    bin_means = (ale[:-1] + ale[1:]) / 2
    mean_ale = np.sum(bin_means * hist) / len(x_val)
    ale -= mean_ale

    return q, ale


def get_shap_values(wrapper: Any, X: pd.DataFrame, mode: str = "width") -> np.ndarray:
    """
    Phase 3: Path-Dependent TreeSHAP.
    If mode='width': Returns Shap(q_upper) - Shap(q_lower).
    If mode='median': Returns Shap(q_median).
    """
    if hasattr(wrapper, "models") and isinstance(wrapper.models, dict):
        if mode == "width":
            m_low = wrapper.models["lower"]
            m_high = wrapper.models["upper"]
            explainer_low = shap.TreeExplainer(m_low, feature_perturbation="tree_path_dependent")
            explainer_high = shap.TreeExplainer(m_high, feature_perturbation="tree_path_dependent")
            sv_low = explainer_low.shap_values(X)
            sv_high = explainer_high.shap_values(X)
            if isinstance(sv_low, list): sv_low = sv_low[0]
            if isinstance(sv_high, list): sv_high = sv_high[0]
            return sv_high - sv_low
        else:
            m_med = wrapper.models["median"]
            explainer = shap.TreeExplainer(m_med, feature_perturbation="tree_path_dependent")
            sv = explainer.shap_values(X)
            return sv[0] if isinstance(sv, list) else sv
        
    if hasattr(wrapper, "model") and "RandomForestQuantileRegressor" in str(type(wrapper.model)):
        m = wrapper.model
        explainer = shap.TreeExplainer(m, feature_perturbation="tree_path_dependent")
        sv = explainer.shap_values(X)
        if mode == "width":
            if isinstance(sv, list) and len(sv) >= 2:
                return sv[-1] - sv[0]
            elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                return sv[:, :, -1] - sv[:, :, 0]
        else:
            # Median is usually in the middle or index 1 for [low, med, high]
            if isinstance(sv, list): return sv[len(sv)//2]
            elif isinstance(sv, np.ndarray) and sv.ndim == 3: return sv[:, :, sv.shape[2]//2]
        return sv
    
    explainer = shap.Explainer(wrapper.predict, X)
    return explainer(X).values


def main():
    parser = argparse.ArgumentParser(description="Advanced Model Interpretability (CQR + Grouped SHAP + ALE)")
    parser.add_argument("--results-dir", type=str, required=True, help="Path to results.")
    parser.add_argument("--week", type=int, default=0, choices=range(NUMBER_OF_WEEKS), help="Week index.")
    parser.add_argument("--mode", type=str, default="width", choices=["width", "median"], 
                        help="Target: 'width' for uncertainty, 'median' for point prediction.")
    parser.add_argument("--tau", type=float, default=CORR_THRESHOLD_TAU, help="Clustering threshold.")
    args = parser.parse_args()

    # Phase 1: Setup
    print(f"--- Phase 1: Context Definition (Mode: {args.mode}) ---")
    config, wrapper, X_test, y_test, _, target_col, _ = prepare_analysis_data(args.results_dir, args.week)
    
    train_cols = X_test.columns.tolist()
    X_cal_full = get_calibration_data(config, train_cols)
    n_samples = min(MAX_EVAL_ROWS, len(X_cal_full))
    X_cal = X_cal_full.sample(n=n_samples, random_state=RANDOM_SEED)
    print(f"Using {n_samples} samples for analysis.")

    # Define target function
    def target_func(X_df: pd.DataFrame) -> np.ndarray:
        if args.mode == "width":
            preds = wrapper.predict(X_df.values, quantiles=[ALPHA / 2, 1 - ALPHA / 2], calibrate=True)
            return preds[:, 1] - preds[:, 0]
        else:
            return wrapper.predict(X_df.values, quantiles="mean")

    # Phase 2: Clustering
    print(f"\n--- Phase 2: Feature Space Partitioning ---")
    corr_matrix = X_cal.corr(method="spearman").abs().fillna(0.0)
    dist_matrix = 1.0 - corr_matrix
    linkage_matrix = complete(squareform(dist_matrix.values))
    cluster_labels = fcluster(linkage_matrix, args.tau, criterion="distance")
    
    feature_to_cluster = {feat: label for feat, label in zip(X_cal.columns, cluster_labels)}
    clusters_dict: dict[int, list[str]] = {}
    for feat, label in feature_to_cluster.items():
        clusters_dict.setdefault(label, []).append(feat)
    
    feature_clusters = []
    for cid, feats in clusters_dict.items():
        sub_dist = dist_matrix.loc[feats, feats]
        medoid = sub_dist.sum(axis=1).idxmin()
        feature_clusters.append(FeatureCluster(cluster_id=cid, features=feats, medoid=str(medoid)))
    
    # Phase 3: SHAP
    print(f"\n--- Phase 3: Grouped Asymmetric Shapley Attribution ---")
    shap_values = get_shap_values(wrapper, X_cal, mode=args.mode)
    feat_to_idx = {feat: i for i, feat in enumerate(X_cal.columns)}
    
    for cluster in feature_clusters:
        indices = [feat_to_idx[f] for f in cluster.features]
        cohort_shap = shap_values[:, indices].sum(axis=1)
        cluster.importance = np.mean(np.abs(cohort_shap))
    
    feature_clusters.sort(key=lambda c: c.importance, reverse=True)
    top_clusters = feature_clusters[:N_TOP_CLUSTERS]

    # Phase 4: ALE
    print(f"\n--- Phase 4: Medoid ALE Curves ---")
    fig_dir = Path(args.results_dir) / "figures" / f"week{args.week}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    n_cols = 3
    n_rows = (len(top_clusters) + n_cols - 1) // n_cols
    
    for i, cluster in enumerate(top_clusters):
        q, ale = compute_ale(target_func, X_cal, cluster.medoid)
        plt.subplot(n_rows, n_cols, i + 1)
        plt.plot(q, ale, marker='o', markersize=3, color=COLORS[0], linewidth=1.5, alpha=0.9)
        plt.axhline(0, color='#333333', linestyle='--', alpha=0.3, linewidth=1.0)
        plt.title(f"{cluster.medoid}\n(Imp: {cluster.importance:.3f})", loc="left", fontsize=10, fontweight="semibold", pad=8)
        plt.ylabel(f"ALE on {args.mode}", fontsize=9)
        plt.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(fig_dir / f"week{args.week}_{args.mode}_ale_curves.png")
    plt.close()

    # Save summary
    pd.DataFrame([{
        "rank": i + 1, "medoid": c.medoid, "size": len(c.features), "importance": c.importance, "features": ", ".join(c.features)
    } for i, c in enumerate(feature_clusters)]).to_csv(fig_dir / f"week{args.week}_{args.mode}_catalog.csv", index=False)

    # Plot Importance
    plt.figure(figsize=(9, 5))
    top_names = [f"{c.medoid} (n={len(c.features)})" for c in top_clusters[::-1]]
    plt.barh(
        top_names, 
        [c.importance for c in top_clusters[::-1]], 
        color=COLORS[6] if args.mode=="width" else COLORS[1],
        alpha=0.88,
        edgecolor="white",
        linewidth=0.5
    )
    plt.title(f"Top {N_TOP_CLUSTERS} Grouped Importance (Target: {args.mode})", loc="left", fontweight="semibold", pad=12)
    plt.xlabel("Global Mean |Grouped SHAP|", fontsize=9)
    plt.tick_params(labelsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / f"week{args.week}_{args.mode}_importance.png")
    plt.close()

    print(f"\nAnalysis for {args.mode} complete. Results in {fig_dir}")

if __name__ == "__main__":
    apply_style()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        main()
