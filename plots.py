import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler

CALIBRATION_ORDER = ["none", "temp", "spatio", "mixed"]

PLOT_COLORS = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#72B7B2",
]


def apply_plot_style():
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.linewidth": 0.8,
            "legend.fontsize": 9,
            "legend.title_fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.3,
            "axes.axisbelow": True,
            "axes.prop_cycle": cycler(color=PLOT_COLORS),
        }
    )


def infer_calibration_mode(config: dict) -> str:
    calib_temp = bool(config.get("calib_temp", False))
    calib_stemp = bool(config.get("calib_stemp", False))
    calib_spatio_only = bool(config.get("calib_spatio_only", False))
    if "calib_temp_ratio" in config:
        calib_temp = float(config["calib_temp_ratio"]) > 0
    if "calib_stemp_ratio" in config:
        calib_stemp = float(config["calib_stemp_ratio"]) > 0

    if calib_spatio_only:
        return "spatio"
    if calib_stemp:
        return "mixed"
    if calib_temp:
        return "temp"
    return "none"


def load_results(result_dirs):
    data = []
    
    for rdir in result_dirs:
        config_path = os.path.join(rdir, "config.json")
        metrics_path = os.path.join(rdir, "metrics_summary.csv")
        
        if not os.path.exists(config_path) or not os.path.exists(metrics_path):
            print(f"Skipping {rdir}: Missing config.json or metrics_summary.csv")
            continue
            
        with open(config_path, "r") as f:
            config = json.load(f)
            
        # Determine label: Experiment Name > Folder Name
        exp_name = config.get("experiment_name", "")
        if not exp_name:
            exp_name = os.path.basename(os.path.normpath(rdir))

        model_id = config.get("model_id", config.get("model", "unknown"))
        calibration = infer_calibration_mode(config)
        
        df = pd.read_csv(metrics_path)
        df["experiment"] = exp_name
        df["model_id"] = model_id
        df["calibration"] = calibration
        df["label"] = f"{exp_name} | {model_id} | {calibration}"
        data.append(df)
        
    if not data:
        raise ValueError("No valid results found.")
        
    return pd.concat(data, ignore_index=True)

def format_value(metric_col: str, value: float) -> str:
    if metric_col in {"coverage", "coverage_gap"}:
        return f"{value:.3f}"
    return f"{value:.2f}"


def annotate_bars(ax, bars, metric_col: str):
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        label = format_value(metric_col, height)
        offset = 3 if height >= 0 else -9
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=8,
        )


def compute_metric_ylim(df, metric_col: str):
    values = df[metric_col].to_numpy(dtype=float)
    values = values[np.isfinite(values)]

    if metric_col == "coverage":
        return (0.0, 1.0)

    if values.size == 0:
        return None

    if metric_col == "coverage_gap":
        max_abs = float(np.max(np.abs(values)))
        bound = max(0.02, max_abs * 1.1)
        return (-bound, bound)

    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmin, vmax):
        pad = max(abs(vmax) * 0.1, 1e-3)
    else:
        pad = (vmax - vmin) * 0.12
    lower = max(0.0, vmin - pad)
    upper = vmax + pad
    return (lower, upper)


def plot_metric(df, dataset_name, metric_col, ylabel, title, ax, label_col, legend_title, palette):
    # Filter for dataset
    subset = df[df["dataset"] == dataset_name].copy()
    
    if subset.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    # Sort by week and label
    subset = subset.sort_values(["week", label_col])
    
    weeks = sorted(subset["week"].unique())
    labels = subset[label_col].unique()
    if label_col == "calibration":
        labels = [c for c in CALIBRATION_ORDER if c in labels]
    
    # Bar Width configuration
    n_exp = len(labels)
    bar_width = 0.8 / n_exp
    indices = np.arange(len(weeks))
    
    for i, label in enumerate(labels):
        exp_data = subset[subset[label_col] == label]
        
        # Align data to weeks ensuring missing weeks don't break plot
        # Reindex to ensure shape matches 'indices'
        exp_data = exp_data.set_index("week").reindex(weeks)
        
        values = exp_data[metric_col].values
        
        # Offset bars
        x_pos = indices + (i - n_exp/2 + 0.5) * bar_width
        
        bars = ax.bar(
            x_pos,
            values,
            width=bar_width,
            label=label,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.6,
            color=palette[i % len(palette)],
        )
        annotate_bars(ax, bars, metric_col)
        
    ax.set_xlabel("Forecast Horizon (Weeks)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(indices)
    ax.set_xticklabels([f"Week {w}" for w in weeks])
    ax.grid(axis='y', linestyle='--', alpha=0.5)

def main():
    parser = argparse.ArgumentParser(description="Compare multiple experiment results.")
    parser.add_argument("results_dirs", nargs='+', help="List of result directories to compare")
    parser.add_argument("--output", type=str, default="figures/comparison", help="Output directory for plots")
    parser.add_argument(
        "--group-by",
        type=str,
        default="none",
        choices=["none", "model"],
        help="Group comparisons by model id.",
    )
    args = parser.parse_args()
    
    apply_plot_style()
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Loading results from {len(args.results_dirs)} directories...")
    df = load_results(args.results_dirs)
    
    print(f"Found experiments: {df['experiment'].unique()}")
    
    # Define Metrics to Plot
    metrics = [
        ("scaled_rmse", "Scaled RMSE (Lower is Better)"),
        ("coverage", "Coverage (Target: 0.90)"),
        ("coverage_gap", "Coverage Gap (Target: 0)"),
        ("scaled_interval_size", "Interval Width (Lower is Better)"),
        ("winkler", "Winkler Score (Lower is Better)"),
        ("wis", "WIS (Lower is Better)"),
    ]

    metrics = [(col, label) for col, label in metrics if col in df.columns]
    metric_ylims = {col: compute_metric_ylim(df, col) for col, _ in metrics}
    
    if args.group_by == "model":
        groups = sorted(df["model_id"].unique())
        label_col = "calibration"
    else:
        groups = [None]
        label_col = "label"

    dataset_titles = {
        "Eval_Temporal": "Temporal Test (Seen Stations)",
        "Eval_SpatioTemporal": "Spatio-Temporal Test (Unseen Stations)",
    }

    for model_id in groups:
        if model_id is None:
            df_group = df
            suffix = ""
            label_title = "Experiment"
        else:
            df_group = df[df["model_id"] == model_id]
            suffix = f"_{model_id}"
            label_title = f"Calibration ({model_id})"

        if df_group.empty:
            print(f"No results found for model group: {model_id}")
            continue

        for col, label in metrics:
            fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), sharey=True)

            plot_metric(
                df_group,
                "Eval_Temporal",
                col,
                label,
                dataset_titles["Eval_Temporal"],
                axes[0],
                label_col,
                label_title,
                PLOT_COLORS,
            )
            plot_metric(
                df_group,
                "Eval_SpatioTemporal",
                col,
                label,
                dataset_titles["Eval_SpatioTemporal"],
                axes[1],
                label_col,
                label_title,
                PLOT_COLORS,
            )

            for ax in axes:
                ax.set_xlabel("Forecast Horizon (Weeks)")
                if col == "coverage":
                    ax.axhline(0.90, color="#2E2E2E", linestyle="--", linewidth=1.5, label="Target (90%)")
                if col == "coverage_gap":
                    ax.axhline(0.0, color="#2E2E2E", linestyle="--", linewidth=1.5, label="Target (0)")
                ylim = metric_ylims.get(col)
                if ylim is not None:
                    ax.set_ylim(*ylim)

            axes[1].set_ylabel("")
            handles, labels = axes[0].get_legend_handles_labels()
            if not handles:
                handles, labels = axes[1].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                title=label_title,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.0),
                ncol=max(1, min(5, len(labels))),
                frameon=False,
            )
            fig.suptitle(label, fontsize=13, fontweight="semibold")
            fig.tight_layout(rect=[0, 0.1, 1, 0.93])

            save_path = os.path.join(args.output, f"metric_{col}{suffix}.png")
            plt.savefig(save_path)
            print(f"Saved {save_path}")
            plt.close()

if __name__ == "__main__":
    main()
