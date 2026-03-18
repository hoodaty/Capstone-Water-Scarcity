import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from cycler import cycler

SPLITS = {
    "Eval_Temporal": "Temporal Test\n(Seen Stations)",
    "Eval_SpatioTemporal": "Spatio-Temporal Test\n(Unseen Stations)",
}
PANEL_LABELS = ["A", "B"]

CALIBRATION_ORDER = ["none", "temp", "mixed", "spatio"]

METRICS = [
    ("scaled_rmse", "Scaled RMSE", "↓ better"),
    ("scaled_mae", "Scaled MAE", "↓ better"),
    ("coverage", "Coverage", "target 0.90"),
    ("coverage_gap", "Coverage Gap", "target 0"),
    ("scaled_interval_size", "Scaled Interval Width", "↓ better"),
    ("wis", "WIS", "↓ better"),
]

# Muted, colorblind-friendly palette
_CALIB_COLORS = {
    "none": "#5778a4",
    "temp": "#e49444",
    "mixed": "#56a64b",
    "spatio": "#d1615d",
}
_FALLBACK_COLORS = [
    "#5778a4",
    "#e49444",
    "#56a64b",
    "#d1615d",
    "#b279a2",
    "#8c613c",
    "#85b6b2",
    "#e15759",
]


def apply_style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.linestyle": "--",
            "grid.alpha": 0.4,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "xtick.bottom": False,
            "legend.fontsize": 9,
            "legend.title_fontsize": 9,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#cccccc",
        }
    )


def infer_calibration(config: dict) -> str:
    if config.get("calib_spatio_only"):
        return "spatio"
    if config.get("calib_stemp"):
        return "mixed"
    if config.get("calib_temp"):
        return "temp"
    return "none"


def load_results(result_dirs: list[str]) -> pd.DataFrame:
    rows = []
    for rdir in result_dirs:
        config_path = os.path.join(rdir, "config.json")
        metrics_path = os.path.join(rdir, "metrics_summary.csv")
        if not os.path.exists(config_path) or not os.path.exists(metrics_path):
            print(f"Skipping {rdir}: missing config.json or metrics_summary.csv")
            continue
        with open(config_path) as f:
            config = json.load(f)
        exp_name = config.get("experiment_name") or os.path.basename(
            os.path.normpath(rdir)
        )
        model_id = config.get("model_id", config.get("model", "unknown"))
        calibration = infer_calibration(config)

        df = pd.read_csv(metrics_path)
        df["experiment"] = exp_name
        df["model_id"] = model_id
        df["calibration"] = calibration
        df["label"] = f"{model_id} | {calibration}"
        rows.append(df)

    if not rows:
        raise ValueError("No valid results found in the provided directories.")
    return pd.concat(rows, ignore_index=True)


def _ylim(df: pd.DataFrame, col: str) -> tuple[float, float] | None:
    if col == "coverage":
        return (0.0, 1.05)

    values = df[col].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None

    if col == "coverage_gap":
        bound = max(0.05, float(np.max(np.abs(values))) * 1.25)
        return (-bound, bound)

    vmin, vmax = float(values.min()), float(values.max())
    pad = max(abs(vmax) * 0.1, 1e-3) if np.isclose(vmin, vmax) else (vmax - vmin) * 0.18
    return (max(0.0, vmin - pad * 0.5), vmax + pad)


def _bar_color(label: str, idx: int, label_col: str) -> str:
    if label_col == "calibration":
        return _CALIB_COLORS.get(label, _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])
    return _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)]


def plot_panel(
    ax,
    df: pd.DataFrame,
    split: str,
    col: str,
    ylabel: str,
    hint: str,
    label_col: str,
    panel_letter: str,
):
    subset = df[df["dataset"] == split].copy()
    title = SPLITS.get(split, split)

    # Panel label + title as a styled header
    ax.set_title(
        f"({panel_letter})  {title}",
        loc="left",
        fontsize=10,
        fontweight="semibold",
        pad=8,
    )

    if subset.empty:
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="gray",
        )
        return

    weeks = sorted(subset["week"].unique())
    labels = subset[label_col].unique().tolist()
    if label_col == "calibration":
        labels = [c for c in CALIBRATION_ORDER if c in labels]

    n = len(labels)
    bar_width = min(0.72 / n, 0.26)
    x = np.arange(len(weeks))
    offsets = (np.arange(n) - (n - 1) / 2) * (bar_width + 0.02)

    for i, label in enumerate(labels):
        values = (
            subset[subset[label_col] == label]
            .set_index("week")
            .reindex(weeks)[col]
            .values
        )
        color = _bar_color(label, i, label_col)
        ax.bar(
            x + offsets[i],
            values,
            width=bar_width,
            label=label,
            color=color,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Forecast horizon", fontsize=9, labelpad=4)
    ax.set_ylabel(f"{ylabel}  ({hint})", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"W{w + 1}" for w in weeks], fontsize=10)
    ax.tick_params(axis="y", labelsize=9)

    # Reference lines
    if col == "coverage":
        ax.axhline(
            0.90,
            color="#333333",
            linestyle="--",
            linewidth=1.2,
            zorder=3,
            label="Target (0.90)",
        )
    if col == "coverage_gap":
        ax.axhline(
            0.0,
            color="#333333",
            linestyle="--",
            linewidth=1.2,
            zorder=3,
            label="Target (0)",
        )

    # Subtle zero-baseline for coverage_gap
    if col == "coverage_gap":
        ax.axhline(0.0, color="#aaaaaa", linewidth=0.5, zorder=2)

    ax.yaxis.set_major_formatter(
        mticker.FormatStrFormatter(
            "%.2f" if col in {"coverage", "coverage_gap"} else "%.1f"
        )
    )


def main():
    parser = argparse.ArgumentParser(description="Plot and compare experiment results.")
    parser.add_argument(
        "results_dirs", nargs="*", help="Result directories to compare."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Auto-discover all experiment subdirectories inside this folder.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/comparison",
        help="Output directory for plots.",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        default="model",
        choices=["model", "all"],
        help="'model': one figure per model. 'all': all experiments in one figure.",
    )
    args = parser.parse_args()

    dirs = list(args.results_dirs)
    if args.results_dir:
        root = args.results_dir.rstrip("/")
        dirs += [
            os.path.join(root, d)
            for d in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, d))
        ]
    if not dirs:
        parser.error(
            "Provide result directories as positional args or via --results-dir."
        )

    apply_style()
    os.makedirs(args.output, exist_ok=True)

    df = load_results(dirs)
    print(
        f"Loaded {len(df['experiment'].unique())} experiment(s): {sorted(df['experiment'].unique())}"
    )

    available = [(col, lbl, hint) for col, lbl, hint in METRICS if col in df.columns]
    ylims = {col: _ylim(df, col) for col, _, _ in available}

    groups = sorted(df["model_id"].unique()) if args.group_by == "model" else [None]
    label_col = "calibration" if args.group_by == "model" else "label"

    for group in groups:
        df_g = df if group is None else df[df["model_id"] == group]
        suffix = f"_{group}" if group else ""
        legend_title = f"Calibration" if group else "Experiment"
        model_subtitle = f" — {group}" if group else ""

        if df_g.empty:
            continue

        for col, ylabel, hint in available:
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(13, 4.8),
                sharey=False,
                gridspec_kw={"wspace": 0.38},
            )

            for ax, (split, _), letter in zip(axes, SPLITS.items(), PANEL_LABELS):
                plot_panel(ax, df_g, split, col, ylabel, hint, label_col, letter)
                lim = ylims.get(col)
                if lim:
                    ax.set_ylim(*lim)

            # Single shared legend, placed to the right of the second panel
            handles, labels = [], []
            for ax in axes:
                h, l = ax.get_legend_handles_labels()
                for handle, lab in zip(h, l):
                    if lab not in labels:
                        handles.append(handle)
                        labels.append(lab)

            fig.legend(
                handles,
                labels,
                title=legend_title,
                loc="center right",
                bbox_to_anchor=(1.01, 0.5),
                frameon=True,
                borderpad=0.8,
            )

            fig.suptitle(
                f"{ylabel}{model_subtitle}",
                fontsize=13,
                fontweight="bold",
                x=0.47,
                y=1.02,
            )
            fig.subplots_adjust(
                left=0.08, right=0.83, top=0.90, bottom=0.13, wspace=0.38
            )

            path = os.path.join(args.output, f"{col}{suffix}.png")
            plt.savefig(path, bbox_inches="tight")
            plt.close()
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
