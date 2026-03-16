import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from analysis import prepare_analysis_data
from src.utils.plots import plot_water_flow_predictions


DEFAULT_ALPHA = 0.1


def predict_with_intervals(wrapper, X_test: pd.DataFrame, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Return point predictions and [lower, upper] intervals.

    Falls back to degenerate intervals if the model does not expose quantile output.
    """
    y_pred = np.asarray(wrapper.predict(X_test, quantiles="mean"))

    try:
        y_pis = np.asarray(wrapper.predict(X_test, quantiles=[alpha / 2, 1 - alpha / 2]))
        if y_pis.ndim != 2 or y_pis.shape[1] != 2:
            raise ValueError("Unexpected interval prediction shape.")
    except Exception:
        # Keep plotting behavior model-agnostic even for point-only models.
        y_pis = np.column_stack([y_pred, y_pred])

    return y_pred, y_pis


def compute_station_rmse(
    test_eval: pd.DataFrame,
    target_col_name: str,
    y_pred: np.ndarray,
) -> pd.Series:
    results_df = test_eval.copy()
    results_df["y_true"] = results_df[target_col_name].values
    results_df["y_pred"] = y_pred

    station_rmse = results_df.groupby("station_code").apply(
        lambda x: float(np.sqrt(mean_squared_error(x["y_true"], x["y_pred"])))
    )
    return station_rmse.sort_values(ascending=False)


def ensure_obsdate_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure plotting input has an ObsDate column."""
    if "ObsDate" in df.columns:
        return df
    if df.index.name == "ObsDate":
        return df.reset_index()
    return df


def run_worst_station_analysis():
    parser = argparse.ArgumentParser(
        description="Model-agnostic worst-station analysis and plotting"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Path to the results directory (containing config.json and model files)",
    )
    parser.add_argument(
        "--week",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Week index to analyze (0-3). Default: 0",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of worst stations to plot. Default: 3",
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")

    config, wrapper, X_test, _y_test, test_eval, target_col_name, model_path = prepare_analysis_data(
        args.results_dir, args.week
    )

    alpha = float(config.get("alpha", DEFAULT_ALPHA))
    model_name = config.get("model", config.get("model_id", "Unknown"))

    print(f"Loaded configuration from {args.results_dir}")
    print(f"Model: {model_name}")
    print(f"Targeting Week: {args.week} ({target_col_name})")
    print(f"Loading model from {model_path}...")
    print(f"Evaluating on Spatio-Temporal Test Set ({len(X_test)} samples).")

    y_pred, y_pis = predict_with_intervals(wrapper, X_test, alpha)
    station_rmse = compute_station_rmse(test_eval, target_col_name, y_pred)
    worst_stations = station_rmse.head(min(args.top_k, len(station_rmse))).index.tolist()

    print(f"\nTop {len(worst_stations)} Worst Stations (RMSE): {worst_stations}")
    print(station_rmse.head(len(worst_stations)))

    out_dir = os.path.join(args.results_dir, "figures", f"week{args.week}", "worst_stations")
    os.makedirs(out_dir, exist_ok=True)

    for rank, station in enumerate(worst_stations, start=1):
        mask = test_eval["station_code"] == station
        station_data = ensure_obsdate_column(test_eval.loc[mask].copy())
        station_pred = y_pred[mask]
        station_pis = y_pis[mask]

        out_path = os.path.join(
            out_dir,
            f"week{args.week}_worst_rank{rank}_{station}.png",
        )

        plot_water_flow_predictions(
            ground_truth=station_data,
            prediction=station_pred,
            y_pis=station_pis,
            prefixe=f"week{args.week}_worst_{station}",
            save_path=out_path,
            display=False,
            target_col=target_col_name,
        )
        print(f"Saved worst-station plot: {out_path}")

    rmse_path = os.path.join(out_dir, f"week{args.week}_station_rmse.csv")
    station_rmse.rename("rmse").to_csv(rmse_path, index=True)
    print(f"Saved station RMSE table: {rmse_path}")
    print("\nWorst-station analysis complete.")


if __name__ == "__main__":
    run_worst_station_analysis()
