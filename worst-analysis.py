import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from analysis import get_point_model, prepare_analysis_data
from pipeline import NUMBER_OF_WEEKS, ALPHA
from src.utils.plots import plot_water_flow_predictions


def predict_with_intervals(
    wrapper, X_test: pd.DataFrame, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return point predictions and (N, 2) interval array [lower, upper].

    Falls back to degenerate point intervals for models without quantile output.
    """
    y_pred = np.asarray(wrapper.predict(X_test, quantiles="mean"))
    try:
        y_pis = np.asarray(
            wrapper.predict(X_test, quantiles=[alpha / 2, 1 - alpha / 2])
        )
        if y_pis.ndim != 2 or y_pis.shape[1] != 2:
            raise ValueError
    except Exception:
        y_pis = np.column_stack([y_pred, y_pred])
    return y_pred, y_pis


def station_rmse(
    test_eval: pd.DataFrame, target_col: str, y_pred: np.ndarray
) -> pd.Series:
    df = test_eval[["station_code", target_col]].copy()
    df["y_pred"] = y_pred
    return (
        df.groupby("station_code")
        .apply(lambda g: float(np.sqrt(mean_squared_error(g[target_col], g["y_pred"]))))
        .sort_values(ascending=False)
    )


def _ensure_obsdate(df: pd.DataFrame) -> pd.DataFrame:
    if "ObsDate" in df.columns:
        return df
    if df.index.name == "ObsDate":
        return df.reset_index()
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Worst-station analysis and prediction plots."
    )
    parser.add_argument(
        "--results-dir", type=str, required=True, help="Results directory."
    )
    parser.add_argument(
        "--week",
        type=int,
        default=0,
        choices=range(NUMBER_OF_WEEKS),
        help="Week index (0-indexed).",
    )
    parser.add_argument(
        "--top-k", type=int, default=3, help="Number of worst stations to plot."
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0.")

    config, wrapper, X_test, _, test_eval, target_col, model_path = (
        prepare_analysis_data(args.results_dir, args.week)
    )

    alpha = float(config.get("alpha", ALPHA))
    model_name = config.get("model", config.get("model_id", "Unknown"))
    print(
        f"Model: {model_name} | Week: {args.week} ({target_col}) | Path: {model_path}"
    )
    print(f"Spatio-temporal test set: {len(X_test)} samples")

    y_pred, y_pis = predict_with_intervals(wrapper, X_test, alpha)
    rmse_by_station = station_rmse(test_eval, target_col, y_pred)
    worst = rmse_by_station.head(min(args.top_k, len(rmse_by_station))).index.tolist()

    print(f"\nTop {len(worst)} worst stations (RMSE):")
    print(rmse_by_station.head(len(worst)).to_string())

    out_dir = Path(args.results_dir) / "figures" / f"week{args.week}" / "worst_stations"
    out_dir.mkdir(parents=True, exist_ok=True)

    for rank, station in enumerate(worst, start=1):
        mask = test_eval["station_code"] == station
        plot_water_flow_predictions(
            ground_truth=_ensure_obsdate(test_eval.loc[mask].copy()),
            prediction=y_pred[mask],
            y_pis=y_pis[mask],
            prefixe=f"week{args.week}_worst_{station}",
            save_path=out_dir / f"week{args.week}_rank{rank}_{station}.png",
            display=False,
            target_col=target_col,
        )
        print(f"Saved rank {rank}: {station}")

    rmse_path = out_dir / f"week{args.week}_station_rmse.csv"
    rmse_by_station.rename("rmse").to_csv(rmse_path)
    print(f"Saved RMSE table: {rmse_path}")


if __name__ == "__main__":
    main()
