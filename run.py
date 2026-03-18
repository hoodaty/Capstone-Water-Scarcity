import argparse
import datetime
import json
import os
import re
from math import sqrt

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models.registry import get_model_spec, list_model_ids
from pipeline import (
    ALPHA,
    BASE_DIR,
    PROCESSED_DIR,
    TEST_FILE,
    TRAIN_FILE,
    TARGET_COLS,
    build_features,
    load_split,
)
from src.utils.model import (
    get_station_stats,
    standardize_prediction_intervals,
    standardize_values,
    summarize_metrics,
    wis_score,
)


def compute_per_station_metrics(
    y_true_std: np.ndarray,
    y_pred_std: np.ndarray,
    stations: np.ndarray,
    y_pred_lower_std: np.ndarray,
    y_pred_upper_std: np.ndarray,
    alpha: float,
) -> pd.DataFrame:
    records = []

    for s in np.unique(stations):
        idx = stations == s
        if idx.sum() == 0:
            continue

        yt, yp = y_true_std[idx], y_pred_std[idx]
        yl, yu = y_pred_lower_std[idx], y_pred_upper_std[idx]

        rmse_s = sqrt(mean_squared_error(yt, yp))
        mae_s = mean_absolute_error(yt, yp)

        # sigma from 95% PI width; clamped to avoid log(0)
        sigma = np.maximum((yu - yl) / 3.29, 0.5)
        nll = np.mean(np.log(sigma) + np.abs(yt - yp) / (2 * sigma))

        coverage = np.mean((yt >= yl) & (yt <= yu))

        records.append(
            {
                "station_code": s,
                "scaled_rmse": rmse_s,
                "scaled_mae": mae_s,
                "coverage": coverage,
                "scaled_interval_size": np.mean(yu - yl),
                "log_likelihood": nll,
                "wis": np.mean(wis_score(yt, yl, yu, yp, alpha)),
                "coverage_gap": coverage - (1 - alpha),
            }
        )

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(
        description="Train quantile model with CQR calibration."
    )
    parser.add_argument(
        "--calib-temp",
        action="store_true",
        help="Calibrate using the temporal calib split.",
    )
    parser.add_argument(
        "--calib-stemp",
        action="store_true",
        help="Calibrate using temporal + spatiotemporal calib splits.",
    )
    parser.add_argument(
        "--calib-spatio-only",
        action="store_true",
        help="Calibrate using only the spatiotemporal calib split.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lgbm_cqr",
        help=f"Model id. Options: {', '.join(list_model_ids())}",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Experiment folder name (defaults to timestamp).",
    )
    args = parser.parse_args()

    if args.calib_spatio_only and (args.calib_temp or args.calib_stemp):
        raise ValueError(
            "Choose one calibration mode: --calib-temp, --calib-stemp, or --calib-spatio-only."
        )
    if args.calib_stemp:
        args.calib_temp = True

    model_spec = get_model_spec(args.model)
    print(
        f"Model={model_spec.id} | "
        f"calib_temp={args.calib_temp} | "
        f"calib_stemp={args.calib_stemp} | "
        f"calib_spatio_only={args.calib_spatio_only}"
    )

    folder = (
        re.sub(r"[^a-zA-Z0-9_\-]", "_", args.name)
        if args.name
        else datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    results_dir = os.path.join(BASE_DIR, "results", folder)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results: {results_dir}")

    if not os.path.exists(PROCESSED_DIR):
        raise FileNotFoundError(
            f"Processed data not found at {PROCESSED_DIR}. Run preprocess.py first."
        )

    train_df = load_split("main_train")

    calib_spatio_enabled = args.calib_stemp or args.calib_spatio_only
    calib_temp_df = load_split("temporal_calib") if args.calib_temp else pd.DataFrame()
    calib_spatio_df = (
        load_split("spatiotemporal_calib") if calib_spatio_enabled else pd.DataFrame()
    )
    eval_temp_df = load_split("temporal_test")
    eval_spatio_df = load_split("spatiotemporal_test")

    # Global stats across train + test for consistent per-station metric normalisation,
    # including unseen (spatiotemporal) stations that are absent from the training split.
    # Used only for scaling reported metrics, not in model training.
    dataset_full = pd.concat(
        [
            pd.read_csv(TRAIN_FILE, index_col=0),
            pd.read_csv(TEST_FILE, index_col=0),
        ],
        ignore_index=False,
    )
    global_stats = {
        col: get_station_stats(
            dataset_full[col].values, dataset_full["station_code"].values
        )
        for col in TARGET_COLS
    }

    print(
        f"Train: {len(train_df)} | "
        f"Calib temp: {len(calib_temp_df)} | Calib spatio: {len(calib_spatio_df)} | "
        f"Eval temp: {len(eval_temp_df)} | Eval spatio: {len(eval_spatio_df)}"
    )

    X_train = build_features(train_df)
    train_cols = X_train.columns.tolist()

    X_calib_temp = build_features(calib_temp_df, reference_columns=train_cols)
    X_calib_spatio = build_features(calib_spatio_df, reference_columns=train_cols)
    X_eval_temp = build_features(eval_temp_df, reference_columns=train_cols)
    X_eval_spatio = build_features(eval_spatio_df, reference_columns=train_cols)

    # --- Training & Calibration ---
    models = {}
    calibrated_any = False

    for i, target_col in enumerate(TARGET_COLS):
        print(f"\n[{target_col}] Training...")
        model = model_spec.cls(alpha=ALPHA, **model_spec.init_kwargs)
        model.fit(X_train.values, train_df[target_col].values)
        models[i] = model

        Xs, ys = [], []
        if args.calib_temp:
            Xs.append(X_calib_temp)
            ys.append(calib_temp_df[target_col].values)
        if calib_spatio_enabled:
            Xs.append(X_calib_spatio)
            ys.append(calib_spatio_df[target_col].values)

        if Xs:
            model.calibrate(pd.concat(Xs).values, np.concatenate(ys))
            calibrated_any = True
            print(
                f"  Calibrated on {sum(len(x) for x in Xs)} samples. q_score={model.q_score:.3f}"
            )
        else:
            print("  No calibration.")

        joblib.dump(model, os.path.join(results_dir, f"model_week{i}.joblib"))

    model_name = (
        model_spec.calibrated_name if calibrated_any else model_spec.display_name
    )

    config = {
        "model": model_name,
        "model_id": model_spec.id,
        "calib_temp": args.calib_temp,
        "calib_stemp": args.calib_stemp,
        "calib_spatio_only": args.calib_spatio_only,
        "alpha": ALPHA,
        "weeks": len(TARGET_COLS),
    }
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # --- Evaluation ---
    print("\n--- Evaluation ---")
    all_metrics, all_summaries = [], []

    def evaluate_split(
        X: pd.DataFrame, source_df: pd.DataFrame, split_name: str
    ) -> None:
        print(f"\n[{split_name}]")
        stations = source_df["station_code"].values

        for i, target_col in enumerate(TARGET_COLS):
            station_stats = (
                global_stats[target_col].reindex(np.unique(stations)).fillna(1.0)
            )

            y_true = source_df[target_col].values
            y_pred = models[i].predict(X.values, quantiles="mean")
            y_q = models[i].predict(X.values, quantiles=[ALPHA / 2, 1 - ALPHA / 2])

            y_true_std = standardize_values(y_true, stations, station_stats)
            y_pred_std = standardize_values(y_pred, stations, station_stats)
            y_lower_std, y_upper_std = standardize_prediction_intervals(
                np.column_stack([y_q[:, 0], y_q[:, 1]]), stations, station_stats
            )

            metrics_df = compute_per_station_metrics(
                y_true_std, y_pred_std, stations, y_lower_std, y_upper_std, ALPHA
            )
            metrics_df["dataset"] = split_name
            metrics_df["week"] = i
            metrics_df["target_col"] = target_col
            all_metrics.append(metrics_df)

            summary_df = summarize_metrics(
                metrics_df, model_name, f"{split_name}_wk{i}", alpha=ALPHA
            )
            summary_df["week"] = i
            summary_df["dataset"] = split_name
            all_summaries.append(summary_df)
            print(summary_df.to_string(index=False))

    evaluate_split(X_eval_temp, eval_temp_df, "Eval_Temporal")
    evaluate_split(X_eval_spatio, eval_spatio_df, "Eval_SpatioTemporal")

    summary = pd.concat(all_summaries, ignore_index=True)
    lead_cols = ["model", "dataset", "week"]
    summary = summary[lead_cols + [c for c in summary.columns if c not in lead_cols]]

    agg_cols = [
        "scaled_rmse",
        "scaled_mae",
        "coverage",
        "coverage_gap",
        "scaled_interval_size",
        "wis",
    ]
    agg_cols = [c for c in agg_cols if c in summary.columns]
    overall = (
        summary.groupby("dataset")[agg_cols]
        .mean()
        .round(4)
        .reset_index()
        .assign(model=model_name)
    )
    overall = overall[["model", "dataset"] + agg_cols]

    pd.concat(all_metrics, ignore_index=True).to_csv(
        os.path.join(results_dir, "metrics_per_station.csv"),
        index=False,
        float_format="%.4f",
    )
    summary.to_csv(
        os.path.join(results_dir, "metrics_summary.csv"),
        index=False,
        float_format="%.4f",
    )
    overall.to_csv(
        os.path.join(results_dir, "metrics_overall.csv"),
        index=False,
        float_format="%.4f",
    )

    print("\n--- Overall (mean across weeks) ---")
    print(overall.to_string(index=False))
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
