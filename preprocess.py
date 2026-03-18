"""Build the five processed splits from raw input files.

Outputs written to data/processed/:
    main_train.csv             — training set (seen stations, pre-cutoff)
    temporal_calib.csv         — CQR calibration (seen stations, post-cutoff)
    spatiotemporal_calib.csv   — CQR calibration (unseen stations, post-cutoff)
    temporal_test.csv          — evaluation (seen stations, from dataset_test)
    spatiotemporal_test.csv    — evaluation (unseen stations, from dataset_test)
"""

from pathlib import Path

import pandas as pd

from pipeline import PROCESSED_DIR, TEST_FILE, TRAIN_FILE
from src.utils.model import split_dataset

OUT_DIR = Path(PROCESSED_DIR)

TIME_CUTOFF = "2000-01-01 00:00:00"
TRAIN_STATION_FRACTION = 0.75


def _sort(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.name != "ObsDate" and "ObsDate" in df.columns:
        return df.set_index("ObsDate").sort_index()
    return df.sort_index()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training data...")
    dataset_train = pd.read_csv(TRAIN_FILE, index_col=0)

    main_train, spatiotemporal_calib, temporal_calib = split_dataset(
        dataset_train, p=TRAIN_STATION_FRACTION, time=TIME_CUTOFF
    )
    main_train, temporal_calib, spatiotemporal_calib = (
        _sort(main_train),
        _sort(temporal_calib),
        _sort(spatiotemporal_calib),
    )

    print(f"  main_train:           {len(main_train):>6} rows")
    print(f"  temporal_calib:       {len(temporal_calib):>6} rows")
    print(f"  spatiotemporal_calib: {len(spatiotemporal_calib):>6} rows")

    main_train.to_csv(OUT_DIR / "main_train.csv")
    temporal_calib.to_csv(OUT_DIR / "temporal_calib.csv")
    spatiotemporal_calib.to_csv(OUT_DIR / "spatiotemporal_calib.csv")

    print("\nLoading test data...")
    dataset_test = pd.read_csv(TEST_FILE, index_col=0)
    seen_stations = set(main_train["station_code"].unique())

    temporal_test = _sort(
        dataset_test[dataset_test["station_code"].isin(seen_stations)]
    )
    spatiotemporal_test = _sort(
        dataset_test[~dataset_test["station_code"].isin(seen_stations)]
    )

    print(f"  temporal_test:        {len(temporal_test):>6} rows")
    print(f"  spatiotemporal_test:  {len(spatiotemporal_test):>6} rows")

    temporal_test.to_csv(OUT_DIR / "temporal_test.csv")
    spatiotemporal_test.to_csv(OUT_DIR / "spatiotemporal_test.csv")

    print(f"\nAll splits saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
