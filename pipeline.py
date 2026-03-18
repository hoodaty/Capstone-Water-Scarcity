"""Shared data pipeline: constants, feature engineering, and split loading."""

import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = os.path.join(BASE_DIR, "data", "input", "dataset_train.csv")
TEST_FILE = os.path.join(BASE_DIR, "data", "input", "dataset_test.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

NUMBER_OF_WEEKS = 4
ALPHA = 0.1
USE_EWMA = False

METEO_LAG_WINDOWS = []
METEO_LAG_COLUMNS = [
    "precipitations",
    "temperatures",
    "soil_moisture",
    "evaporation",
    "precipitation_region",
    "temperature_region",
    "soil_moisture_region",
    "evaporation_region",
    "precipitation_zone",
    "temperature_zone",
    "soil_moisture_zone",
    "evaporation_zone",
    "precipitation_sector",
    "temperature_sector",
    "soil_moisture_sector",
    "evaporation_sector",
    "precipitation_sub_sector",
    "temperature_sub_sector",
    "soil_moisture_sub_sector",
    "evaporation_sub_sector",
]

TARGET_COLS = [f"water_flow_week{i + 1}" for i in range(NUMBER_OF_WEEKS)]
FORBIDDEN_COLS = ["water_flow_lag_1w", "water_flow_lag_2w"]
FEATURE_DROP_COLS = ["station_code"] + TARGET_COLS + FORBIDDEN_COLS


def add_meteo_lags(
    df: pd.DataFrame,
    columns: list[str],
    windows: list[int],
    use_ewm: bool = False,
) -> pd.DataFrame:
    df = df.copy()
    if df.index.name is None:
        df.index.name = "ObsDate"
    df = df.sort_values(["station_code", df.index.name])
    available = [c for c in columns if c in df.columns]

    new_features = []
    for col in available:
        agg = "sum" if "precip" in col.lower() or col == "tp" else "mean"
        for w in windows:
            if use_ewm:
                name = f"{col}_ewm_{w}w_mean"
                series = df.groupby("station_code")[col].transform(
                    lambda s: s.ewm(span=w, min_periods=1).mean()
                )
            else:
                name = f"{col}_roll_{w}w_{agg}"
                series = df.groupby("station_code")[col].transform(
                    lambda s: s.rolling(w, min_periods=1).agg(agg)
                )
            series.name = name
            new_features.append(series)

    if new_features:
        df = pd.concat([df] + new_features, axis=1)
    return df


def build_features(
    df: pd.DataFrame,
    reference_columns: list[str] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return (
            pd.DataFrame(columns=reference_columns)
            if reference_columns
            else pd.DataFrame()
        )

    X = df.drop(columns=FEATURE_DROP_COLS, errors="ignore")
    X = pd.get_dummies(X, drop_first=True)

    if reference_columns is not None:
        X = X.reindex(columns=reference_columns, fill_value=0)
    return X


def load_split(name: str) -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, f"{name}.csv")
    df = pd.read_csv(path, index_col=0)
    return add_meteo_lags(df, METEO_LAG_COLUMNS, METEO_LAG_WINDOWS, use_ewm=USE_EWMA)
