import os
import argparse
import numpy as np
import pandas as pd

from src.utils.model import split_dataset

# --- CONFIGURATION ---
BASE_DIR_CONF = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR_CONF, "data", "input", "dataset_baseline.csv")
PROCESSED_DIR = os.path.join(BASE_DIR_CONF, "data", "processed")
TIME_VALIDATION = "2000-01-01 00:00:00"
TRAIN_STATION_FRACTION = 0.75

def load_dataset(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath, index_col=0)

def split_chronologically(df, ratio):
    """
    Splits a DataFrame chronologically to prevent time leakage.
    Sorts by Date (index) and splits sequentially.
    """
    if ratio <= 0:
        return pd.DataFrame(), df
        
    # Ensure strict chronological order
    if df.index.name != 'ObsDate' and 'ObsDate' in df.columns:
        df = df.set_index('ObsDate').sort_index()
    else:
        df = df.sort_index()
        
    n_calib = int(len(df) * ratio)
    calib_df = df.iloc[:n_calib]
    eval_df = df.iloc[n_calib:]
    return calib_df, eval_df

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset splits")
    parser.add_argument("--calib-ratio", type=float, default=0.3, help="Ratio of Test sets to reserve for Calibration (Train). Default: 0.3")
    args = parser.parse_args()

    print(f"Creating splits with Calibration Ratio = {args.calib_ratio}")
    print("Loading raw dataset...")
    dataset = load_dataset(INPUT_FILE)
    
    # --- 1. Main Train / Test Split ---
    # This splits the stations and time based on the problem definition
    main_train, test_spatio_temporal, test_temporal = split_dataset(
        dataset, p=TRAIN_STATION_FRACTION, time=TIME_VALIDATION
    )
    
    print(f"Base Split:")
    print(f"  Main Train: {len(main_train)} samples")
    print(f"  Test Temporal (Base): {len(test_temporal)} samples")
    print(f"  Test SpatioTemporal (Base): {len(test_spatio_temporal)} samples")
    
    # Save Main Train
    main_train.to_csv(os.path.join(PROCESSED_DIR, "main_train.csv"))
    print(f"Saved data/processed/main_train.csv")
    
    # --- 2. Temporal Split (Calib/Train vs Eval/Test) ---
    # Sort chronologically before splitting
    test_temporal = test_temporal.sort_index()
    temporal_train, temporal_test = split_chronologically(test_temporal, args.calib_ratio)
    
    temporal_train.to_csv(os.path.join(PROCESSED_DIR, "temporal_train.csv"))
    temporal_test.to_csv(os.path.join(PROCESSED_DIR, "temporal_test.csv"))
    print(f"Saved data/processed/temporal_train.csv ({len(temporal_train)} samples)")
    print(f"Saved data/processed/temporal_test.csv ({len(temporal_test)} samples)")
    
    # --- 3. SpatioTemporal Split (Calib/Train vs Eval/Test) ---
    test_spatio_temporal = test_spatio_temporal.sort_index()
    spatiotemporal_train, spatiotemporal_test = split_chronologically(test_spatio_temporal, args.calib_ratio)
    
    spatiotemporal_train.to_csv(os.path.join(PROCESSED_DIR, "spatiotemporal_train.csv"))
    spatiotemporal_test.to_csv(os.path.join(PROCESSED_DIR, "spatiotemporal_test.csv"))
    print(f"Saved data/processed/spatiotemporal_train.csv ({len(spatiotemporal_train)} samples)")
    print(f"Saved data/processed/spatiotemporal_test.csv ({len(spatiotemporal_test)} samples)")
    
    print("\nData preparation complete.")

if __name__ == "__main__":
    main()
