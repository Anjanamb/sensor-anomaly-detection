"""
Data loader for NASA C-MAPSS Turbofan Engine Degradation dataset.

Each file contains run-to-failure sensor readings for multiple engine units.
"""

import os
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Column definitions for C-MAPSS dataset
COLUMN_NAMES = (
    ["unit_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

DATA_DIR = Path(__file__).parent.parent / "data"


def load_cmapss(
    subset: str = "FD001",
    data_dir: Path = DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load a C-MAPSS subset (FD001-FD004).

    Returns:
        train_df: Training data (run-to-failure)
        test_df: Test data (truncated before failure)
        rul_df: True Remaining Useful Life for test data
    """
    train_path = data_dir / f"train_{subset}.txt"
    test_path = data_dir / f"test_{subset}.txt"
    rul_path = data_dir / f"RUL_{subset}.txt"

    for path in [train_path, test_path, rul_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Download the dataset first:\n"
                f"  python src/data_loader.py --download"
            )

    train_df = pd.read_csv(
        train_path, sep=r"\s+", header=None, names=COLUMN_NAMES
    )
    test_df = pd.read_csv(
        test_path, sep=r"\s+", header=None, names=COLUMN_NAMES
    )
    rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["rul"])

    logger.info(
        f"Loaded {subset}: train={len(train_df)} rows, "
        f"test={len(test_df)} rows, {rul_df.shape[0]} engines"
    )

    return train_df, test_df, rul_df


def add_rul_to_train(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Remaining Useful Life (RUL) column to training data.
    For each engine unit, RUL = max_cycle - current_cycle.
    """
    df = df.copy()
    max_cycles = df.groupby("unit_id")["cycle"].max().reset_index()
    max_cycles.columns = ["unit_id", "max_cycle"]
    df = df.merge(max_cycles, on="unit_id")
    df["rul"] = df["max_cycle"] - df["cycle"]
    df.drop("max_cycle", axis=1, inplace=True)
    return df


def create_anomaly_labels(
    df: pd.DataFrame, threshold: int = 30
) -> pd.DataFrame:
    """
    Create binary anomaly labels based on RUL threshold.
    Anomaly = 1 when RUL <= threshold (engine approaching failure).
    """
    df = df.copy()
    df["anomaly"] = (df["rul"] <= threshold).astype(int)
    logger.info(
        f"Anomaly labels: {df['anomaly'].sum()} anomalous "
        f"({df['anomaly'].mean():.1%}) out of {len(df)} samples"
    )
    return df


def get_sensor_columns(df: pd.DataFrame) -> list[str]:
    """Return list of sensor column names."""
    return [col for col in df.columns if col.startswith("sensor_")]


def get_op_setting_columns(df: pd.DataFrame) -> list[str]:
    """Return list of operational setting column names."""
    return [col for col in df.columns if col.startswith("op_setting_")]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="C-MAPSS Data Loader")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Print download instructions for the dataset",
    )
    parser.add_argument(
        "--subset",
        default="FD001",
        choices=["FD001", "FD002", "FD003", "FD004"],
        help="Which subset to load",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.download:
        print(
            "\n📥 Download the NASA C-MAPSS dataset:\n"
            "   1. Go to: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data\n"
            "   2. Extract files into the data/ directory\n"
            "   3. You should have: train_FD001.txt, test_FD001.txt, RUL_FD001.txt, etc.\n"
        )
    else:
        train_df, test_df, rul_df = load_cmapss(args.subset)
        train_df = add_rul_to_train(train_df)
        train_df = create_anomaly_labels(train_df)
        print(train_df.head())
        print(f"\nShape: {train_df.shape}")
        print(f"Engines: {train_df['unit_id'].nunique()}")
