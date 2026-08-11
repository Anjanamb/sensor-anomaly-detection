"""Load the C-MAPSS FD004 turbofan degradation dataset.

FD004 has 6 operating conditions and 2 failure modes. It is the hardest of
the four C-MAPSS subsets. Both train and test are run-to-failure trajectories
recorded per engine:

- train: 249 engines, each run until failure (last recorded cycle == fault).
- test:  248 engines, sensor recordings cut off partway through their life.
         The true remaining useful life at the cut is in RUL_FD004.txt.

Raw file format (space-separated, no header):
    unit  cycle  os1 os2 os3  s1 s2 ... s21
"""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
OP_COND_COLS = ["os1", "os2", "os3"]
ID_COLS = ["unit", "cycle"]
ALL_COLS = ID_COLS + OP_COND_COLS + SENSOR_COLS


def load_fd004(kind: str = "train") -> pd.DataFrame:
    """Load train or test FD004 as a DataFrame with named columns."""
    if kind not in {"train", "test"}:
        raise ValueError(f"kind must be 'train' or 'test', got {kind!r}")
    path = DATA_DIR / f"{kind}_FD004.txt"
    return pd.read_csv(path, sep=r"\s+", header=None, names=ALL_COLS)


def load_rul_fd004() -> pd.Series:
    """True RUL at the last recorded cycle for each test engine.

    Returns a Series of length 248, index 0..247 aligning with test units
    1..248.
    """
    path = DATA_DIR / "RUL_FD004.txt"
    return pd.read_csv(path, header=None, names=["rul"]).squeeze("columns")


def add_rul_train(df: pd.DataFrame) -> pd.DataFrame:
    """Attach true RUL to each row of *training* data.

    Training engines run to failure, so RUL at cycle c is (max_cycle - c).
    Returns a new DataFrame; the original is not mutated.
    """
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    return df.assign(rul=max_cycle - df["cycle"])
