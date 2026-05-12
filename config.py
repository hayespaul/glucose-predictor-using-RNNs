"""Dataset loading and pre-processing."""
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent / "glucose.csv"
GLUCOSE_COLUMN = "glucose"
OUTLIER_THRESHOLD_MMOL_L = 40.0


def initialise_dataset(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the glucose CSV and return a cleaned single-column DataFrame.

    The CSV is expected to contain a `glucose` column with readings in mmol/L
    sampled at a fixed interval (5 minutes for the Dexcom G6 used here).

    Pre-processing:
    - Keep only the glucose column.
    - Drop readings above OUTLIER_THRESHOLD_MMOL_L (sensor errors).
    - Forward-fill missing readings with the previous timestep's value.
    """
    df = pd.read_csv(data_path)
    df = df[[GLUCOSE_COLUMN]]
    df = df[df[GLUCOSE_COLUMN] <= OUTLIER_THRESHOLD_MMOL_L]
    df = df.ffill()
    return df.reset_index(drop=True)
