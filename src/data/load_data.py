"""Load the processed Bitstamp daily dataset into a tidy DataFrame."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PathLike = str | Path


def load_bitcoin_data(csv_path: PathLike) -> pd.DataFrame:
    """Load the processed Bitstamp daily CSV.

    The file is expected to have at least the columns
    ``DateTime, Open, High, Low, Close, Volume_(BTC)``.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Path to the processed daily CSV file.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by ``DateTime`` and sorted chronologically.

    Raises
    ------
    FileNotFoundError
        If ``csv_path`` does not exist.
    KeyError
        If the ``DateTime`` column is missing.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}")

    df = pd.read_csv(path)

    if "DateTime" not in df.columns:
        raise KeyError(
            "Expected column 'DateTime' in the processed CSV. "
            f"Columns found: {list(df.columns)}"
        )

    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.set_index("DateTime").sort_index()
    return df
