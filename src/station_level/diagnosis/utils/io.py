from __future__ import annotations

from pathlib import Path

import pandas as pd

from station_level.diagnosis.utils.validation import validate_required_columns


def load_station_daily_data(
    input_path: str | Path,
    date_col: str,
    station_col: str,
    target_col: str,
) -> pd.DataFrame:
    """Load station daily data from CSV or parquet and normalize required columns.

    The loader preserves potentially invalid values so that the validation layer can fail loudly
    instead of silently dropping problematic rows.
    """

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path, low_memory=False)

    validate_required_columns(frame, [date_col, station_col, target_col])
    normalized = frame[[date_col, station_col, target_col]].rename(
        columns={date_col: "date", station_col: "station_id", target_col: "target"}
    )
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized["station_id"] = normalized["station_id"].astype(str).str.strip()
    normalized["target"] = pd.to_numeric(normalized["target"], errors="coerce")
    normalized = normalized.sort_values(["station_id", "date"], na_position="last").reset_index(drop=True)
    return normalized


def write_dataframe(frame: pd.DataFrame, path: Path) -> Path:
    """Write a dataframe to CSV or parquet depending on the suffix."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
    return path
