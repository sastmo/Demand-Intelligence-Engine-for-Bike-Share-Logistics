from __future__ import annotations

from pathlib import Path

import pandas as pd

from station_level_analysis.config import StationDiagnosisConfig


def ensure_analysis_directories(config: StationDiagnosisConfig) -> dict[str, Path]:
    """Create the station-level diagnosis output directory and migrate a legacy temp folder if present."""

    legacy_temp = Path("temp")
    system_level_analysis = Path("system_level_analysis")
    if legacy_temp.exists() and not system_level_analysis.exists():
        legacy_temp.rename(system_level_analysis)
    system_level_analysis.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "system_level_analysis": system_level_analysis,
        "station_level_output": config.output_dir,
    }


def load_station_daily_data(
    input_path: str | Path,
    date_col: str,
    station_col: str,
    target_col: str,
    filter_col: str | None = None,
    filter_value: str | None = None,
) -> pd.DataFrame:
    """Load station-level daily data from CSV or parquet and normalize required columns."""

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path, low_memory=False)

    if filter_col is not None:
        if filter_col not in frame.columns:
            raise ValueError(f"Filter column `{filter_col}` was not found in {path}.")
        frame = frame.loc[frame[filter_col].astype(str) == str(filter_value)].copy()

    required = {date_col, station_col, target_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Input data is missing required columns: {sorted(missing)}")

    normalized = frame[[date_col, station_col, target_col]].rename(
        columns={date_col: "date", station_col: "station_id", target_col: "target"}
    )
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized["station_id"] = normalized["station_id"].astype(str)
    normalized["target"] = pd.to_numeric(normalized["target"], errors="coerce")
    normalized = normalized.dropna(subset=["date", "station_id"])
    normalized = (
        normalized.groupby(["date", "station_id"], as_index=False)["target"]
        .sum(min_count=1)
        .sort_values(["station_id", "date"])
        .reset_index(drop=True)
    )
    return normalized


def write_dataframe(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
    return path


def safe_ratio(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return float("nan")
    return float(numerator) / float(denominator)
