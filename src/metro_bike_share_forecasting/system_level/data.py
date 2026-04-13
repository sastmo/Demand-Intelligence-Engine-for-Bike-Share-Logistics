from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from metro_bike_share_forecasting.system_level.config import SystemLevelConfig


OUTPUT_SUBDIRECTORIES = (
    "forecasts",
    "metrics",
    "backtests",
    "reports",
    "models",
    "feature_artifacts",
    "figures",
)


def ensure_output_directories(config: SystemLevelConfig) -> dict[str, Path]:
    config.output_root.mkdir(parents=True, exist_ok=True)
    directories = {name: config.output_root / name for name in OUTPUT_SUBDIRECTORIES}
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def load_system_level_target(config: SystemLevelConfig) -> pd.DataFrame:
    if config.daily_aggregate_path.exists():
        frame = pd.read_csv(config.daily_aggregate_path, low_memory=False)
        if {"segment_type", "segment_id", config.date_column, config.target_column}.issubset(frame.columns):
            filtered = frame.loc[
                (frame["segment_type"].astype(str) == config.segment_type)
                & (frame["segment_id"].astype(str) == config.segment_id),
                [config.date_column, config.target_column],
            ].copy()
            if not filtered.empty:
                return _finalize_target_frame(filtered, config.date_column, config.target_column)

    if not config.cleaned_trip_path.exists():
        raise FileNotFoundError(
            f"Could not build the system-level target because neither {config.daily_aggregate_path} nor "
            f"{config.cleaned_trip_path} was available."
        )

    cleaned = pd.read_csv(config.cleaned_trip_path, low_memory=False, parse_dates=["start_ts_local"])
    if "start_ts_local" not in cleaned.columns:
        raise ValueError("The cleaned trip file does not contain `start_ts_local`, so the daily target cannot be rebuilt.")
    rebuilt = (
        cleaned.assign(bucket_start=cleaned["start_ts_local"].dt.floor("D"))
        .groupby("bucket_start", as_index=False)
        .size()
        .rename(columns={"size": config.target_column, "bucket_start": config.date_column})
    )
    return _finalize_target_frame(rebuilt, config.date_column, config.target_column)


def _finalize_target_frame(frame: pd.DataFrame, date_column: str, target_column: str) -> pd.DataFrame:
    renamed = frame.rename(columns={date_column: "date", target_column: "target"}).copy()
    renamed["date"] = pd.to_datetime(renamed["date"])
    renamed["target"] = pd.to_numeric(renamed["target"], errors="coerce").fillna(0.0)
    renamed = renamed.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    full_index = pd.date_range(renamed["date"].min(), renamed["date"].max(), freq="D")
    completed = pd.DataFrame({"date": full_index}).merge(renamed, on="date", how="left")
    completed["missing_period_flag"] = completed["target"].isna().astype(int)
    completed["target"] = completed["target"].fillna(0.0)
    completed["series_scope"] = "system_level"
    return completed


def load_external_features(config: SystemLevelConfig) -> pd.DataFrame:
    if config.external_features_path is None or not config.external_features_path.exists():
        return pd.DataFrame(columns=["date"])
    frame = pd.read_csv(config.external_features_path, low_memory=False)
    if config.external_date_column not in frame.columns:
        raise ValueError(
            f"Configured external feature file {config.external_features_path} does not contain "
            f"{config.external_date_column}."
        )
    frame = frame.rename(columns={config.external_date_column: "date"}).copy()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)


def write_dataframe(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def write_json(payload: dict[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
