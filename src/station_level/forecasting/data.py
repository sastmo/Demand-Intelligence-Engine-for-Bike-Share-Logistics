from __future__ import annotations

from pathlib import Path

import pandas as pd

from system_level.common.io import (
    ensure_output_directories as ensure_scope_output_directories,
)
from system_level.common.io import write_dataframe, write_json, write_text
from station_level.diagnosis.categorization import assign_station_categories
from station_level.diagnosis.clustering import cluster_station_summary
from station_level.diagnosis.config import StationDiagnosisConfig
from station_level.diagnosis.features import build_station_inventory, build_station_summary_table
from station_level.forecasting.config import StationLevelForecastConfig

def ensure_output_directories(config: StationLevelForecastConfig) -> dict[str, Path]:
    return ensure_scope_output_directories(config.output_root)


def load_station_forecast_panel(config: StationLevelForecastConfig) -> pd.DataFrame:
    if not config.daily_aggregate_path.exists():
        raise FileNotFoundError(f"Station forecast input not found: {config.daily_aggregate_path}")

    frame = pd.read_csv(config.daily_aggregate_path, low_memory=False)
    required = {
        config.date_column,
        config.station_column,
        config.target_column,
        "segment_type",
        config.in_service_column,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Station forecast input is missing columns: {sorted(missing)}")

    panel = frame.loc[
        frame["segment_type"].astype(str) == config.segment_type,
        [config.date_column, config.station_column, config.target_column, config.in_service_column],
    ].rename(
        columns={
            config.date_column: "date",
            config.station_column: "station_id",
            config.target_column: "raw_target",
            config.in_service_column: "in_service",
        }
    ).copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel["station_id"] = panel["station_id"].astype(str)
    panel["raw_target"] = pd.to_numeric(panel["raw_target"], errors="coerce")
    panel["in_service"] = panel["in_service"].astype(str).str.lower().eq("true")
    panel = panel.dropna(subset=["date", "station_id"]).sort_values(["station_id", "date"]).drop_duplicates(
        subset=["station_id", "date"], keep="last"
    )

    bounded: list[pd.DataFrame] = []
    for _, station_frame in panel.groupby("station_id", sort=True):
        observed = station_frame.loc[station_frame["in_service"]].copy()
        if observed.empty:
            continue
        start_date = observed["date"].min()
        end_date = observed["date"].max()
        bounded.append(station_frame.loc[station_frame["date"].between(start_date, end_date)].copy())

    if not bounded:
        return pd.DataFrame(columns=["date", "station_id", "raw_target", "in_service", "target", "missing_period_flag", "series_scope"])

    station_panel = pd.concat(bounded, ignore_index=True).sort_values(["station_id", "date"]).reset_index(drop=True)
    station_panel["target"] = station_panel["raw_target"].where(station_panel["in_service"])
    station_panel["missing_period_flag"] = (~station_panel["in_service"]).astype(int)
    station_panel["series_scope"] = "station_level"
    return station_panel


def observed_station_daily(panel: pd.DataFrame) -> pd.DataFrame:
    observed = panel.loc[panel["in_service"]].copy()
    return observed[["date", "station_id", "target"]].dropna(subset=["target"]).reset_index(drop=True)


def load_station_slice_lookup(config: StationLevelForecastConfig, panel: pd.DataFrame) -> pd.DataFrame:
    if config.diagnosis_summary_path is not None and config.diagnosis_summary_path.exists():
        summary = pd.read_csv(config.diagnosis_summary_path, low_memory=False)
        if "station_id" in summary.columns:
            columns = [
                column
                for column in [
                    "station_id",
                    "history_group",
                    "station_category",
                    "cluster_label",
                    "is_short_history",
                    "is_zero_almost_always",
                    "appears_active_recently",
                ]
                if column in summary.columns
            ]
            if columns:
                summary = summary[columns].copy()
                summary["station_id"] = summary["station_id"].astype(str)
                return summary.drop_duplicates(subset=["station_id"], keep="last").reset_index(drop=True)

    observed = observed_station_daily(panel)
    diagnosis_config = StationDiagnosisConfig()
    inventory = build_station_inventory(observed, diagnosis_config)
    summary = build_station_summary_table(observed, inventory, diagnosis_config)
    categorized = assign_station_categories(summary, diagnosis_config)
    with_clusters, _, _ = cluster_station_summary(categorized, diagnosis_config)
    columns = [
        "station_id",
        "history_group",
        "station_category",
        "cluster_label",
        "is_short_history",
        "is_zero_almost_always",
        "appears_active_recently",
    ]
    return with_clusters[columns].copy()


def active_station_ids_for_production(panel: pd.DataFrame, config: StationLevelForecastConfig) -> list[str]:
    if panel.empty:
        return []
    global_last_date = pd.to_datetime(panel["date"]).max()
    recent_start = global_last_date - pd.Timedelta(days=config.recent_activity_window_days - 1)
    station_recent_service = (
        panel.loc[(panel["in_service"]) & (panel["date"] >= recent_start)]
        .groupby("station_id")["date"]
        .nunique()
        .reset_index(name="recent_service_days")
    )
    active = station_recent_service.loc[
        station_recent_service["recent_service_days"] >= config.min_recent_service_days, "station_id"
    ].astype(str)
    return sorted(active.tolist())
