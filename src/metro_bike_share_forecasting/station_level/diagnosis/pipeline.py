from __future__ import annotations

from pathlib import Path

import pandas as pd

from metro_bike_share_forecasting.station_level.diagnosis.categorization import (
    assign_station_categories,
    build_station_category_summary,
)
from metro_bike_share_forecasting.station_level.diagnosis.clustering import cluster_station_summary
from metro_bike_share_forecasting.station_level.diagnosis.config import StationDiagnosisConfig
from metro_bike_share_forecasting.station_level.diagnosis.features import build_station_summary_table
from metro_bike_share_forecasting.station_level.diagnosis.reports import build_station_markdown_report
from metro_bike_share_forecasting.station_level.diagnosis.utils import (
    ensure_analysis_directories,
    load_station_daily_data,
    write_dataframe,
)


def _top_table(frame: pd.DataFrame, column: str, top_n: int, ascending: bool = False) -> pd.DataFrame:
    return frame.sort_values(column, ascending=ascending).head(top_n).reset_index(drop=True)


def build_station_level_diagnosis(
    input_path: str | Path,
    date_col: str,
    station_col: str,
    target_col: str,
    config: StationDiagnosisConfig | None = None,
) -> dict[str, str]:
    config = config or StationDiagnosisConfig()
    paths = ensure_analysis_directories(config)
    station_daily = load_station_daily_data(input_path, date_col, station_col, target_col)
    summary = build_station_summary_table(station_daily, config)
    categorized = assign_station_categories(summary, config)
    with_clusters, cluster_profile = cluster_station_summary(categorized, config)
    category_summary = build_station_category_summary(with_clusters)

    written = {
        "station_summary_csv": str(write_dataframe(with_clusters, paths["tables"] / "station_summary_table.csv")),
        "station_summary_parquet": str(write_dataframe(with_clusters, paths["tables"] / "station_summary_table.parquet")),
        "station_category_summary": str(write_dataframe(category_summary, paths["tables"] / "station_category_summary.csv")),
        "station_cluster_profile": str(write_dataframe(cluster_profile, paths["tables"] / "station_cluster_profile.csv")),
        "top_avg_demand": str(write_dataframe(_top_table(with_clusters, "avg_demand", config.top_n), paths["tables"] / "top_by_avg_demand.csv")),
        "top_zero_rate": str(write_dataframe(_top_table(with_clusters, "zero_rate", config.top_n), paths["tables"] / "top_by_zero_rate.csv")),
        "top_outlier_rate": str(write_dataframe(_top_table(with_clusters, "outlier_rate", config.top_n), paths["tables"] / "top_by_outlier_rate.csv")),
        "top_cv": str(write_dataframe(_top_table(with_clusters, "coefficient_of_variation", config.top_n), paths["tables"] / "top_by_coefficient_of_variation.csv")),
        "summary_with_clusters": str(write_dataframe(with_clusters, paths["diagnostics"] / "station_summary_with_clusters.csv")),
    }
    report_path = build_station_markdown_report(
        with_clusters,
        category_summary,
        cluster_profile,
        config,
        paths["reports"] / "station_level_diagnosis_summary.md",
    )
    written["report"] = str(report_path)
    return written
