from __future__ import annotations

from pathlib import Path

import pandas as pd

from station_level.diagnosis.categorization import (
    assign_station_categories,
    build_station_category_summary,
)
from station_level.diagnosis.clustering import cluster_station_summary
from station_level.diagnosis.config import StationDiagnosisConfig
from station_level.diagnosis.features.summary_features import (
    build_station_analysis_panel,
    build_station_inventory,
    build_station_summary_table,
)
from station_level.diagnosis.utils import (
    ensure_analysis_directories,
    load_station_daily_data,
    write_dataframe,
)
from station_level.diagnosis.utils.validation import build_validation_artifacts
from station_level.diagnosis.visualization import build_station_visuals


def _top_table(frame: pd.DataFrame, column: str, top_n: int, ascending: bool = False) -> pd.DataFrame:
    clean = frame.copy()
    clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean.replace([float("inf"), float("-inf")], pd.NA)
    return clean.sort_values(column, ascending=ascending, na_position="last").head(top_n).reset_index(drop=True)


def build_station_level_diagnosis(
    input_path: str | Path,
    date_col: str,
    station_col: str,
    target_col: str,
    config: StationDiagnosisConfig | None = None,
) -> dict[str, str]:
    """Build the station-level diagnosis package and persist tables and figures."""

    config = config or StationDiagnosisConfig()
    paths = ensure_analysis_directories(config)
    station_daily = load_station_daily_data(input_path, date_col, station_col, target_col)
    validation_summary, validation_station_warnings = build_validation_artifacts(station_daily, config)

    analysis_panel = build_station_analysis_panel(station_daily, config)
    inventory = build_station_inventory(station_daily, config, analysis_panel=analysis_panel)
    summary = build_station_summary_table(station_daily, inventory, config, analysis_panel=analysis_panel)
    categorized = assign_station_categories(summary, config)
    with_clusters, cluster_profile, cluster_selection = cluster_station_summary(categorized, config)
    category_summary = build_station_category_summary(with_clusters)
    station_labels = with_clusters[
        [
            "station_id",
            "history_group",
            "history_group_reason",
            "behavior_label",
            "behavior_label_confidence",
            "behavior_label_reliability_reason",
            "cluster_label",
            "cluster_assignment_confidence",
            "cluster_eligible",
            "cluster_ineligible_reason",
        ]
    ].copy()
    figure_paths = build_station_visuals(
        analysis_panel,
        with_clusters,
        cluster_profile,
        cluster_selection,
        config,
        paths["figures"],
    )
    figure_manifest = pd.DataFrame(
        [{"artifact_label": label, "artifact_path": path} for label, path in sorted(figure_paths.items())]
    )

    written = {
        "validation_summary": str(write_dataframe(validation_summary, paths["tables"] / "validation_summary.csv")),
        "validation_station_warnings": str(
            write_dataframe(validation_station_warnings, paths["tables"] / "validation_station_warnings.csv")
        ),
        "analysis_panel_parquet": str(write_dataframe(analysis_panel, paths["tables"] / "station_analysis_panel.parquet")),
        "station_inventory": str(write_dataframe(inventory, paths["tables"] / "station_inventory.csv")),
        "station_summary_csv": str(write_dataframe(categorized, paths["tables"] / "station_summary_table.csv")),
        "station_summary_parquet": str(write_dataframe(categorized, paths["tables"] / "station_summary_table.parquet")),
        "station_labels": str(write_dataframe(station_labels, paths["tables"] / "station_labels.csv")),
        "station_category_summary": str(write_dataframe(category_summary, paths["tables"] / "station_category_summary.csv")),
        "station_cluster_profile": str(write_dataframe(cluster_profile, paths["tables"] / "station_cluster_profile.csv")),
        "cluster_model_selection": str(write_dataframe(cluster_selection, paths["tables"] / "cluster_model_selection.csv")),
        "summary_with_clusters": str(write_dataframe(with_clusters, paths["tables"] / "station_summary_with_clusters.csv")),
        "figure_manifest": str(write_dataframe(figure_manifest, paths["tables"] / "figure_manifest.csv")),
        "top_avg_demand": str(
            write_dataframe(_top_table(with_clusters, "avg_demand_observed", config.top_n), paths["tables"] / "top_by_avg_demand.csv")
        ),
        "top_zero_rate": str(
            write_dataframe(_top_table(with_clusters, "zero_rate_observed", config.top_n), paths["tables"] / "top_by_zero_rate.csv")
        ),
        "top_outlier_rate": str(
            write_dataframe(_top_table(with_clusters, "outlier_rate", config.top_n), paths["tables"] / "top_by_outlier_rate.csv")
        ),
        "top_cv": str(
            write_dataframe(
                _top_table(with_clusters, "coefficient_of_variation", config.top_n),
                paths["tables"] / "top_by_coefficient_of_variation.csv",
            )
        ),
    }

    written.update({label: path for label, path in figure_paths.items()})
    return written
