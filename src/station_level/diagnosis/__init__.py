"""Station-level diagnosis tools.

This module is intentionally scoped to station-level diagnosis only.
It does not train forecasting models.
"""

from station_level.diagnosis.categorization.categorize_stations import (
    assign_station_categories,
)
from station_level.diagnosis.clustering.cluster_stations import cluster_station_summary
from station_level.diagnosis.config import StationDiagnosisConfig
from station_level.diagnosis.features.summary_features import (
    build_station_inventory,
    build_station_summary_table,
)
from station_level.diagnosis.visualization.build_station_visuals import build_station_visuals

__all__ = [
    "StationDiagnosisConfig",
    "assign_station_categories",
    "build_station_inventory",
    "build_station_summary_table",
    "build_station_visuals",
    "cluster_station_summary",
]
