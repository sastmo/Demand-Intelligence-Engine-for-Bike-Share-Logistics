"""Station-level diagnosis tools.

This module is intentionally scoped to station-level diagnosis only.
It does not train forecasting models.
"""

from diagnosis.station_level_analysis.categorization.categorize_stations import assign_station_categories
from diagnosis.station_level_analysis.clustering.cluster_stations import cluster_station_summary
from diagnosis.station_level_analysis.config import StationDiagnosisConfig
from diagnosis.station_level_analysis.features.summary_features import build_station_summary_table

__all__ = [
    "StationDiagnosisConfig",
    "assign_station_categories",
    "build_station_summary_table",
    "cluster_station_summary",
]
