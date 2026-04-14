from __future__ import annotations

import os

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from diagnosis.station_level_analysis.config import StationDiagnosisConfig

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


def cluster_station_summary(
    station_summary: pd.DataFrame,
    config: StationDiagnosisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add KMeans cluster labels based on station summary features and build cluster profiles."""

    if station_summary.empty:
        return station_summary.copy(), pd.DataFrame()

    feature_columns = [column for column in config.cluster_feature_columns if column in station_summary.columns]
    if not feature_columns:
        raise ValueError("No clustering feature columns were available in the station summary table.")

    feature_frame = station_summary[feature_columns].copy()
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=config.n_clusters, n_init=20, random_state=config.random_state)),
        ]
    )
    labels = pipeline.fit_predict(feature_frame)

    clustered = station_summary.copy()
    clustered["cluster_label"] = [f"cluster_{label}" for label in labels]

    cluster_profile = (
        clustered.groupby("cluster_label", as_index=False)
        .agg(
            station_count=("station_id", "nunique"),
            avg_demand_mean=("avg_demand", "mean"),
            zero_rate_mean=("zero_rate", "mean"),
            cv_mean=("coefficient_of_variation", "mean"),
            weekday_effect_mean=("weekday_effect_strength", "mean"),
            outlier_rate_mean=("outlier_rate", "mean"),
            correlation_with_system_mean=("correlation_with_system_total", "mean"),
            demand_share_mean=("demand_share_of_system", "mean"),
        )
        .sort_values("cluster_label")
        .reset_index(drop=True)
    )
    return clustered, cluster_profile
