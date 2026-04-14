from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from metro_bike_share_forecasting.station_level.diagnosis.config import StationDiagnosisConfig

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


def _cluster_size_threshold(n_rows: int, config: StationDiagnosisConfig) -> int:
    return max(config.min_cluster_size_abs, int(math.ceil(n_rows * config.min_cluster_size_ratio)))


def _prepare_cluster_matrix(
    station_summary: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    feature_frame = station_summary[feature_columns].copy().replace([np.inf, -np.inf], np.nan)
    feature_frame = feature_frame.dropna(axis=1, how="all")
    if feature_frame.empty:
        raise ValueError("All clustering feature columns were empty after cleaning.")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    imputed = imputer.fit_transform(feature_frame)
    non_constant = np.nanstd(imputed, axis=0) > 0.0
    if not bool(np.any(non_constant)):
        raise ValueError("All clustering feature columns were constant after imputation.")
    feature_frame = feature_frame.loc[:, feature_frame.columns[non_constant]]
    imputed = imputed[:, non_constant]
    transformed = scaler.fit_transform(imputed)
    return feature_frame, transformed


def _evaluate_kmeans_candidates(
    station_summary: pd.DataFrame,
    config: StationDiagnosisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    feature_columns = [column for column in config.cluster_feature_columns if column in station_summary.columns]
    if not feature_columns:
        raise ValueError("No clustering feature columns were available in the station summary table.")
    feature_frame, transformed = _prepare_cluster_matrix(station_summary, feature_columns)
    feature_columns = feature_frame.columns.tolist()
    size_threshold = _cluster_size_threshold(len(station_summary), config)

    rows: list[dict[str, object]] = []
    label_map: dict[int, np.ndarray] = {}
    for k in sorted(set(config.cluster_k_values)):
        if k < 2 or len(station_summary) <= k:
            continue
        model = KMeans(n_clusters=k, n_init=20, random_state=config.random_state)
        labels = model.fit_predict(transformed)
        sizes = pd.Series(labels).value_counts().sort_index()
        tiny_cluster_count = int((sizes < size_threshold).sum())
        silhouette = float(silhouette_score(transformed, labels)) if len(sizes) > 1 else float("nan")
        label_map[k] = labels
        rows.append(
            {
                "candidate_k": int(k),
                "silhouette_score": silhouette,
                "min_cluster_size": int(sizes.min()),
                "max_cluster_size": int(sizes.max()),
                "tiny_cluster_count": tiny_cluster_count,
                "size_threshold": int(size_threshold),
                "all_clusters_valid": bool(tiny_cluster_count == 0),
                "cluster_sizes": ",".join(str(int(size)) for size in sizes.tolist()),
                "feature_columns_used": ",".join(feature_columns),
            }
        )

    selection = pd.DataFrame(rows)
    if selection.empty:
        return selection, pd.DataFrame(), feature_columns

    valid = selection.loc[selection["all_clusters_valid"]].copy()
    if not valid.empty:
        valid["distance_from_preferred_k"] = (valid["candidate_k"] - config.n_clusters).abs()
        selected_row = (
            valid.sort_values(["silhouette_score", "distance_from_preferred_k", "candidate_k"], ascending=[False, True, True])
            .iloc[0]
            .copy()
        )
        selected_row["selection_reason"] = "best_silhouette_without_tiny_clusters"
    else:
        selection["distance_from_preferred_k"] = (selection["candidate_k"] - config.n_clusters).abs()
        selected_row = (
            selection.sort_values(
                ["silhouette_score", "tiny_cluster_count", "distance_from_preferred_k"],
                ascending=[False, True, True],
            )
            .iloc[0]
            .copy()
        )
        selected_row["selection_reason"] = "best_silhouette_despite_tiny_clusters"

    selected_k = int(selected_row["candidate_k"])
    selection["selected"] = selection["candidate_k"] == selected_k
    selection["selection_reason"] = selection["candidate_k"].map(
        lambda value: selected_row["selection_reason"] if int(value) == selected_k else ""
    )
    selection = selection.drop(columns=["distance_from_preferred_k"], errors="ignore")

    label_frame = pd.DataFrame({"station_id": station_summary["station_id"].to_numpy(), "_raw_cluster": label_map[selected_k]})
    return selection.sort_values("candidate_k").reset_index(drop=True), label_frame, feature_columns


def _relabel_clusters(clustered: pd.DataFrame) -> pd.DataFrame:
    profile = (
        clustered.groupby("_raw_cluster", as_index=False)
        .agg(avg_demand_mean=("avg_demand", "mean"), zero_rate_mean=("zero_rate", "mean"))
        .sort_values(["avg_demand_mean", "zero_rate_mean"], ascending=[False, True])
        .reset_index(drop=True)
    )
    mapping = {int(raw_cluster): f"cluster_{index + 1}" for index, raw_cluster in enumerate(profile["_raw_cluster"])}
    relabeled = clustered.copy()
    relabeled["cluster_label"] = relabeled["_raw_cluster"].map(mapping)
    return relabeled


def _cluster_profile(clustered: pd.DataFrame) -> pd.DataFrame:
    if clustered.empty:
        return pd.DataFrame()
    return (
        clustered.groupby("cluster_label", as_index=False)
        .agg(
            station_count=("station_id", "nunique"),
            avg_demand_mean=("avg_demand", "mean"),
            zero_rate_mean=("zero_rate", "mean"),
            cv_mean=("coefficient_of_variation", "mean"),
            lag1_autocorr_mean=("lag1_autocorr", "mean"),
            lag7_autocorr_mean=("lag7_autocorr", "mean"),
            weekday_effect_mean=("weekday_effect_strength", "mean"),
            month_effect_mean=("month_effect_strength", "mean"),
            weekend_ratio_mean=("weekend_ratio", "mean"),
            trend_slope_mean=("trend_slope", "mean"),
            outlier_rate_mean=("outlier_rate", "mean"),
            correlation_with_system_mean=("correlation_with_system_total", "mean"),
            demand_share_mean=("demand_share_of_system", "mean"),
        )
        .sort_values("cluster_label")
        .reset_index(drop=True)
    )


def cluster_station_summary(
    station_summary: pd.DataFrame,
    config: StationDiagnosisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add robust KMeans cluster labels based on station summary features."""

    if station_summary.empty:
        return station_summary.copy(), pd.DataFrame(), pd.DataFrame()

    mature_mask = station_summary["history_group"].eq("mature") if config.cluster_mature_only and "history_group" in station_summary.columns else pd.Series(True, index=station_summary.index)
    mature_summary = station_summary.loc[mature_mask].copy()
    base = station_summary.copy()

    if mature_summary.empty or len(mature_summary) < 4:
        base["cluster_label"] = np.where(
            mature_mask,
            "not_clustered_insufficient_mature_history",
            "not_clustered_short_history",
        )
        selection = pd.DataFrame(
            [
                {
                    "candidate_k": config.n_clusters,
                    "silhouette_score": np.nan,
                    "min_cluster_size": np.nan,
                    "max_cluster_size": np.nan,
                    "tiny_cluster_count": np.nan,
                    "size_threshold": np.nan,
                    "all_clusters_valid": False,
                    "cluster_sizes": "",
                    "feature_columns_used": "",
                    "selected": True,
                    "selection_reason": "insufficient_mature_stations_for_clustering",
                }
            ]
        )
        return base, pd.DataFrame(), selection

    selection, label_frame, _ = _evaluate_kmeans_candidates(mature_summary, config)
    mature_with_labels = mature_summary.merge(label_frame, on="station_id", how="left")
    mature_with_labels = _relabel_clusters(mature_with_labels)
    cluster_profile = _cluster_profile(mature_with_labels)

    with_clusters = base.merge(mature_with_labels[["station_id", "cluster_label"]], on="station_id", how="left")
    with_clusters["cluster_label"] = with_clusters["cluster_label"].fillna("not_clustered_short_history")
    return with_clusters, cluster_profile, selection
