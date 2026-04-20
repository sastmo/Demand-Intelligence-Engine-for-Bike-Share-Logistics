from __future__ import annotations

import math
import os
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import RobustScaler

from station_level.diagnosis.config import StationDiagnosisConfig

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


def _cluster_size_threshold(n_rows: int, config: StationDiagnosisConfig) -> int:
    return max(config.min_cluster_size_abs, int(math.ceil(n_rows * config.min_cluster_size_ratio)))


def _prepare_cluster_matrix(
    station_summary: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    feature_frame = station_summary[feature_columns].copy().replace([np.inf, -np.inf], np.nan)
    feature_frame = feature_frame.dropna(axis=1, how="all")
    if feature_frame.empty:
        raise ValueError("All clustering feature columns were empty after cleaning.")

    transformed = feature_frame.copy()
    transformed_columns: list[str] = []
    for column in transformed.columns:
        series = pd.to_numeric(transformed[column], errors="coerce")
        finite = series.dropna()
        if finite.empty:
            continue
        if finite.min() >= 0 and abs(float(finite.skew())) > 1.0:
            transformed[column] = np.log1p(series)
            transformed_columns.append(f"{column}:log1p")
        else:
            transformed[column] = series
            transformed_columns.append(f"{column}:raw")

    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(transformed)
    non_constant = np.nanstd(imputed, axis=0) > 0.0
    if not bool(np.any(non_constant)):
        raise ValueError("All clustering feature columns were constant after imputation.")
    transformed = transformed.loc[:, transformed.columns[non_constant]]
    imputed = imputed[:, non_constant]
    scaler = RobustScaler()
    scaled = scaler.fit_transform(imputed)
    transformed_columns = [column for keep, column in zip(non_constant, transformed_columns) if keep]
    return transformed, scaled, transformed_columns


def _mean_pairwise_ari(label_sets: list[np.ndarray]) -> float:
    if len(label_sets) < 2:
        return float("nan")
    scores = [adjusted_rand_score(left, right) for left, right in combinations(label_sets, 2)]
    return float(np.mean(scores)) if scores else float("nan")


def _bootstrap_stability(
    transformed: np.ndarray,
    reference_labels: np.ndarray,
    k: int,
    config: StationDiagnosisConfig,
) -> float:
    rng = np.random.RandomState(config.random_state + k)
    n_rows = len(transformed)
    sample_size = max(k * 2, int(round(n_rows * config.cluster_bootstrap_fraction)))
    scores: list[float] = []
    for iteration in range(config.cluster_bootstrap_iterations):
        indices = rng.choice(n_rows, size=sample_size, replace=True)
        model = KMeans(n_clusters=k, n_init=20, random_state=config.random_state + k + iteration)
        model.fit(transformed[indices])
        predicted = model.predict(transformed)
        scores.append(float(adjusted_rand_score(reference_labels, predicted)))
    return float(np.mean(scores)) if scores else float("nan")


def _evaluate_kmeans_candidates(
    station_summary: pd.DataFrame,
    config: StationDiagnosisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    feature_columns = [column for column in config.cluster_feature_columns if column in station_summary.columns]
    if not feature_columns:
        raise ValueError("No clustering feature columns were available in the station summary table.")
    feature_frame, transformed, transformed_feature_columns = _prepare_cluster_matrix(station_summary, feature_columns)
    feature_columns = feature_frame.columns.tolist()
    size_threshold = _cluster_size_threshold(len(station_summary), config)

    rows: list[dict[str, object]] = []
    best_models: dict[int, tuple[KMeans, np.ndarray, float]] = {}
    for k in sorted(set(config.cluster_k_values)):
        if k < 2 or len(station_summary) <= k:
            continue

        labels_by_seed: list[np.ndarray] = []
        silhouettes: list[float] = []
        min_sizes: list[int] = []
        max_sizes: list[int] = []
        tiny_counts: list[int] = []
        model_results: list[tuple[KMeans, np.ndarray, float]] = []
        for seed in config.cluster_random_seeds:
            model = KMeans(n_clusters=k, n_init=20, random_state=seed)
            labels = model.fit_predict(transformed)
            sizes = pd.Series(labels).value_counts().sort_index()
            tiny_cluster_count = int((sizes < size_threshold).sum())
            silhouette = float(silhouette_score(transformed, labels)) if len(sizes) > 1 else float("nan")
            labels_by_seed.append(labels)
            silhouettes.append(silhouette)
            min_sizes.append(int(sizes.min()))
            max_sizes.append(int(sizes.max()))
            tiny_counts.append(tiny_cluster_count)
            model_results.append((model, labels, silhouette))

        best_model, best_labels, best_silhouette = max(model_results, key=lambda item: (item[2], -item[0].inertia_))
        best_models[k] = (best_model, best_labels, best_silhouette)
        seed_stability = _mean_pairwise_ari(labels_by_seed)
        bootstrap_stability = _bootstrap_stability(transformed, best_labels, k, config)
        sizes = pd.Series(best_labels).value_counts().sort_index()
        rows.append(
            {
                "candidate_k": int(k),
                "silhouette_score": float(np.nanmean(silhouettes)),
                "silhouette_score_std": float(np.nanstd(silhouettes)),
                "seed_stability_ari": seed_stability,
                "bootstrap_stability_ari": bootstrap_stability,
                "min_cluster_size": int(sizes.min()),
                "max_cluster_size": int(sizes.max()),
                "tiny_cluster_count": int((sizes < size_threshold).sum()),
                "size_threshold": int(size_threshold),
                "all_clusters_valid": bool((sizes >= size_threshold).all()),
                "cluster_sizes": ",".join(str(int(size)) for size in sizes.tolist()),
                "feature_columns_used": ",".join(feature_columns),
                "feature_transformations": ",".join(transformed_feature_columns),
            }
        )

    selection = pd.DataFrame(rows)
    if selection.empty:
        return selection, pd.DataFrame(), feature_columns

    selection["distance_from_preferred_k"] = (selection["candidate_k"] - config.n_clusters).abs()
    selection = selection.sort_values(
        [
            "all_clusters_valid",
            "bootstrap_stability_ari",
            "seed_stability_ari",
            "silhouette_score",
            "distance_from_preferred_k",
            "candidate_k",
        ],
        ascending=[False, False, False, False, True, True],
    ).reset_index(drop=True)
    selected_row = selection.iloc[0].copy()
    if bool(selected_row["all_clusters_valid"]):
        selected_row["selection_reason"] = "best_balance_of_valid_cluster_sizes_stability_and_silhouette"
    else:
        selected_row["selection_reason"] = "selected_despite_size_issues_due_to_limited_valid_candidates"

    selected_k = int(selected_row["candidate_k"])
    selection["selected"] = selection["candidate_k"] == selected_k
    selection["selection_reason"] = selection["candidate_k"].map(
        lambda value: selected_row["selection_reason"] if int(value) == selected_k else ""
    )
    selection = selection.sort_values("candidate_k").drop(columns=["distance_from_preferred_k"]).reset_index(drop=True)

    model, labels, _ = best_models[selected_k]
    distance_matrix = model.transform(transformed)
    nearest = np.partition(distance_matrix, 1, axis=1)[:, :2]
    margin = (nearest[:, 1] - nearest[:, 0]) / np.maximum(nearest[:, 1], 1e-9)
    label_frame = pd.DataFrame(
        {
            "station_id": station_summary["station_id"].to_numpy(),
            "_raw_cluster": labels,
            "_distance_margin": margin,
            "_model_bootstrap_stability": float(selected_row["bootstrap_stability_ari"]),
            "_model_seed_stability": float(selected_row["seed_stability_ari"]),
        }
    )
    return selection, label_frame, feature_columns


def _cluster_assignment_confidence(margin: float, model_stability: float, config: StationDiagnosisConfig) -> str:
    if pd.isna(margin) or pd.isna(model_stability):
        return "low"
    if margin >= 0.35 and model_stability >= config.cluster_stability_good_threshold:
        return "high"
    if margin >= 0.15 and model_stability >= config.cluster_stability_ok_threshold:
        return "medium"
    return "low"


def _relabel_clusters(clustered: pd.DataFrame) -> pd.DataFrame:
    profile = (
        clustered.groupby("_raw_cluster", as_index=False)
        .agg(avg_demand_mean=("avg_demand_observed", "mean"), zero_rate_mean=("zero_rate_observed", "mean"))
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
            avg_demand_mean=("avg_demand_observed", "mean"),
            zero_rate_mean=("zero_rate_observed", "mean"),
            coverage_ratio_mean=("coverage_ratio", "mean"),
            cv_mean=("coefficient_of_variation", "mean"),
            lag1_autocorr_mean=("lag1_autocorr", "mean"),
            lag7_autocorr_mean=("lag7_autocorr", "mean"),
            weekday_effect_mean=("weekday_effect_strength", "mean"),
            month_effect_mean=("month_effect_strength", "mean"),
            weekend_ratio_mean=("weekend_ratio", "mean"),
            trend_slope_mean=("trend_slope", "mean"),
            outlier_rate_mean=("outlier_rate", "mean"),
            correlation_with_system_mean=("correlation_with_system_excl_self", "mean"),
            demand_share_mean=("demand_share_of_system_observed_window", "mean"),
            behavior_mode=("behavior_label", lambda values: pd.Series(values).mode().iloc[0] if not pd.Series(values).mode().empty else "unknown"),
            cluster_assignment_confidence_mode=(
                "cluster_assignment_confidence",
                lambda values: pd.Series(values).mode().iloc[0] if not pd.Series(values).mode().empty else "unknown",
            ),
        )
        .sort_values("cluster_label")
        .reset_index(drop=True)
    )


def _determine_cluster_eligibility(
    station_summary: pd.DataFrame,
    config: StationDiagnosisConfig,
) -> tuple[pd.Series, pd.Series]:
    mature_ok = station_summary["history_group"].eq("mature") if config.cluster_mature_only else station_summary["history_group"].isin(["mature", "young"])
    coverage_ok = pd.to_numeric(station_summary["coverage_ratio"], errors="coerce") >= config.min_cluster_coverage_ratio
    observed_ok = pd.to_numeric(station_summary["observed_days"], errors="coerce") >= config.min_cluster_observed_days
    behavior_ok = station_summary["metric_reliable_behavior"].fillna(False)
    eligible = mature_ok & coverage_ok & observed_ok & behavior_ok

    reason = pd.Series("eligible", index=station_summary.index, dtype=object)
    reason.loc[~mature_ok] = "history_group_not_cluster_eligible"
    reason.loc[mature_ok & ~coverage_ok] = "coverage_too_low_for_clustering"
    reason.loc[mature_ok & coverage_ok & ~observed_ok] = "insufficient_observed_days_for_clustering"
    reason.loc[mature_ok & coverage_ok & observed_ok & ~behavior_ok] = "insufficient_behavior_reliability"
    return eligible, reason


def cluster_station_summary(
    station_summary: pd.DataFrame,
    config: StationDiagnosisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add robust cluster labels and diagnostics based on an eligible subset of stations."""

    if station_summary.empty:
        return station_summary.copy(), pd.DataFrame(), pd.DataFrame()

    base = station_summary.copy()
    eligible_mask, ineligible_reason = _determine_cluster_eligibility(base, config)
    eligible_summary = base.loc[eligible_mask].copy()
    base["cluster_eligible"] = eligible_mask
    base["cluster_ineligible_reason"] = np.where(eligible_mask, "eligible", ineligible_reason)

    if eligible_summary.empty or len(eligible_summary) < config.min_cluster_eligible_stations:
        base["cluster_label"] = np.where(
            eligible_mask,
            "not_clustered_insufficient_eligible_stations",
            "not_clustered_not_eligible",
        )
        base["cluster_assignment_confidence"] = "low"
        base["cluster_model_stability"] = np.nan
        selection = pd.DataFrame(
            [
                {
                    "candidate_k": config.n_clusters,
                    "silhouette_score": np.nan,
                    "silhouette_score_std": np.nan,
                    "seed_stability_ari": np.nan,
                    "bootstrap_stability_ari": np.nan,
                    "min_cluster_size": np.nan,
                    "max_cluster_size": np.nan,
                    "tiny_cluster_count": np.nan,
                    "size_threshold": np.nan,
                    "all_clusters_valid": False,
                    "cluster_sizes": "",
                    "feature_columns_used": "",
                    "feature_transformations": "",
                    "selected": True,
                    "selection_reason": "insufficient_eligible_stations_for_clustering",
                }
            ]
        )
        return base, pd.DataFrame(), selection

    selection, label_frame, feature_columns = _evaluate_kmeans_candidates(eligible_summary, config)
    eligible_with_labels = eligible_summary.merge(label_frame, on="station_id", how="left")
    eligible_with_labels = _relabel_clusters(eligible_with_labels)
    selected_stability = float(selection.loc[selection["selected"], "bootstrap_stability_ari"].iloc[0]) if not selection.empty else float("nan")
    eligible_with_labels["cluster_model_stability"] = selected_stability
    eligible_with_labels["cluster_assignment_confidence"] = eligible_with_labels.apply(
        lambda row: _cluster_assignment_confidence(float(row.get("_distance_margin", np.nan)), selected_stability, config),
        axis=1,
    )
    cluster_profile = _cluster_profile(eligible_with_labels)

    merge_cols = ["station_id", "cluster_label", "cluster_assignment_confidence", "cluster_model_stability"]
    with_clusters = base.merge(eligible_with_labels[merge_cols], on="station_id", how="left")
    with_clusters["cluster_label"] = with_clusters["cluster_label"].fillna("not_clustered_not_eligible")
    with_clusters["cluster_assignment_confidence"] = with_clusters["cluster_assignment_confidence"].fillna("low")
    with_clusters["cluster_model_stability"] = with_clusters["cluster_model_stability"].fillna(selected_stability)
    with_clusters["cluster_feature_columns_used"] = ",".join(feature_columns)
    return with_clusters, cluster_profile, selection
