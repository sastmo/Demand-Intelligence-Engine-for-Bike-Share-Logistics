from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_OUTPUT_ROOT = Path("diagnosis") / "station_level_analysis" / "outputs"


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_diagnosis_artifacts(output_root: Path) -> tuple[dict[str, pd.DataFrame | str], list[str]]:
    tables_dir = output_root / "tables"
    reports_dir = output_root / "reports"

    summary_with_clusters = _read_optional_csv(tables_dir / "station_summary_with_clusters.csv")
    station_summary = _read_optional_csv(tables_dir / "station_summary_table.csv")
    category_summary = _read_optional_csv(tables_dir / "station_category_summary.csv")
    cluster_profile = _read_optional_csv(tables_dir / "station_cluster_profile.csv")
    cluster_selection = _read_optional_csv(tables_dir / "cluster_model_selection.csv")

    assumptions: list[str] = []

    if summary_with_clusters.empty and station_summary.empty:
        raise FileNotFoundError(
            f"Neither station_summary_with_clusters.csv nor station_summary_table.csv was found under {tables_dir}."
        )

    summary = summary_with_clusters.copy() if not summary_with_clusters.empty else station_summary.copy()
    if "station_category" not in summary.columns:
        assumptions.append("Station categories were not available in the summary table, so category-based recommendations are limited.")
        summary["station_category"] = "unknown"
    if "cluster_label" not in summary.columns:
        assumptions.append("Cluster labels were not available in the summary table, so cluster refinement advice is based on limited evidence.")
        summary["cluster_label"] = "not_available"
    if "history_group" not in summary.columns:
        assumptions.append("History groups were not available, so maturity handling is based on the available summary only.")
        summary["history_group"] = "unknown"

    if category_summary.empty and "station_category" in summary.columns:
        category_summary = (
            summary.groupby("station_category", as_index=False)
            .agg(station_count=("station_id", "nunique"))
            .sort_values("station_count", ascending=False)
            .reset_index(drop=True)
        )
        assumptions.append("station_category_summary.csv was missing, so the category summary was derived from the per-station table.")

    if cluster_profile.empty and "cluster_label" in summary.columns:
        numeric_means = {
            column: (column, "mean")
            for column in [
                "avg_demand",
                "zero_rate",
                "coefficient_of_variation",
                "lag1_autocorr",
                "lag7_autocorr",
                "weekday_effect_strength",
                "month_effect_strength",
                "weekend_ratio",
                "trend_slope",
                "outlier_rate",
                "correlation_with_system_total",
                "demand_share_of_system",
            ]
            if column in summary.columns
        }
        if numeric_means:
            cluster_profile = (
                summary.groupby("cluster_label", as_index=False)
                .agg(station_count=("station_id", "nunique"), **{f"{key}_mean": value for key, value in numeric_means.items()})
                .reset_index(drop=True)
            )
            assumptions.append("station_cluster_profile.csv was missing, so a simple cluster profile table was derived from the summary table.")

    diagnosis_report_text = ""
    diagnosis_report_path = reports_dir / "station_level_diagnosis_summary.md"
    if diagnosis_report_path.exists():
        diagnosis_report_text = diagnosis_report_path.read_text()
    else:
        assumptions.append("The finalized station diagnosis markdown report was not found, so readiness planning relies on tables only.")

    return (
        {
            "summary": summary,
            "category_summary": category_summary,
            "cluster_profile": cluster_profile,
            "cluster_selection": cluster_selection,
            "diagnosis_report_text": diagnosis_report_text,
        },
        assumptions,
    )


def _selected_cluster_strength(cluster_selection: pd.DataFrame, summary: pd.DataFrame) -> tuple[str, str]:
    if not cluster_selection.empty and "selected" in cluster_selection.columns:
        selected = cluster_selection.loc[cluster_selection["selected"] == True].head(1)  # noqa: E712
        if not selected.empty:
            row = selected.iloc[0]
            silhouette = pd.to_numeric(pd.Series([row.get("silhouette_score")]), errors="coerce").iloc[0]
            tiny_clusters = int(pd.to_numeric(pd.Series([row.get("tiny_cluster_count")]), errors="coerce").fillna(0).iloc[0])
            candidate_k = int(pd.to_numeric(pd.Series([row.get("candidate_k")]), errors="coerce").fillna(0).iloc[0])
            if pd.notna(silhouette) and silhouette >= 0.20 and tiny_clusters == 0:
                return "meaningful", f"Selected KMeans solution k={candidate_k} has usable separation without tiny unstable clusters."
            if pd.notna(silhouette) and silhouette >= 0.12 and tiny_clusters <= 1:
                return "moderate", f"Selected KMeans solution k={candidate_k} shows some structure, but it should stay a later refinement rather than a first-line split."
            return "weak", "Selected clustering does not look strong enough to drive the first modeling workflow."

    cluster_labels = summary["cluster_label"].astype(str)
    valid_clusters = cluster_labels[cluster_labels.str.startswith("cluster_")]
    cluster_count = valid_clusters.nunique()
    if cluster_count >= 4:
        return "moderate", "Cluster labels exist for a meaningful share of mature stations, but formal cluster quality metrics were unavailable."
    if cluster_count >= 2:
        return "weak", "Some cluster structure is present, but there is not enough evidence to rely on it yet."
    return "weak", "No reliable cluster structure is available yet."


def _build_group_recommendations(summary: pd.DataFrame, cluster_strength: str) -> pd.DataFrame:
    station_count = int(summary["station_id"].nunique())
    short_mask = summary["history_group"].isin(["newborn", "young"]) | summary["station_category"].eq("short_history")
    sparse_mask = summary["station_category"].eq("sparse_intermittent")
    anomaly_mask = summary["station_category"].eq("anomaly_heavy")
    mature_core_mask = summary["history_group"].eq("mature") & ~sparse_mask & ~anomaly_mask

    cluster_count = int(summary.loc[summary["cluster_label"].astype(str).str.startswith("cluster_"), "cluster_label"].nunique())
    cluster_action = (
        "Do not use cluster-specific models as the default next step. Revisit cluster-based refinement only after a strong global model is benchmarked."
        if cluster_strength in {"meaningful", "moderate"}
        else "Defer cluster-based modeling for now because the current diagnosis does not justify it strongly enough."
    )

    rows = [
        {
            "group_name": "all_station_day_panel",
            "group_definition": f"All {station_count} stations modeled at the station-day level.",
            "recommended_modeling_treatment": "Use one global model as the default next-stage workflow, with evaluation sliced by maturity, sparsity, category, and cluster.",
            "special_handling_needed": "Yes",
            "baseline_recommendation": "Seasonal naive per station (lag 7) plus a plain naive benchmark.",
            "notes": "This is the main panel for model comparison, but sparse and short-history stations should not drive first model selection.",
        },
        {
            "group_name": "mature_core_stations",
            "group_definition": f"{int(mature_core_mask.sum())} mature non-sparse stations with the clearest signal for first benchmark development.",
            "recommended_modeling_treatment": "Use as the main reference slice for the first global benchmark and early model diagnostics.",
            "special_handling_needed": "No",
            "baseline_recommendation": "Seasonal naive per station and naive per station.",
            "notes": "This group is the best place to judge whether the first global model is learning real shared structure.",
        },
        {
            "group_name": "sparse_intermittent_stations",
            "group_definition": f"{int(sparse_mask.sum())} stations flagged as sparse or intermittent.",
            "recommended_modeling_treatment": "Keep in the panel, but give them simpler fallback behavior or a separate sparse-treatment policy later.",
            "special_handling_needed": "Yes",
            "baseline_recommendation": "Seasonal naive per station with a simple naive or short rolling-mean fallback.",
            "notes": "These stations can distort pooled metrics and should always be reported separately.",
        },
        {
            "group_name": "short_history_stations",
            "group_definition": f"{int(short_mask.sum())} newborn or young stations with limited history.",
            "recommended_modeling_treatment": "Keep out of first benchmark tuning decisions and score them as a separate maturity slice.",
            "special_handling_needed": "Yes",
            "baseline_recommendation": "Naive per station, or seasonal naive only when enough history exists.",
            "notes": "These stations are important for later cold-start strategy, but not for choosing the first core benchmark.",
        },
        {
            "group_name": "anomaly_heavy_stations",
            "group_definition": f"{int(anomaly_mask.sum())} stations flagged as anomaly-heavy.",
            "recommended_modeling_treatment": "Keep them in evaluation slices and monitor whether robust loss choices or capped features are needed later.",
            "special_handling_needed": "Yes",
            "baseline_recommendation": "Seasonal naive per station.",
            "notes": "This group should not define the default model, but it should be tracked so unusual stations do not hide forecast instability.",
        },
        {
            "group_name": "cluster_refinement_option",
            "group_definition": f"{cluster_count} mature-station clusters currently available for later segmentation analysis.",
            "recommended_modeling_treatment": cluster_action,
            "special_handling_needed": "Later",
            "baseline_recommendation": "Use the same baseline set first, then compare errors by cluster before any refinement.",
            "notes": "Clusters are a later lens, not the first modeling split.",
        },
    ]
    return pd.DataFrame(rows)


def _build_modeling_readiness_markdown(
    summary: pd.DataFrame,
    category_summary: pd.DataFrame,
    cluster_profile: pd.DataFrame,
    cluster_selection: pd.DataFrame,
    assumptions: list[str],
) -> str:
    station_count = int(summary["station_id"].nunique())
    mature_count = int(summary["history_group"].eq("mature").sum())
    short_count = int(summary["history_group"].isin(["newborn", "young"]).sum())
    sparse_count = int(summary["station_category"].eq("sparse_intermittent").sum())
    anomaly_count = int(summary["station_category"].eq("anomaly_heavy").sum())
    cluster_count = int(summary.loc[summary["cluster_label"].astype(str).str.startswith("cluster_"), "cluster_label"].nunique())
    cluster_strength, cluster_text = _selected_cluster_strength(cluster_selection, summary)

    heterogeneous = summary["station_category"].nunique() >= 5 or cluster_count >= 4
    global_default = "Yes"
    sparse_treatment = "Yes" if sparse_count > 0 else "No"
    cluster_later = "Yes, as a later refinement." if cluster_strength in {"meaningful", "moderate"} else "Not yet."
    deepar_candidate = "Yes"

    final_recommendation = (
        "One global model plus sparse-station handling"
        if sparse_count > 0
        else "One global model first"
    )

    lines = [
        "# Station-Level Modeling Readiness Summary",
        "",
        "This stage is planning only. It does not train models, run backtests, or build training features yet.",
        "",
        "## Diagnosis Signals Used",
        f"- Stations available for planning: {station_count}",
        f"- Mature stations: {mature_count}",
        f"- Newborn or young stations: {short_count}",
        f"- Sparse or intermittent stations: {sparse_count}",
        f"- Anomaly-heavy stations: {anomaly_count}",
        f"- Category groups observed: {int(category_summary['station_category'].nunique()) if not category_summary.empty else summary['station_category'].nunique()}",
        f"- Mature-station clusters available: {cluster_count}",
        f"- Cluster assessment: {cluster_strength}",
        f"- {cluster_text}",
        "",
        "## Direct Answers",
        f"- Is one global model the right default next step? **{global_default}**",
        f"- Do sparse or intermittent stations need separate treatment? **{sparse_treatment}**",
        f"- Should cluster-based forecasting be considered later? **{cluster_later}**",
        f"- Is DeepAR a strong next candidate? **{deepar_candidate}**",
        "",
        "## Recommended Next Modeling Workflow",
        "- Primary forecasting unit: **station-day**",
        "- Maturity handling: keep all stations in reporting, but anchor early model selection on mature stations and score newborn/young stations as a separate slice.",
        "- Sparse handling: keep sparse and intermittent stations in the panel, but pair the global model with simpler fallback behavior or a separate sparse policy later.",
        "- Primary baseline benchmark: seasonal naive per station (lag 7).",
        "- Additional simple benchmark: naive per station, plus a short rolling-mean fallback only as a sparse-station sanity check.",
        "- First global non-deep model: pooled LightGBM or XGBoost with lag and calendar features at the station-day level.",
        "- First global probabilistic model: DeepAR.",
        "- Cluster-based refinement: not the default first step; only consider it later if model errors remain clearly cluster-specific.",
        "- Evaluation design: rolling-origin evaluation with shared calendar cutoffs across all stations, using 7-day and 30-day horizons and reporting overall metrics plus slices for mature, short-history, sparse, category, and cluster groups.",
        "- Recommended next-stage metrics: MAE, RMSE, MASE, and bias at minimum; add coverage and interval width once probabilistic models are in scope.",
        "",
        "## Planning Interpretation",
        f"- Station behavior looks {'heterogeneous' if heterogeneous else 'fairly mixed but not strongly segmented'} across the current diagnosis outputs.",
        "- A single global workflow still looks like the best default because the project has many related daily station series with shared calendar structure.",
        "- Sparse stations should not be treated as a side note; they need explicit handling so they do not distort the first benchmark decisions.",
        "- Cluster labels are useful as an extra diagnostic lens, but not strong enough to replace a first global benchmark.",
        "- DeepAR remains a strong later candidate because it fits the multi-series probabilistic use case well once the simpler pooled benchmarks are established.",
        "",
        "## Final Recommendation",
        f"- Recommended next-stage default: **{final_recommendation}**",
        "- Build a practical benchmark stack first: seasonal naive, naive, pooled tree-based global model.",
        "- Keep DeepAR as the first global probabilistic benchmark after the simpler global benchmark is in place.",
        "- Use cluster-based refinement only if later error analysis shows that cluster-specific differences remain important after the global model is trained.",
    ]

    if assumptions:
        lines.extend(["", "## Assumptions"])
        for assumption in assumptions:
            lines.append(f"- {assumption}")

    return "\n".join(lines)


def build_station_modeling_readiness_package(output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> dict[str, str]:
    """Read finalized diagnosis artifacts and write a lightweight modeling-readiness package."""

    root = Path(output_root)
    tables_dir = root / "tables"
    reports_dir = root / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    artifacts, assumptions = _load_diagnosis_artifacts(root)
    summary = artifacts["summary"]
    category_summary = artifacts["category_summary"]
    cluster_profile = artifacts["cluster_profile"]
    cluster_selection = artifacts["cluster_selection"]

    group_recommendations = _build_group_recommendations(summary, _selected_cluster_strength(cluster_selection, summary)[0])
    markdown = _build_modeling_readiness_markdown(
        summary=summary,
        category_summary=category_summary,
        cluster_profile=cluster_profile,
        cluster_selection=cluster_selection,
        assumptions=assumptions,
    )

    recommendations_path = tables_dir / "modeling_group_recommendations.csv"
    report_path = reports_dir / "modeling_readiness_summary.md"
    group_recommendations.to_csv(recommendations_path, index=False)
    report_path.write_text(markdown)

    return {
        "modeling_group_recommendations": str(recommendations_path),
        "modeling_readiness_summary": str(report_path),
    }
