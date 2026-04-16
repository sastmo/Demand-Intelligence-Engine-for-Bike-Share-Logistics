from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from metro_bike_share_forecasting.dashboard.data import (
    diagnosis_bundle,
    forecast_bundle,
    format_short_number,
    prediction_column,
    station_forecast_chart_frame,
    system_forecast_chart_frame,
)


def _metric_cards(cards: list[tuple[str, str, str]]) -> None:
    columns = st.columns(len(cards))
    for column, (label, value, caption) in zip(columns, cards):
        column.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{value}</div>
              <div class="metric-caption">{caption}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_note(title: str, path: Path | None, text: str | None) -> None:
    st.markdown(f"### {title}")
    if path is not None:
        st.caption(f"Source note: `{path}`")
    if text:
        st.markdown(text)
    else:
        st.info("No note file found for this section yet.")


def _render_figure_gallery(items: list[tuple[str, Path]], columns: int = 2) -> None:
    if not items:
        st.info("No figures found for this section.")
        return
    grid = st.columns(columns)
    for index, (label, path) in enumerate(items):
        with grid[index % columns]:
            st.image(str(path), caption=label, use_container_width=True)


def render_system_diagnosis() -> None:
    bundle = diagnosis_bundle("system")
    summary = bundle["tables"]["summary"]
    baseline = bundle["tables"]["baseline"]

    st.markdown("## System Diagnosis")
    if not summary.empty:
        row = summary.iloc[0]
        _metric_cards(
            [
                ("Rows", format_short_number(row.get("row_count")), "Observed system-level daily records"),
                ("Trend Strength", format_short_number(row.get("trend_strength")), "Signal persistence and trend"),
                ("Seasonal Strength", format_short_number(row.get("seasonal_strength")), "Repeatable seasonal structure"),
                ("Primary Period", format_short_number(row.get("primary_period")), "Detected dominant cycle"),
            ]
        )

    story_tab, figures_tab, tables_tab = st.tabs(["Narrative", "Figures", "Tables"])
    with story_tab:
        _render_note("System Diagnosis Notes", bundle["note_path"], bundle["note_text"])
    with figures_tab:
        _render_figure_gallery(bundle["figures"])
    with tables_tab:
        st.markdown("### Diagnostics Summary")
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.markdown("### Baseline Summary")
        st.dataframe(baseline, use_container_width=True, hide_index=True)
        for label, key in [
            ("Weekday Profile", "weekday_profile"),
            ("Monthly Profile", "monthly_profile"),
            ("Outlier Candidates", "outliers"),
            ("Level Shifts", "level_shifts"),
            ("Gap Distribution", "gaps"),
        ]:
            table = bundle["tables"][key]
            if not table.empty:
                st.markdown(f"### {label}")
                st.dataframe(table, use_container_width=True, hide_index=True)


def render_station_diagnosis() -> None:
    bundle = diagnosis_bundle("station")
    inventory = bundle["tables"]["inventory"]
    summary = bundle["tables"]["summary"]
    category_summary = bundle["tables"]["category_summary"]
    cluster_profile = bundle["tables"]["cluster_profile"]

    st.markdown("## Station Diagnosis")
    if not summary.empty:
        mature_count = int((summary["history_group"] == "mature").sum()) if "history_group" in summary.columns else 0
        short_count = int(summary["is_short_history"].fillna(False).sum()) if "is_short_history" in summary.columns else 0
        inactive_count = int((~summary["appears_active_recently"].fillna(False)).sum()) if "appears_active_recently" in summary.columns else 0
        _metric_cards(
            [
                ("Stations", format_short_number(summary["station_id"].nunique()), "Observed station universe"),
                ("Mature", format_short_number(mature_count), "Stations with longer usable history"),
                ("Short History", format_short_number(short_count), "Newborn and young stations"),
                ("Not Recently Active", format_short_number(inactive_count), "Operationally weak recent activity"),
            ]
        )

    story_tab, figures_tab, tables_tab = st.tabs(["Narrative", "Figures", "Tables"])
    with story_tab:
        _render_note("Station Diagnosis Notes", bundle["note_path"], bundle["note_text"])
    with figures_tab:
        _render_figure_gallery(bundle["figures"])
    with tables_tab:
        selector_col, filter_col = st.columns([1, 2])
        selected_station = selector_col.selectbox(
            "Inspect station",
            options=sorted(summary["station_id"].astype(str).tolist()) if not summary.empty else [],
        )
        selected_category = filter_col.selectbox(
            "Filter summary by category",
            options=["all"] + sorted(summary["station_category"].dropna().astype(str).unique().tolist()) if "station_category" in summary.columns else ["all"],
        )
        filtered_summary = summary.copy()
        if selected_category != "all":
            filtered_summary = filtered_summary.loc[filtered_summary["station_category"].astype(str) == selected_category]
        st.markdown("### Station Summary")
        st.dataframe(filtered_summary, use_container_width=True, hide_index=True)
        if selected_station:
            station_row = summary.loc[summary["station_id"].astype(str) == str(selected_station)]
            if not station_row.empty:
                st.markdown("### Selected Station Snapshot")
                st.dataframe(station_row, use_container_width=True, hide_index=True)
        for title, table in [
            ("Station Inventory", inventory),
            ("Category Summary", category_summary),
            ("Cluster Profile", cluster_profile),
            ("Cluster Model Selection", bundle["tables"]["cluster_selection"]),
            ("Top by Average Demand", bundle["tables"]["top_avg_demand"]),
            ("Top by Zero Rate", bundle["tables"]["top_zero_rate"]),
            ("Top by Outlier Rate", bundle["tables"]["top_outlier_rate"]),
            ("Top by Coefficient of Variation", bundle["tables"]["top_cv"]),
        ]:
            if not table.empty:
                st.markdown(f"### {title}")
                st.dataframe(table, use_container_width=True, hide_index=True)


def render_system_forecast() -> None:
    bundle = forecast_bundle("system")
    comparison = bundle["tables"]["comparison"]
    future = bundle["tables"]["future"]
    manifest = bundle["manifest"]

    st.markdown("## System Forecast")
    if not comparison.empty:
        best_7 = comparison.loc[comparison["horizon"] == 7].sort_values("mean_mase").head(1)
        best_30 = comparison.loc[comparison["horizon"] == 30].sort_values("mean_mase").head(1)
        _metric_cards(
            [
                ("Models in Run", format_short_number(len(manifest.get("enabled_models", []))), "Current forecast registry"),
                ("Best 7-Day", best_7["model_name"].iat[0] if not best_7.empty else "NA", "Lowest mean MASE at h=7"),
                ("Best 30-Day", best_30["model_name"].iat[0] if not best_30.empty else "NA", "Lowest mean MASE at h=30"),
                ("Forecast Rows", format_short_number(len(future)), "Saved future predictions"),
            ]
        )

    story_tab, visuals_tab, tables_tab = st.tabs(["Narrative", "Visuals", "Tables"])
    with story_tab:
        _render_note("Forecast Contract", bundle["note_path"], bundle["note_text"])
        if manifest:
            st.markdown("### Run Manifest")
            st.json(manifest)
    with visuals_tab:
        _render_figure_gallery(bundle["figures"], columns=2)
        if not future.empty:
            model = st.selectbox("Model", options=sorted(future["model_name"].astype(str).unique().tolist()), key="system_forecast_model")
            horizon = st.selectbox("Horizon", options=sorted(future["horizon"].dropna().astype(int).unique().tolist()), key="system_forecast_horizon")
            chart = system_forecast_chart_frame(future, model, int(horizon))
            if not chart.empty:
                st.line_chart(chart.set_index("date")[["forecast", "lower_80", "upper_80"]], use_container_width=True)
    with tables_tab:
        st.markdown("### Model Comparison")
        st.dataframe(comparison.sort_values(["horizon", "mean_mase"]), use_container_width=True, hide_index=True)
        st.markdown("### Future Forecasts")
        st.dataframe(future, use_container_width=True, hide_index=True)
        for title, table in [
            ("Interval Coverage", bundle["tables"]["interval_coverage"]),
            ("Interval Coverage by Step", bundle["tables"]["interval_coverage_by_step"]),
            ("Interval Calibration", bundle["tables"]["interval_calibration"]),
            ("Production Fit Diagnostics", bundle["tables"]["fit_diagnostics"]),
            ("Model Registry", bundle["tables"]["registry"]),
        ]:
            if not table.empty:
                st.markdown(f"### {title}")
                st.dataframe(table, use_container_width=True, hide_index=True)


def render_station_forecast() -> None:
    bundle = forecast_bundle("station")
    comparison = bundle["tables"]["comparison"]
    future = bundle["tables"]["future"]
    manifest = bundle["manifest"]
    observed = bundle["tables"]["observed_panel"]
    slice_metrics = bundle["tables"]["slice_metrics"]

    st.markdown("## Station Forecast")
    enabled_models = manifest.get("enabled_models", [])
    if not future.empty:
        _metric_cards(
            [
                ("Models in Run", format_short_number(len(enabled_models)), "Current station comparison set"),
                ("Stations Forecasted", format_short_number(future["station_id"].nunique()), "Stations in the future forecast file"),
                ("Horizons", ", ".join(str(value) for value in sorted(future["horizon"].dropna().astype(int).unique().tolist())), "Saved forecast windows"),
                ("Forecast Rows", format_short_number(len(future)), "Rows in current station forecast output"),
            ]
        )
    if len(enabled_models) <= 1:
        st.warning("Current station forecast output contains only one model. Run the full station pipeline to compare all models together.")

    overview_tab, station_tab, slices_tab = st.tabs(["Overview", "Station View", "Slice Metrics"])
    with overview_tab:
        if manifest:
            st.markdown("### Run Manifest")
            st.json(manifest)
        _render_figure_gallery(bundle["figures"], columns=1)
        st.markdown("### Model Comparison")
        st.dataframe(comparison.sort_values(["horizon", "mean_mase"]), use_container_width=True, hide_index=True)
        st.markdown("### Future Forecasts")
        st.dataframe(future.head(500), use_container_width=True, hide_index=True)
    with station_tab:
        if future.empty:
            st.info("No station-level future forecast file found yet.")
        else:
            station_options = sorted(future["station_id"].astype(str).unique().tolist())
            model_options = sorted(future["model_name"].astype(str).unique().tolist())
            horizon_options = sorted(future["horizon"].dropna().astype(int).unique().tolist())
            col1, col2, col3 = st.columns(3)
            station_id = col1.selectbox("Station", options=station_options)
            model_name = col2.selectbox("Model", options=model_options)
            horizon = col3.selectbox("Horizon", options=horizon_options)

            station_subset = future.loc[
                (future["station_id"].astype(str) == str(station_id))
                & (future["model_name"].astype(str) == str(model_name))
                & (future["horizon"] == int(horizon))
            ].copy()
            if not station_subset.empty:
                metadata_cols = [
                    column
                    for column in ["history_group", "station_category", "cluster_label", "is_short_history", "is_zero_almost_always", "appears_active_recently"]
                    if column in station_subset.columns
                ]
                st.markdown("### Selected Station Metadata")
                st.dataframe(station_subset[["station_id"] + metadata_cols].drop_duplicates(), use_container_width=True, hide_index=True)

            chart = station_forecast_chart_frame(observed, future, station_id, model_name, int(horizon))
            if not chart.empty:
                st.markdown("### History + Forecast")
                st.line_chart(chart.set_index("date")[["observed", "forecast", "lower_80", "upper_80"]], use_container_width=True)
            st.markdown("### Selected Station Forecast Rows")
            st.dataframe(station_subset, use_container_width=True, hide_index=True)
    with slices_tab:
        if slice_metrics.empty:
            st.info("No slice metrics file found yet.")
        else:
            col1, col2 = st.columns(2)
            model_filter = col1.selectbox("Slice model", options=sorted(slice_metrics["model_name"].astype(str).unique().tolist()))
            horizon_filter = col2.selectbox("Slice horizon", options=sorted(slice_metrics["horizon"].dropna().astype(int).unique().tolist()))
            filtered = slice_metrics.loc[
                (slice_metrics["model_name"].astype(str) == str(model_filter))
                & (slice_metrics["horizon"] == int(horizon_filter))
            ].copy()
            st.dataframe(filtered.sort_values(["slice_type", "slice_value"]), use_container_width=True, hide_index=True)
            for title, table in [
                ("Interval Coverage", bundle["tables"]["interval_coverage"]),
                ("Interval Coverage by Step", bundle["tables"]["interval_coverage_by_step"]),
                ("Interval Calibration", bundle["tables"]["interval_calibration"]),
                ("Backtest Metrics", bundle["tables"]["backtest_metrics"]),
                ("Backtest Windows", bundle["tables"]["windows"]),
                ("Model Registry", bundle["tables"]["registry"]),
            ]:
                if not table.empty:
                    st.markdown(f"### {title}")
                    st.dataframe(table, use_container_width=True, hide_index=True)
