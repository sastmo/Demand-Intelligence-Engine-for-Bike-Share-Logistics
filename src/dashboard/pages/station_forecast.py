from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.actions import render_update_toolbar
from dashboard.components import (
    TOP_HIGHLIGHT_TITLE,
    best_model_name,
    bullet_box,
    candidate_baseline,
    decision_summary_table,
    get_figure,
    ranked_models,
    render_single_figure,
    render_table,
    section_header,
    station_representative_options,
    station_slice_summary,
)
from dashboard.data import forecast_bundle, station_forecast_chart_frame
from dashboard.editor import PageContentEditor


def render_station_forecast() -> None:
    editor = PageContentEditor("station_forecast", "Station-Level Forecasting")
    render_update_toolbar("station_forecast")

    bundle = forecast_bundle("station")
    comparison = bundle["tables"]["comparison"]
    future = bundle["tables"]["future"]
    manifest = bundle["manifest"]
    observed = bundle["tables"]["observed_panel"]
    slice_metrics = bundle["tables"]["slice_metrics"]
    enabled_models = [str(value) for value in manifest.get("enabled_models", [])]
    horizon_options = (
        sorted(future["horizon"].dropna().astype(int).unique().tolist())
        if not future.empty and "horizon" in future.columns
        else [7, 30, 90]
    )

    st.markdown("## Station-Level Forecasting")
    st.caption("This page should show one global station-day workflow, then prove whether it holds across the slices that actually matter.")
    bullet_box(
        TOP_HIGHLIGHT_TITLE,
        [
            "Station-level forecasting should be built as one global station-day pipeline, not as many separate first-stage station models.",
            "The panel is intentionally unbalanced because stations open, mature, and leave service at different times.",
            "XGBoost and LightGBM are the strongest performers in the current backtest comparison.",
            "DeepAR is included as a heavier benchmark, but it is not the leading option in the current run.",
            "Forecast quality should be interpreted by horizon and slice, not by one pooled average alone.",
            "Slice-based reporting is important because station behavior is heterogeneous across maturity, sparsity, category, and cluster."
        ],
        tone="accent",
        editor=editor,
    )

    if len(enabled_models) <= 1:
        st.warning("Current station forecast output contains only one model. The page layout supports full comparison, but the saved outputs do not yet expose the full candidate set.")

    main_tab, appendix_tab, tables_tab = st.tabs(["Main page", "Appendix", "Tables"])
    with main_tab:
        section_header(
            "Overall backtest ranking",
            "Use the main comparison chart as a portfolio view, then immediately qualify it with slice-aware evaluation.",
            editor=editor,
        )
        col1, col2 = st.columns([1.8, 1.1])
        with col1:
            render_single_figure(
                get_figure(bundle["figures"], "station_level_model_comparison.png"),
                "Missing figure: station_level_model_comparison.png",
                [
                    ("Shows", "Mean station-level backtest performance by model and horizon across the portfolio."),
                    ("Insight", "The current untuned run identifies leading model families, but not a final production winner by itself."),
                    ("Why it matters", "Overall ranking is useful for narrowing candidates, but it is not enough for final selection."),
                ],
                editor=editor,
                block_id="figure:station-level-model-comparison",
            )
        with col2:
            station_summary = decision_summary_table(
                comparison,
                [value for value in [7, 30, 90] if value in horizon_options] or horizon_options,
                primary="strongest short-horizon candidates",
                secondary="best current medium-horizon candidates",
                directional="sensitivity only",
            )
            st.markdown("#### Horizon summary")
            st.dataframe(station_summary, width="stretch", hide_index=True)

        section_header(
            "Slice-based performance",
            "This is the most important station-only block because the diagnosis already showed that the portfolio is heterogeneous.",
            editor=editor,
        )
        horizon = st.selectbox("Slice horizon", options=horizon_options, index=0, key="station_slice_horizon")
        slice_summary = station_slice_summary(slice_metrics, int(horizon))
        if slice_summary.empty:
            st.info("No slice metrics file found yet. Keep this block in the layout because it is mandatory for the final station decision page.")
        else:
            st.dataframe(slice_summary, width="stretch", hide_index=True)
            bullet_box(
                "How to read this block",
                [
                    "Overall mean MASE is not enough.",
                    "The production choice should be checked against the mature active core.",
                    "Sparse and short-history slices should be reported separately instead of dominating the final decision.",
                ],
                editor=editor,
            )

        section_header(
            "Representative station examples",
            "Use only a few representative station views to show where the global workflow works well and where it struggles.",
            editor=editor,
        )
        ranked = ranked_models(comparison, 7)
        selected_model = ranked[0] if ranked else (enabled_models[0] if enabled_models else "NA")
        baseline_model = candidate_baseline(ranked or enabled_models)
        representative = station_representative_options(future)
        if future.empty or observed.empty:
            st.info("Representative station panels need both observed history and saved future forecasts.")
        else:
            categories = list(representative.keys())[:4]
            if not categories:
                st.info("No representative station slices could be derived from the current forecast output.")
            else:
                for category in categories:
                    station_ids = representative[category]
                    st.markdown(f"#### {category}")
                    selection_cols = st.columns(3)
                    with selection_cols[0]:
                        station_id = st.selectbox(
                            f"Station for {category}",
                            options=station_ids,
                            index=0,
                            key=f"station_example_{category}",
                        )
                    with selection_cols[1]:
                        model_name = st.selectbox(
                            f"Selected model for {category}",
                            options=ranked or enabled_models or [selected_model],
                            index=0,
                            key=f"station_model_{category}",
                        )
                    with selection_cols[2]:
                        baseline_choice = st.selectbox(
                            f"Baseline for {category}",
                            options=ranked or enabled_models or [baseline_model],
                            index=(ranked or enabled_models or [baseline_model]).index(baseline_model) if baseline_model in (ranked or enabled_models or [baseline_model]) else 0,
                            key=f"station_baseline_{category}",
                        )
                    chart_cols = st.columns(2)
                    with chart_cols[0]:
                        selected_chart = station_forecast_chart_frame(
                            observed,
                            future,
                            station_id,
                            model_name,
                            int(horizon),
                            history_days=90,
                        )
                        if selected_chart.empty:
                            st.info("No selected-model chart for this station.")
                        else:
                            st.markdown("**Selected model**")
                            st.line_chart(
                                selected_chart.set_index("date")[
                                    [column for column in ["observed", "forecast", "lower_80", "upper_80"] if column in selected_chart.columns]
                                ],
                                width="stretch",
                            )
                    with chart_cols[1]:
                        baseline_chart = station_forecast_chart_frame(
                            observed,
                            future,
                            station_id,
                            baseline_choice,
                            int(horizon),
                            history_days=90,
                        )
                        if baseline_chart.empty:
                            st.info("No baseline chart for this station.")
                        else:
                            st.markdown("**Baseline**")
                            st.line_chart(
                                baseline_chart.set_index("date")[
                                    [column for column in ["observed", "forecast", "lower_80", "upper_80"] if column in baseline_chart.columns]
                                ],
                                width="stretch",
                            )

        recommendation = pd.DataFrame(
            [
                {"Decision area": "Default model family", "Recommendation": best_model_name(comparison, 7)},
                {
                    "Decision area": "Main benchmark",
                    "Recommendation": candidate_baseline(ranked_models(comparison, 7) or enabled_models),
                },
                {"Decision area": "Primary judging slice", "Recommendation": "mature active non-sparse core"},
                {"Decision area": "Reporting rule", "Recommendation": "always report overall plus slice metrics"},
                {"Decision area": "90-day use", "Recommendation": "directional only"},
            ]
        )
        section_header("Final recommendation", editor=editor)
        st.dataframe(recommendation, width="stretch", hide_index=True)

    with appendix_tab:
        if manifest:
            st.markdown("#### Run metadata")
            with st.expander("Show technical run details", expanded=False):
                st.json(manifest)
        else:
            st.info("No additional appendix content is available for this run.")

    with tables_tab:
        render_table("Model comparison", comparison)
        render_table("Slice metrics", slice_metrics, rows=300)
        render_table("Future forecasts", future, rows=300)
        render_table("Interval coverage", bundle["tables"]["interval_coverage"])
        render_table("Interval coverage by step", bundle["tables"]["interval_coverage_by_step"])
        render_table("Interval calibration", bundle["tables"]["interval_calibration"])
        render_table("Backtest metrics", bundle["tables"]["backtest_metrics"], rows=300)
        render_table("Backtest windows", bundle["tables"]["windows"], rows=300)
        render_table("Model registry", bundle["tables"]["registry"])
    editor.render_sidebar()
