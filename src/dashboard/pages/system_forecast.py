from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.actions import render_update_toolbar
from dashboard.components import (
    TOP_HIGHLIGHT_TITLE,
    bullet_box,
    candidate_baseline,
    decision_summary_table,
    get_figure,
    model_forecast_table,
    ranked_models,
    render_single_figure,
    render_table,
    section_header,
)
from dashboard.data import forecast_bundle, system_forecast_chart_frame
from dashboard.editor import PageContentEditor


def render_system_forecast() -> None:
    editor = PageContentEditor("system_forecast", "System-Level Forecasting")
    render_update_toolbar("system_forecast")

    bundle = forecast_bundle("system")
    comparison = bundle["tables"]["comparison"]
    future = bundle["tables"]["future"]
    manifest = bundle["manifest"]
    enabled_models = [str(value) for value in manifest.get("enabled_models", [])]

    st.markdown("## System-Level Forecasting")
    st.caption("The main decision is horizon trust. Use the system page to show what is forecast, how it was validated, and what should be used now.")

    horizons = (
        sorted(future["horizon"].dropna().astype(int).unique().tolist())
        if not future.empty and "horizon" in future.columns
        else [7, 30, 90]
    )

    bullet_box(
        TOP_HIGHLIGHT_TITLE,
        [
            "System-level demand is forecastable. The overall pattern is strong enough to support practical planning.",
            "The forecasting pipeline is working end to end. The current focus is now on improving quality and trust by horizon.",
            "We use both point and interval forecasts. This gives a clearer view for planning, inventory, and redistribution decisions.",
            "Each horizon has a different role. 7-day is operational, 30-day is tactical, and 90-day is directional sensitivity only.",
        ],
        tone="accent",
        editor=editor,
    )

    main_tab, contract_tab, tables_tab = st.tabs(["Main page", "Forecasting Contract", "Tables"])
    with main_tab:
        section_header(
            "Backtest comparison",
            "Judge trust by horizon, not by model name alone.",
            editor=editor,
        )
        col1, col2 = st.columns([1.8, 1.1])
        with col1:
            render_single_figure(
                get_figure(bundle["figures"], "system_level_model_comparison.png"),
                "Missing figure: system_level_model_comparison.png",
                [
                    ("Shows", "System-level backtest performance across forecast horizons."),
                    ("Insight", "Short-horizon performance should drive practical trust more than the long-horizon averages."),
                    ("Why it matters", "A model can lead at one horizon without being the right production choice everywhere."),
                ],
                editor=editor,
                block_id="figure:system-level-model-comparison",
            )
        with col2:
            decision_table = decision_summary_table(
                comparison,
                [value for value in [7, 30, 90] if value in horizons] or horizons,
                primary="operational benchmark",
                secondary="tactical benchmark",
                directional="directional only",
            )
            st.markdown("#### Horizon summary")
            st.dataframe(decision_table, width="stretch", hide_index=True)

        ranked_7 = ranked_models(comparison, 7)
        selected_model = ranked_7[0] if ranked_7 else (enabled_models[0] if enabled_models else "NA")
        baseline_model = candidate_baseline(ranked_7 or enabled_models)
        challenger_model = ranked_7[1] if len(ranked_7) > 1 else selected_model

        section_header(
            "Production forecast outlook",
            "Keep the main page focused on the selected production view, one baseline, and one challenger.",
            editor=editor,
        )
        controls = st.columns(4)
        with controls[0]:
            selected_model = st.selectbox(
                "Selected model",
                options=ranked_7 or enabled_models or ["NA"],
                index=0,
                key="system_selected_model",
            )
        with controls[1]:
            baseline_choices = ranked_7 or enabled_models or [selected_model]
            baseline_index = baseline_choices.index(baseline_model) if baseline_model in baseline_choices else 0
            baseline_model = st.selectbox(
                "Baseline",
                options=baseline_choices,
                index=baseline_index,
                key="system_baseline_model",
            )
        with controls[2]:
            challenger_choices = ranked_7 or enabled_models or [selected_model]
            challenger_index = challenger_choices.index(challenger_model) if challenger_model in challenger_choices else 0
            challenger_model = st.selectbox(
                "Challenger",
                options=challenger_choices,
                index=challenger_index,
                key="system_challenger_model",
            )
        with controls[3]:
            horizon = st.selectbox("Display horizon", options=horizons, index=0, key="system_display_horizon")

        plot_cols = st.columns(3)
        for column, model_name, title in [
            (plot_cols[0], selected_model, "Selected production view"),
            (plot_cols[1], baseline_model, "Benchmark baseline"),
            (plot_cols[2], challenger_model, "Challenger"),
        ]:
            with column:
                st.markdown(f"#### {title}")
                chart = system_forecast_chart_frame(future, model_name, int(horizon)) if not future.empty else pd.DataFrame()
                if chart.empty:
                    st.info("No saved forecast rows for this combination.")
                else:
                    st.line_chart(
                        chart.set_index("date")[
                            [column for column in ["forecast", "lower_80", "upper_80"] if column in chart.columns]
                        ],
                        width="stretch",
                    )
                    st.dataframe(
                        model_forecast_table(future, model_name, int(horizon)),
                        width="stretch",
                        hide_index=True,
                    )

        recommendation = pd.DataFrame(
            [
                {"Horizon": "7", "Use case": "operations", "Confidence": "highest", "Reporting language": "ready for practical use"},
                {"Horizon": "30", "Use case": "tactical planning", "Confidence": "medium", "Reporting language": "usable, still improving"},
                {"Horizon": "90", "Use case": "strategic sensitivity", "Confidence": "low", "Reporting language": "directional only"},
            ]
        )
        section_header("Recommendation and caveats", editor=editor)
        st.dataframe(recommendation, width="stretch", hide_index=True)
        bullet_box(
            "Modeling implications",
            [
                "The diagnosis already showed strong weekly structure and regime sensitivity, so trust should stay horizon-specific.",
                "Use rolling-origin backtesting as the decision contract for 7-day and 30-day selection.",
                "Keep 90-day outputs visible, but communicate them as directional sensitivity rather than production-grade guidance.",
            ],
            editor=editor,
        )

    with contract_tab:
        if bundle["note_text"]:
            st.markdown(str(bundle["note_text"]))
        else:
            st.info("No forecasting contract file found for the system-level forecasting page.")

    with tables_tab:
        render_table("Model comparison", comparison)
        render_table("Future forecasts", future, rows=300)
        render_table("Interval coverage", bundle["tables"]["interval_coverage"])
        render_table("Interval coverage by step", bundle["tables"]["interval_coverage_by_step"])
        render_table("Interval calibration", bundle["tables"]["interval_calibration"])
        render_table("Production fit diagnostics", bundle["tables"]["fit_diagnostics"])
        render_table("Model registry", bundle["tables"]["registry"])
        render_table("Backtest metrics", bundle["tables"]["backtest_metrics"], rows=200)
        render_table("Backtest residuals", bundle["tables"]["residuals"], rows=200)
    editor.render_sidebar()
