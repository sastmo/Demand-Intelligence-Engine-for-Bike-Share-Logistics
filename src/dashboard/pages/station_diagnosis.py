from __future__ import annotations

import streamlit as st

from dashboard.actions import render_update_toolbar
from dashboard.components import (
    TOP_HIGHLIGHT_TITLE,
    bullet_box,
    render_figure_pair,
    render_note,
    render_remaining_figures,
    render_table,
    section_header,
)
from dashboard.data import diagnosis_bundle
from dashboard.editor import PageContentEditor


def render_station_diagnosis() -> None:
    editor = PageContentEditor("station_diagnosis", "Station-Level Diagnosis")
    render_update_toolbar("station_diagnosis")

    bundle = diagnosis_bundle("station")
    inventory = bundle["tables"]["inventory"]
    summary = bundle["tables"]["summary"]
    category_summary = bundle["tables"]["category_summary"]
    cluster_profile = bundle["tables"]["cluster_profile"]
    figures = bundle["figures"]
    used = {
        "history_group_counts.png",
        "history_days_histogram.png",
        "avg_demand_histogram.png",
        "avg_demand_vs_zero_rate_by_category.png",
        "category_counts.png",
        "representative_station_timeseries.png",
        "cluster_counts.png",
        "cluster_profile_heatmap.png",
    }

    st.markdown("## Station-Level Diagnosis")
    st.caption("This page is the bridge from diagnosis to forecasting design. Read it top to bottom to see who is in the station universe, which stations carry usable signal, and why the first modeling choice should stay global station-day with slice-based evaluation.")

    bullet_box(
        TOP_HIGHLIGHT_TITLE,
        [
            "The station network is not one uniform population; it contains a mature active core alongside short-history, sparse, inactive, and special-behavior groups.",
            "The observed station universe is broader than the expected operational count, so inventory and activity need to be interpreted before any model result is taken at face value.",
            "Demand is strongly skewed, with a small high-demand group and a long low-demand tail, which makes one average station a weak summary.",
            "Model quality should be judged by slice, not only by one overall average, especially when separating short-history, sparse, and inactive stations from the healthy core.",
            "The clearest next step is one global station-day forecasting workflow: train globally, evaluate by explicit slices, and refine only where the results show a real need.",
        ],
        tone="accent",
        editor=editor,
    )

    main_tab, appendix_tab, tables_tab = st.tabs(["Main page", "Appendix", "Tables"])
    with main_tab:
        section_header(
            "Station universe and maturity",
            "Start with who is in the network and how much usable history each station actually has.",
            editor=editor,
        )
        render_figure_pair(
            figures,
            "history_group_counts.png",
            [
                ("Shows", "how the station universe splits across mature and shorter-history groups"),
                ("Insight", "forecasting quality is partly a maturity problem, not just a behavior problem"),
                ("Why it matters", "the portfolio should be read as a mix of mature and early-life stations before any model score is interpreted"),
            ],
            "history_days_histogram.png",
            [
                ("Shows", "the distribution of available history length across stations"),
                ("Insight", "the network contains a meaningful short-history tail alongside a large mature core"),
                ("Why it matters", "history length affects confidence, comparability, and the way slice results should be reported"),
            ],
            editor=editor,
        )

        section_header(
            "Signal quality and skew",
            "These views explain why one overall average station view is misleading.",
            editor=editor,
        )
        render_figure_pair(
            figures,
            "avg_demand_histogram.png",
            [
                ("Shows", "the distribution of average station demand across the network"),
                ("Insight", "a small strong group and a long weak tail sit in the same portfolio"),
                ("Why it matters", "one portfolio-level average can hide the real spread between strong-signal and weak-signal stations"),
            ],
            "avg_demand_vs_zero_rate_by_category.png",
            [
                ("Shows", "average demand against zero-day frequency across station categories"),
                ("Insight", "strong-signal stations separate clearly from sparse and intermittent stations"),
                ("Why it matters", "this is one of the clearest guides for building evaluation slices and deciding where special handling is needed"),
            ],
            editor=editor,
        )

        section_header(
            "Behavior slices",
            "Use categories to explain the portfolio and to organize reporting, not as the first production split.",
            editor=editor,
        )
        render_figure_pair(
            figures,
            "category_counts.png",
            [
                ("Shows", "how many stations fall into each behavioral category"),
                ("Insight", "the network contains several meaningful station stories instead of one typical profile"),
                ("Why it matters", "reporting should explain the portfolio mix, not just the average score"),
            ],
            "representative_station_timeseries.png",
            [
                ("Shows", "representative station time-series examples from the main category groups"),
                ("Insight", "short-history and sparse stations are not just smaller versions of busy and stable stations"),
                ("Why it matters", "a few concrete examples make it easier to explain why slice-based evaluation is necessary before choosing a default model"),
            ],
            editor=editor,
        )
        bullet_box(
            "Category interpretation",
            [
                "busy_stable: productive core",
                "mixed_profile: broad middle",
                "weekend_leisure: clear non-commuter segment",
                "sparse_intermittent: weak-signal tail",
                "anomaly_heavy: monitoring slice",
                "short_history: maturity bucket, not stable behavior",
            ],
            editor=editor,
        )

        section_header(
            "Mature-core structure and later refinement",
            "After separating the broad station types, look inside the mature core to understand where later refinement could help.",
            editor=editor,
        )
        render_figure_pair(
            figures,
            "cluster_counts.png",
            [
                ("Shows", "the cluster size distribution among mature stations"),
                ("Insight", "even the mature core does not behave like one homogeneous group"),
                ("Why it matters", "cluster structure can guide later refinement, but it should not be the first production split"),
            ],
            "cluster_profile_heatmap.png",
            [
                ("Shows", "the compact feature profile for mature-station clusters"),
                ("Insight", "mature stations separate into structurally different numerical regimes"),
                ("Why it matters", "use clusters first for interpretation and slice analysis, then only later for model specialization if the residuals justify it"),
            ],
            editor=editor,
        )
        bullet_box(
            "Forecasting implication",
            [
                "Forecast at the station-day level.",
                "Train one global model first, not many station or cluster models.",
                "Evaluate by mature core, short-history, sparse/intermittent, category, and cluster slices.",
            ],
            editor=editor,
        )

    with appendix_tab:
        if bundle["note_text"]:
            render_note("Station diagnosis notes", bundle["note_path"], bundle["note_text"])
        st.markdown("#### Appendix figures")
        render_remaining_figures(figures, used_filenames=used, columns=2)

    with tables_tab:
        selector_col, filter_col = st.columns([1, 2])
        selected_station = selector_col.selectbox(
            "Inspect station",
            options=sorted(summary["station_id"].astype(str).tolist()) if not summary.empty and "station_id" in summary.columns else [],
        )
        selected_category = filter_col.selectbox(
            "Filter summary by category",
            options=["all"] + sorted(summary["station_category"].dropna().astype(str).unique().tolist()) if not summary.empty and "station_category" in summary.columns else ["all"],
        )
        filtered_summary = summary.copy()
        if selected_category != "all" and "station_category" in filtered_summary.columns:
            filtered_summary = filtered_summary.loc[
                filtered_summary["station_category"].astype(str) == selected_category
            ]

        render_table("Station summary", filtered_summary)
        if selected_station:
            station_row = (
                summary.loc[summary["station_id"].astype(str) == str(selected_station)]
                if not summary.empty and "station_id" in summary.columns
                else summary.iloc[0:0]
            )
            render_table("Selected station snapshot", station_row)
        render_table("Station inventory", inventory)
        render_table("Category summary", category_summary)
        render_table("Cluster profile", cluster_profile)
        render_table("Cluster model selection", bundle["tables"]["cluster_selection"])
        render_table("Top by average demand", bundle["tables"]["top_avg_demand"])
        render_table("Top by zero rate", bundle["tables"]["top_zero_rate"])
        render_table("Top by outlier rate", bundle["tables"]["top_outlier_rate"])
        render_table("Top by coefficient of variation", bundle["tables"]["top_cv"])
    editor.render_sidebar()
