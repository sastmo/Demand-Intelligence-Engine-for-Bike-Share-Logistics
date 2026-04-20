from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.actions import render_update_toolbar
from dashboard.components import (
    TOP_HIGHLIGHT_TITLE,
    bullet_box,
    decision_summary_table,
    render_figure_pair,
    render_remaining_figures,
    render_table,
    section_header,
    story_box,
)
from dashboard.data import diagnosis_bundle, forecast_bundle, format_short_number
from dashboard.editor import PageContentEditor


def render_system_diagnosis() -> None:
    editor = PageContentEditor("system_diagnosis", "System-Level Diagnosis")
    render_update_toolbar("system_diagnosis")

    bundle = diagnosis_bundle("system")
    summary = bundle["tables"]["summary"]
    baseline = bundle["tables"]["baseline"]
    weekday_profile = bundle["tables"]["weekday_profile"]
    monthly_profile = bundle["tables"]["monthly_profile"]
    outliers = bundle["tables"]["outliers"]
    level_shifts = bundle["tables"]["level_shifts"]
    gaps = bundle["tables"]["gaps"]
    figures = bundle["figures"]

    forecast = forecast_bundle("system")
    comparison = forecast["tables"]["comparison"]
    horizon_options = (
        sorted(comparison["horizon"].dropna().astype(int).unique().tolist())
        if not comparison.empty and "horizon" in comparison.columns
        else [7, 30, 90]
    )

    used = {
        "series.png",
        "distribution.png",
        "seasonal_profile.png",
        "acf.png",
        "periodogram.png",
        "stl.png",
        "rolling_stats.png",
        "outliers.png",
    }

    def _detect_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
        if frame.empty:
            return None
        lowered = {str(column).lower(): str(column) for column in frame.columns}
        for candidate in candidates:
            if candidate.lower() in lowered:
                return lowered[candidate.lower()]
        for column in frame.columns:
            column_text = str(column).lower()
            for candidate in candidates:
                if candidate.lower() in column_text:
                    return str(column)
        return None

    def _top_label(frame: pd.DataFrame, label_candidates: list[str], value_candidates: list[str]) -> str:
        if frame.empty:
            return "NA"
        label_col = _detect_column(frame, label_candidates)
        value_col = _detect_column(frame, value_candidates)
        if label_col is None:
            return "NA"
        ordered = frame.copy()
        if value_col is not None:
            ordered[value_col] = pd.to_numeric(ordered[value_col], errors="coerce")
            ordered = ordered.sort_values(value_col, ascending=False)
        value = ordered.iloc[0].get(label_col)
        return str(value) if pd.notna(value) else "NA"

    def _sum_metric(frame: pd.DataFrame, candidates: list[str]) -> str:
        if frame.empty:
            return "0"
        value_col = _detect_column(frame, candidates)
        if value_col is None:
            return format_short_number(frame.shape[0])
        value = pd.to_numeric(frame[value_col], errors="coerce").fillna(0).sum()
        return format_short_number(value)

    peak_weekday = _top_label(
        weekday_profile,
        ["weekday", "day_name", "day_of_week", "day"],
        ["mean_demand", "avg_demand", "average_demand", "target", "rides", "count", "value"],
    )
    peak_month = _top_label(
        monthly_profile,
        ["month_name", "month_label", "month"],
        ["mean_demand", "avg_demand", "average_demand", "target", "rides", "count", "value"],
    )
    shift_count = _sum_metric(
        level_shifts,
        ["shift_count", "n_shifts", "window_count", "count", "level_shift", "flag"],
    )
    outlier_count = _sum_metric(
        outliers,
        ["outlier_count", "n_outliers", "window_count", "count", "is_outlier", "flag"],
    )

    st.markdown("## System-Level Diagnosis")
    st.caption("We start here because the aggregate signal is smoother and easier to read than station-level demand. It gives the base demand story before we zoom into station differences.")

    bullet_box(
        TOP_HIGHLIGHT_TITLE,
        [
            "The aggregate demand signal is forecastable and not random.",
            "The series shows persistence and clear seasonality.",
            "Frequency analysis suggests recurring weekly, monthly, and yearly structure.",
            "Prediction intervals alongside the main forecast to make results more realistic and practical.",
            "Multi-horizon forecasting is needed because operational, planning, and directional decisions happen on different time windows.",
        ],
        tone="accent",
        editor=editor,
    )

    main_tab, appendix_tab, tables_tab = st.tabs(["Main page", "Appendix", "Tables"])
    with main_tab:
        section_header(
            "1. Demand level and spread",
            "Start with the full path and the operating range.",
            editor=editor,
        )
        render_figure_pair(
            figures,
            "series.png",
            "Demand rises over time and seems to follow a short, repeating pattern. It does not look random.",
            "distribution.png",
            "Most days stay in a middle demand range, with fewer very high-demand days. Forecasts should include intervals, not just one number.",
            editor=editor,
        )

        section_header(
            "2. Repeating structure",
            "Keep the pattern read direct and practical.",
            editor=editor,
        )
        seasonal_text = (
            f"Demand changes in a repeatable way across the week and across months. Some days are clearly stronger than others, so short-horizon models should learn that pattern first."
            if peak_weekday != "NA"
            else "This plot shows the recurring weekday and month pattern in aggregate demand. The weekly shape is clear and stable enough to guide short-horizon forecasting. Takeaway: weekly structure should be explicit in the benchmark set."
        )
        acf_text = "Recent days still help explain what comes next. The series keeps memory, so both recent demand and repeated timing matter for forecasting."
        render_figure_pair(
            figures,
            "seasonal_profile.png",
            seasonal_text,
            "acf.png",
            acf_text,
            editor=editor,
        )

        section_header(
            "3. Deeper seasonality",
            "Use only the two plots that add signal, not clutter.",
            editor=editor,
        )
        periodogram_text = "The signal repeats at more than one pace. It is not just day-to-day noise, which means medium-horizon forecasts should allow for more than one seasonal cycle."
        stl_text = (
            f"The level changes over time, but the seasonal shape stays visible. The model should treat level and seasonality as separate parts of the signal."
            if peak_month != "NA"
            else "This plot separates trend, seasonality, and residual movement. The seasonal layer stays visible even as the baseline shifts. Takeaway: the model should separate level and seasonality instead of forcing one fixed regime."
        )
        render_figure_pair(
            figures,
            "periodogram.png",
            periodogram_text,
            "stl.png",
            stl_text,
            editor=editor,
        )

        section_header(
            "4. Stability and change",
            "This is the practical warning block.",
            editor=editor,
        )
        rolling_text = f"The average level and the spread both move over time. The system is forecastable, but it is not fully stable, so backtesting should stay rolling and horizon-specific."
        outlier_text = f"Some periods behave differently from the norm. They are part of real operations and should stay in the evaluation so forecast quality matches real use."
        render_figure_pair(
            figures,
            "rolling_stats.png",
            rolling_text,
            "outliers.png",
            outlier_text,
            editor=editor,
        )

        section_header(
            "Simple forecasting read",
            "Leave the page with one practical direction.",
            editor=editor,
        )
        decision_table = decision_summary_table(
            comparison,
            [value for value in [7, 30, 90] if value in horizon_options] or horizon_options,
            primary="main operating window",
            secondary="planning window",
            directional="directional only",
        )
        col1, col2 = st.columns([1.15, 1.0])
        with col1:
            st.dataframe(decision_table, width="stretch", hide_index=True)
        with col2:
            story_box(
                "Takeaway",
                "Use the system page as the main forecasting view for the network. Start with the repeating weekly structure, judge each horizon on its own, and keep intervals with the forecast because both demand level and spread change over time.",
                editor=editor,
            )

    with appendix_tab:
        st.markdown("#### Extra figures")
        render_remaining_figures(figures, used_filenames=used, columns=2)

    with tables_tab:
        render_table("Diagnostics summary", summary)
        render_table("Baseline summary", baseline)
        render_table("Weekday profile", weekday_profile)
        render_table("Monthly profile", monthly_profile)
        render_table("Outlier candidates", outliers)
        render_table("Level shifts", level_shifts)
        render_table("Gap distribution", gaps)
        render_table("Model comparison", comparison)
    editor.render_sidebar()
