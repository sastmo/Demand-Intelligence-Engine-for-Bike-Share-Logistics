from __future__ import annotations

import streamlit as st

from dashboard.actions import render_update_toolbar
from dashboard.components import TOP_HIGHLIGHT_TITLE, bullet_box, metric_cards, section_header, story_box
from dashboard.data import diagnosis_bundle, forecast_bundle, format_short_number
from dashboard.editor import PageContentEditor


def render_overview() -> None:
    editor = PageContentEditor("overview", "Intro")
    render_update_toolbar("overview")

    system_diag = diagnosis_bundle("system")
    station_diag = diagnosis_bundle("station")
    system_forecast = forecast_bundle("system")
    station_forecast = forecast_bundle("station")

    system_summary = system_diag["tables"]["summary"]
    station_summary = station_diag["tables"]["summary"]
    system_comparison = system_forecast["tables"]["comparison"]
    station_comparison = station_forecast["tables"]["comparison"]

    st.markdown("## Intro")
    st.caption("Use this page as the front door to the dashboard. Start with system-level diagnosis, then station-level diagnosis, then move into the two forecasting pages once the signal and portfolio structure are clear.")

    bullet_box(
        TOP_HIGHLIGHT_TITLE,
        [
            "Start with the system view to understand the main demand pattern before going into detail.",
            "Then use the station view to see that the network is not one uniform group, so one average station is not enough.",
            "Use the system forecast page to judge what is useful at each horizon: short term, medium term, and longer term.",
            "SUse the station forecast page to test one global station-day workflow and check results by slice, not just one overall score.",
            "Read the dashboard as one short story: signal first, station mix second, system forecast third, station forecast last."
        ],
        tone="accent",
        editor=editor,
    )

    section_header(
        "Reading order",
        "The dashboard works best as one short narrative instead of four isolated pages.",
        editor=editor,
    )
    story_box(
        "1. Start with the system signal",
        "Read the system-level diagnosis first to understand trend, seasonality, spread, and regime movement in the aggregate demand path.",
        editor=editor,
    )
    story_box(
        "2. Then read the station portfolio",
        "Move to station-level diagnosis to see how maturity, sparsity, intermittency, and category mix change the meaning of any overall score.",
        editor=editor,
    )
    story_box(
        "3. Then judge the system forecast by horizon",
        "Use the system-level forecasting page to decide what is operationally usable now, what is tactical, and what should stay directional only.",
        editor=editor,
    )
    story_box(
        "4. End with station forecast slices",
        "Use the station-level forecasting page to decide whether one global workflow is holding across the mature core and the weaker-signal tails.",
        editor=editor,
    )

    section_header(
        "How updates work",
        "Every page now starts with a standard update control that runs the canonical package CLI for that workflow.",
        editor=editor,
    )
    story_box(
        "Operational rule",
        "When the source data or outputs change, use the update control at the top of the relevant page. The dashboard then reloads from the refreshed diagnosis or forecasting artifacts instead of from a separate hidden path.",
        editor=editor,
    )
    editor.render_sidebar()
