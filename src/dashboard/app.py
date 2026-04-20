from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import streamlit as st

from dashboard.pages import (
    render_overview,
    render_station_diagnosis,
    render_station_forecast,
    render_system_diagnosis,
    render_system_forecast,
)


def _load_css() -> None:
    css_path = Path(__file__).resolve().parent / "assets" / "dashboard.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def run_dashboard() -> None:
    st.set_page_config(
        page_title="Metro Bike Share Forecast Console",
        page_icon="🚲",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _load_css()

    st.markdown(
        dedent(
            """
            <div class="hero-shell">
              <div class="hero-kicker">Metro Bike Share</div>
              <h1 class="hero-title">Diagnosis and Forecast Console</h1>
              <p class="hero-copy">
              </p>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("## Navigate")
    section = st.sidebar.radio(
        "Section",
        options=[
            "Intro",
            "System-Level Diagnosis",
            "Station-Level Diagnosis",
            "System-Level Forecasting",
            "Station-Level Forecasting",
        ],
        label_visibility="collapsed",
    )

    if section == "Intro":
        render_overview()
    elif section == "System-Level Diagnosis":
        render_system_diagnosis()
    elif section == "Station-Level Diagnosis":
        render_station_diagnosis()
    elif section == "System-Level Forecasting":
        render_system_forecast()
    else:
        render_station_forecast()


if __name__ == "__main__":
    run_dashboard()
