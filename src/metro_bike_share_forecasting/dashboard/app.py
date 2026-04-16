from __future__ import annotations

from pathlib import Path

import streamlit as st

from metro_bike_share_forecasting.dashboard.sections import (
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
        """
        <div class="hero-shell">
          <div class="hero-kicker">Metro Bike Share</div>
          <h1 class="hero-title">Diagnosis and Forecast Console</h1>
          <p class="hero-copy">
            One place to review diagnosis outputs and forecasting artifacts across system-level and station-level workflows.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("## Navigate")
    section = st.sidebar.radio(
        "Section",
        options=[
            "System Diagnosis",
            "Station Diagnosis",
            "System Forecast",
            "Station Forecast",
        ],
        label_visibility="collapsed",
    )

    if section == "System Diagnosis":
        render_system_diagnosis()
    elif section == "Station Diagnosis":
        render_station_diagnosis()
    elif section == "System Forecast":
        render_system_forecast()
    else:
        render_station_forecast()


if __name__ == "__main__":
    run_dashboard()
