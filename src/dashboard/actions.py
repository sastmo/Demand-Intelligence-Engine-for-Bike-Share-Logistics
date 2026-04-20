from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass
from textwrap import dedent

import streamlit as st

from dashboard.data import repo_root


@dataclass(frozen=True)
class PipelineAction:
    title: str
    description: str
    commands: tuple[tuple[str, ...], ...]


def _python_command(*parts: str) -> tuple[str, ...]:
    return (sys.executable, *parts)


def action_for_page(page_key: str) -> PipelineAction:
    actions = {
        "overview": PipelineAction(
            title="Refresh all outputs",
            description="Run the diagnosis and forecasting pipelines in the standard order so the dashboard reloads from a coherent set of artifacts.",
            commands=(
                _python_command(
                    "-m",
                    "system_level.cli",
                    "diagnose",
                    "--level",
                    "system",
                    "--dataset",
                    "data/processed/daily_aggregate.csv.gz",
                    "--time-col",
                    "bucket_start",
                    "--target-col",
                    "trip_count",
                    "--segment-type",
                    "system_total",
                    "--segment-id",
                    "all",
                    "--frequency",
                    "daily",
                    "--output-root",
                    "diagnosis/system_level/outputs",
                ),
                _python_command(
                    "-m",
                    "system_level.cli",
                    "diagnose",
                    "--level",
                    "station",
                    "--input",
                    "data/interim/station_level/station_daily.csv",
                    "--date-col",
                    "date",
                    "--station-col",
                    "station_id",
                    "--target-col",
                    "target",
                ),
                _python_command(
                    "-m",
                    "system_level.cli",
                    "forecast",
                    "--level",
                    "system",
                    "--config",
                    "configs/system_level/config.yaml",
                ),
                _python_command(
                    "-m",
                    "system_level.cli",
                    "forecast",
                    "--level",
                    "station",
                    "--config",
                    "configs/station_level/config.yaml",
                ),
            ),
        ),
        "system_diagnosis": PipelineAction(
            title="Update system-level diagnosis",
            description="Rebuild the aggregate diagnosis outputs from the processed daily aggregate series.",
            commands=(
                _python_command(
                    "-m",
                    "system_level.cli",
                    "diagnose",
                    "--level",
                    "system",
                    "--dataset",
                    "data/processed/daily_aggregate.csv.gz",
                    "--time-col",
                    "bucket_start",
                    "--target-col",
                    "trip_count",
                    "--segment-type",
                    "system_total",
                    "--segment-id",
                    "all",
                    "--frequency",
                    "daily",
                    "--output-root",
                    "diagnosis/system_level/outputs",
                ),
            ),
        ),
        "station_diagnosis": PipelineAction(
            title="Update station-level diagnosis",
            description="Rebuild the station portfolio analysis from the canonical station-day panel.",
            commands=(
                _python_command(
                    "-m",
                    "system_level.cli",
                    "diagnose",
                    "--level",
                    "station",
                    "--input",
                    "data/interim/station_level/station_daily.csv",
                    "--date-col",
                    "date",
                    "--station-col",
                    "station_id",
                    "--target-col",
                    "target",
                ),
            ),
        ),
        "system_forecast": PipelineAction(
            title="Update system-level forecasting",
            description="Run the canonical system-level forecasting pipeline from the system config.",
            commands=(
                _python_command(
                    "-m",
                    "system_level.cli",
                    "forecast",
                    "--level",
                    "system",
                    "--config",
                    "configs/system_level/config.yaml",
                ),
            ),
        ),
        "station_forecast": PipelineAction(
            title="Update station-level forecasting",
            description="Run the canonical station-level forecasting pipeline from the station config.",
            commands=(
                _python_command(
                    "-m",
                    "system_level.cli",
                    "forecast",
                    "--level",
                    "station",
                    "--config",
                    "configs/station_level/config.yaml",
                ),
            ),
        ),
    }
    return actions[page_key]


def _result_key(page_key: str) -> str:
    return f"dashboard_action_result:{page_key}"


def _format_command_block(commands: tuple[tuple[str, ...], ...]) -> str:
    return "\n".join(f"$ {shlex.join(command)}" for command in commands)


def _run_action(action: PipelineAction) -> dict[str, object]:
    logs: list[str] = []
    success = True
    for index, command in enumerate(action.commands, start=1):
        completed = subprocess.run(
            list(command),
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=False,
        )
        logs.append(f"$ {shlex.join(command)}")
        if completed.stdout.strip():
            logs.append(completed.stdout.strip())
        if completed.stderr.strip():
            logs.append(completed.stderr.strip())
        if completed.returncode != 0:
            success = False
            logs.append(f"Command {index} failed with exit code {completed.returncode}.")
            break
    return {"success": success, "logs": "\n\n".join(logs).strip()}


def render_update_toolbar(page_key: str) -> None:
    action = action_for_page(page_key)
    st.markdown(
        dedent(
            f"""
            <div class="action-shell">
              <div class="action-title">{action.title}</div>
              <div class="action-copy">{action.description}</div>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )
    button_col, command_col = st.columns([0.28, 0.72], vertical_alignment="bottom")
    with button_col:
        if st.button("Run update pipeline", key=f"dashboard_run:{page_key}", width="stretch"):
            with st.spinner("Running pipeline update..."):
                st.session_state[_result_key(page_key)] = _run_action(action)
            st.rerun()
    with command_col:
        with st.expander("Show canonical command", expanded=False):
            st.code(_format_command_block(action.commands), language="bash")

    result = st.session_state.get(_result_key(page_key))
    if not result:
        return
    if bool(result.get("success")):
        st.success("Pipeline update completed.")
    else:
        st.error("Pipeline update failed. See the command log below.")
    with st.expander("Latest pipeline log", expanded=not bool(result.get("success"))):
        st.code(str(result.get("logs", "")) or "No command log available.", language="bash")
