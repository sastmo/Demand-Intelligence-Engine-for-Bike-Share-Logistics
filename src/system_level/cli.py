from __future__ import annotations

import argparse
from dataclasses import replace

from system_level.common.cli_utils import (
    default_forecast_package_report,
    emit_model_report,
    emit_notes,
    emit_package_report,
    emit_report,
    make_progress_logger,
    runtime_environment_notes,
    runtime_environment_report,
)
from station_level.diagnosis import StationDiagnosisConfig
from station_level.diagnosis.pipeline import build_station_level_diagnosis
from station_level.forecasting import (
    load_station_level_config,
    run_station_level_pipeline,
)
from station_level.forecasting.models import station_model_runtime_notes, station_model_runtime_report
from system_level import load_system_level_config, run_system_level_pipeline
from system_level.diagnosis.cli import run_from_namespace as run_system_diagnosis_from_namespace
from system_level.forecasting.models import system_model_runtime_notes, system_model_runtime_report


def _filter_system_models(config, model: str):
    model = model.lower()
    if model == "all":
        return config
    if model == "baseline":
        return replace(
            config,
            classical_enabled={key: False for key in config.classical_enabled},
            ml_enabled={key: False for key in config.ml_enabled},
        )
    raise SystemExit(f"Unsupported system-level model selection: {model}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run diagnosis or forecasting for system-level and station-level workflows.")
    subparsers = parser.add_subparsers(dest="task", required=True)

    diagnose = subparsers.add_parser("diagnose")
    diagnose.add_argument("--level", choices=["system", "station"], required=True)
    diagnose.add_argument("--dataset", default=None)
    diagnose.add_argument("--target-col", default=None)
    diagnose.add_argument("--time-col", default=None)
    diagnose.add_argument("--segment-type", default=None)
    diagnose.add_argument("--segment-id", default=None)
    diagnose.add_argument("--start-date", default=None)
    diagnose.add_argument("--end-date", default=None)
    diagnose.add_argument("--frequency", default="daily")
    diagnose.add_argument("--output-root", default=None)
    diagnose.add_argument("--synthetic-demo", action="store_true")
    diagnose.add_argument(
        "--anomaly-method",
        choices=["retrospective_centered_mad", "causal_rolling_mad"],
        default="retrospective_centered_mad",
    )
    diagnose.add_argument("--input", default="data/interim/station_level/station_daily.csv")
    diagnose.add_argument("--date-col", default="date")
    diagnose.add_argument("--station-col", default="station_id")
    diagnose.add_argument("--n-clusters", type=int, default=6)

    forecast = subparsers.add_parser("forecast")
    forecast.add_argument("--level", choices=["system", "station"], required=True)
    forecast.add_argument("--model", default="all")
    forecast.add_argument("--config", default=None)
    forecast.add_argument("--tune", action="store_true")
    forecast.add_argument("--verbose", action="store_true", help="Print stage and fold/model progress.")
    doctor = subparsers.add_parser("doctor")
    doctor.add_argument("--level", choices=["system", "station"], required=True)
    doctor.add_argument("--config", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.task == "diagnose" and args.level == "system":
        if args.target_col is None:
            args.target_col = "trip_count"
        if args.frequency is None:
            args.frequency = "daily"
        if args.output_root is None:
            args.output_root = "diagnosis/system_level/outputs"
        result = run_system_diagnosis_from_namespace(args)
        print(f"system diagnosis complete: {result.output_root}")
        return

    if args.task == "diagnose" and args.level == "station":
        written = build_station_level_diagnosis(
            input_path=args.input,
            date_col=args.date_col,
            station_col=args.station_col,
            target_col=args.target_col or "target",
            config=StationDiagnosisConfig(n_clusters=args.n_clusters, cluster_k_values=tuple(sorted({4, 5, 6, 7, args.n_clusters}))),
        )
        print("station diagnosis complete")
        for key, value in written.items():
            print(f"{key}: {value}")
        return

    if args.task == "forecast" and args.level == "system":
        config = load_system_level_config(args.config or "configs/system_level/config.yaml")
        config = _filter_system_models(config, args.model)
        emit_notes(runtime_environment_notes(), prefix="warning")
        emit_notes(system_model_runtime_notes(config), prefix="warning")
        if args.verbose:
            emit_report("runtime_environment", runtime_environment_report())
            emit_package_report(default_forecast_package_report())
            emit_model_report(system_model_runtime_report(config).to_dict(orient="records"))
        summary = run_system_level_pipeline(config, progress=make_progress_logger(args.verbose, prefix="system-forecast"))
        print("system forecast complete")
        for key, value in summary.items():
            print(f"{key}: {value}")
        return

    if args.task == "forecast" and args.level == "station":
        if str(args.model).lower() != "all":
            raise SystemExit("Station forecast CLI currently supports only --model all.")
        config = load_station_level_config(args.config or "configs/station_level/config.yaml")
        emit_notes(runtime_environment_notes(), prefix="warning")
        emit_notes(station_model_runtime_notes(config), prefix="warning")
        if args.verbose:
            emit_report("runtime_environment", runtime_environment_report())
            emit_package_report(default_forecast_package_report())
            emit_model_report(station_model_runtime_report(config).to_dict(orient="records"))
        summary = run_station_level_pipeline(
            config,
            model=args.model,
            tune=args.tune,
            progress=make_progress_logger(args.verbose, prefix="station-forecast"),
        )
        print("station forecast complete")
        for key, value in summary.items():
            print(f"{key}: {value}")
        return

    if args.task == "doctor" and args.level == "system":
        config = load_system_level_config(args.config or "configs/system_level/config.yaml")
        emit_report("runtime_environment", runtime_environment_report())
        emit_package_report(default_forecast_package_report())
        emit_model_report(system_model_runtime_report(config).to_dict(orient="records"))
        emit_notes(runtime_environment_notes(), prefix="warning")
        emit_notes(system_model_runtime_notes(config), prefix="warning")
        print("doctor complete")
        return

    if args.task == "doctor" and args.level == "station":
        config = load_station_level_config(args.config or "configs/station_level/config.yaml")
        emit_report("runtime_environment", runtime_environment_report())
        emit_package_report(default_forecast_package_report())
        emit_model_report(station_model_runtime_report(config).to_dict(orient="records"))
        emit_notes(runtime_environment_notes(), prefix="warning")
        emit_notes(station_model_runtime_notes(config), prefix="warning")
        print("doctor complete")
        return

    raise SystemExit("Unsupported command.")


if __name__ == "__main__":
    main()
