from __future__ import annotations

import argparse

import pandas as pd

from system_level.common.cli_utils import (
    default_forecast_package_report,
    emit_model_report,
    emit_notes,
    emit_package_report,
    emit_report,
    emit_summary,
    make_progress_logger,
    runtime_environment_notes,
    runtime_environment_report,
)
from station_level.forecasting.config import load_station_level_config
from station_level.forecasting.data import ensure_output_directories, write_dataframe
from station_level.forecasting.evaluation import (
    build_recommendation_table,
    plot_model_comparison,
    summarize_backtest_metrics,
)
from station_level.forecasting.models import station_model_runtime_notes, station_model_runtime_report
from station_level.forecasting.pipeline import (
    build_station_level_artifacts,
    run_station_level_backtest_stage,
    run_station_level_pipeline,
    run_station_level_production_stage,
    write_station_level_backtest_outputs,
    write_station_level_feature_artifacts,
    write_station_level_production_outputs,
)


def _load_calibration_or_backtest(config, requested_model: str, tune: bool, verbose: bool = False):
    directories = ensure_output_directories(config)
    calibration_path = directories["metrics"] / "station_level_interval_calibration.csv"
    if calibration_path.exists():
        return directories, pd.read_csv(calibration_path)

    progress = make_progress_logger(verbose, prefix="station-forecast")
    artifacts = build_station_level_artifacts(config)
    write_station_level_feature_artifacts(directories, artifacts)
    backtest_outputs = run_station_level_backtest_stage(
        config,
        artifacts["panel"],
        artifacts["slice_lookup"],
        artifacts["feature_frame"],
        model=requested_model,
        tune=tune,
        progress=progress,
    )
    write_station_level_backtest_outputs(directories, backtest_outputs)
    return directories, backtest_outputs["interval_calibration"]


def _selected_model_keys_or_raise(config, requested_model: str) -> list[str]:
    if str(requested_model).lower() != "all":
        raise SystemExit("Station forecast CLI currently supports only --model all.")
    return config.enabled_model_keys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Station-level forecasting CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("build-panel", "build-slices", "build-features", "backtest", "evaluate", "run-all", "doctor"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", default="configs/station_level/config.yaml")
        subparser.add_argument("--verbose", action="store_true", help="Print stage and fold/model progress.")
        if command in {"backtest", "run-all"}:
            subparser.add_argument("--model", default="all")
            subparser.add_argument("--tune", action="store_true")

    train = subparsers.add_parser("train")
    train.add_argument("--config", default="configs/station_level/config.yaml")
    train.add_argument("--model", default="all")
    train.add_argument("--tune", action="store_true")
    train.add_argument("--verbose", action="store_true", help="Print stage and fold/model progress.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_station_level_config(args.config)
    directories = ensure_output_directories(config)
    progress = make_progress_logger(bool(getattr(args, "verbose", False)), prefix="station-forecast")
    emit_notes(runtime_environment_notes(), prefix="warning")
    emit_notes(station_model_runtime_notes(config), prefix="warning")
    if bool(getattr(args, "verbose", False)) or args.command == "doctor":
        emit_report("runtime_environment", runtime_environment_report())
        emit_package_report(default_forecast_package_report())
        emit_model_report(station_model_runtime_report(config).to_dict(orient="records"))

    if args.command == "doctor":
        emit_summary("Station-level runtime checks completed.", {"config": args.config})
        return

    if args.command in {"build-panel", "build-slices", "build-features"}:
        progress("Preparing station-level artifacts.")
        artifacts = build_station_level_artifacts(config)
        if args.command == "build-panel":
            output_path = directories["feature_artifacts"] / "station_level_observed_panel.csv"
            write_dataframe(artifacts["observed_daily"], output_path)
            emit_summary("Station-level observed panel written.", {"path": output_path, "rows": len(artifacts["observed_daily"])})
            return
        if args.command == "build-slices":
            output_path = directories["feature_artifacts"] / "station_level_slice_lookup.csv"
            write_dataframe(artifacts["slice_lookup"], output_path)
            emit_summary("Station-level slice lookup written.", {"path": output_path, "rows": len(artifacts["slice_lookup"])})
            return
        output_path = directories["feature_artifacts"] / "station_level_features.csv"
        write_dataframe(artifacts["feature_frame"], output_path)
        emit_summary("Station-level features written.", {"path": output_path, "rows": len(artifacts["feature_frame"])})
        return

    if args.command == "backtest":
        progress("Preparing station-level artifacts.")
        artifacts = build_station_level_artifacts(config)
        write_station_level_feature_artifacts(directories, artifacts)
        outputs = run_station_level_backtest_stage(
            config,
            artifacts["panel"],
            artifacts["slice_lookup"],
            artifacts["feature_frame"],
            model=args.model,
            tune=args.tune,
            progress=progress,
        )
        write_station_level_backtest_outputs(directories, outputs)
        emit_summary(
            "Station-level backtests completed.",
            {
                "metric_rows": len(outputs["backtest_metrics"]),
                "forecast_rows": len(outputs["backtest_forecasts"]),
                "output_root": config.output_root,
            },
        )
        return

    if args.command == "train":
        progress("Preparing station-level artifacts.")
        artifacts = build_station_level_artifacts(config)
        write_station_level_feature_artifacts(directories, artifacts)
        _, calibration = _load_calibration_or_backtest(config, args.model, args.tune, verbose=args.verbose)
        model_keys = _selected_model_keys_or_raise(config, args.model)
        outputs = run_station_level_production_stage(
            config,
            artifacts["panel"],
            artifacts["slice_lookup"],
            artifacts["feature_frame"],
            calibration,
            model_keys,
            tune=args.tune or config.tune_enabled,
            requested_model=args.model,
            progress=progress,
        )
        write_station_level_production_outputs(
            directories,
            outputs,
            config,
            requested_model=args.model,
            enabled_model_keys=model_keys,
            tune=args.tune or config.tune_enabled,
            runtime_metadata=None,
        )
        emit_summary(
            "Station-level production forecasts completed.",
            {"forecast_rows": len(outputs["future_forecasts"]), "output_root": config.output_root},
        )
        return

    if args.command == "evaluate":
        metrics_path = directories["backtests"] / "station_level_fold_metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Expected backtest metrics at {metrics_path}. Run `backtest` first.")
        metrics = pd.read_csv(metrics_path)
        summary = summarize_backtest_metrics(metrics)
        recommendations = build_recommendation_table(summary)
        write_dataframe(summary, directories["metrics"] / "station_level_model_comparison.csv")
        write_dataframe(recommendations, directories["metrics"] / "station_level_recommended_models.csv")
        plot_model_comparison(summary, directories["figures"] / "station_level_model_comparison.png")
        emit_summary("Station-level evaluation completed.", {"rows": len(summary), "output_root": config.output_root})
        return

    if args.command == "run-all":
        summary = run_station_level_pipeline(config, model=args.model, tune=args.tune, progress=progress)
        emit_summary("Station-level forecasting pipeline completed.", summary)
        return

    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
