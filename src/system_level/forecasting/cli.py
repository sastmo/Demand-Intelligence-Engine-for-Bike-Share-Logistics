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
from system_level.forecasting.config import load_system_level_config
from system_level.forecasting.data import ensure_output_directories, write_dataframe
from system_level.forecasting.evaluation import (
    build_recommendation_table,
    plot_model_comparison,
    summarize_backtest_metrics,
)
from system_level.forecasting.pipeline import (
    build_external_feature_artifact,
    build_feature_artifact,
    build_time_index_and_target_artifact,
    run_system_level_backtest_stage,
    run_system_level_pipeline,
    run_system_level_production_stage,
    write_system_level_backtest_outputs,
    write_system_level_production_outputs,
)
from system_level.forecasting.models import system_model_runtime_notes, system_model_runtime_report


def _load_calibration_or_backtest(config, verbose: bool = False):
    directories = ensure_output_directories(config)
    calibration_path = directories["metrics"] / "system_level_interval_calibration.csv"
    if calibration_path.exists():
        return directories, pd.read_csv(calibration_path)

    progress = make_progress_logger(verbose, prefix="system-forecast")
    target = build_time_index_and_target_artifact(config)
    external = build_external_feature_artifact(config)
    backtest_outputs = run_system_level_backtest_stage(target, external, config, progress=progress)
    write_system_level_backtest_outputs(directories, backtest_outputs)
    return directories, backtest_outputs["interval_calibration"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="System-level forecasting CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("build-target", "build-external", "build-features", "backtest", "evaluate", "run-all", "doctor"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", default="configs/system_level/config.yaml")
        subparser.add_argument("--verbose", action="store_true", help="Print stage and fold/model progress.")

    train = subparsers.add_parser("train")
    train.add_argument("--config", default="configs/system_level/config.yaml")
    train.add_argument("--family", choices=["all", "baselines", "classical", "ml"], default="all")
    train.add_argument("--verbose", action="store_true", help="Print stage and fold/model progress.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_system_level_config(args.config)
    directories = ensure_output_directories(config)
    progress = make_progress_logger(bool(getattr(args, "verbose", False)), prefix="system-forecast")
    emit_notes(runtime_environment_notes(), prefix="warning")
    emit_notes(system_model_runtime_notes(config), prefix="warning")
    if bool(getattr(args, "verbose", False)) or args.command == "doctor":
        emit_report("runtime_environment", runtime_environment_report())
        emit_package_report(default_forecast_package_report())
        emit_model_report(system_model_runtime_report(config).to_dict(orient="records"))

    if args.command == "doctor":
        emit_summary("System-level runtime checks completed.", {"config": args.config})
        return

    if args.command == "build-target":
        frame = build_time_index_and_target_artifact(config)
        output_path = directories["feature_artifacts"] / "system_level_target.csv"
        write_dataframe(frame, output_path)
        emit_summary("System-level target written.", {"path": output_path, "rows": len(frame)})
        return

    if args.command == "build-external":
        frame = build_external_feature_artifact(config)
        output_path = directories["feature_artifacts"] / "system_level_external_features.csv"
        write_dataframe(frame, output_path)
        emit_summary("System-level external features written.", {"path": output_path, "rows": len(frame)})
        return

    if args.command == "build-features":
        target = build_time_index_and_target_artifact(config)
        external = build_external_feature_artifact(config)
        features = build_feature_artifact(target, external, config)
        output_path = directories["feature_artifacts"] / "system_level_features.csv"
        write_dataframe(features, output_path)
        emit_summary("System-level features written.", {"path": output_path, "rows": len(features)})
        return

    if args.command == "backtest":
        target = build_time_index_and_target_artifact(config)
        external = build_external_feature_artifact(config)
        outputs = run_system_level_backtest_stage(target, external, config, progress=progress)
        write_system_level_backtest_outputs(directories, outputs)
        emit_summary(
            "System-level backtests completed.",
            {
                "metric_rows": len(outputs["backtest_metrics"]),
                "forecast_rows": len(outputs["backtest_forecasts"]),
                "output_root": config.output_root,
            },
        )
        return

    if args.command == "train":
        target = build_time_index_and_target_artifact(config)
        external = build_external_feature_artifact(config)
        _, calibration = _load_calibration_or_backtest(config, verbose=args.verbose)
        selected_family = None if args.family == "all" else args.family
        outputs = run_system_level_production_stage(target, external, config, calibration, family=selected_family, progress=progress)
        write_system_level_production_outputs(directories, outputs, config, family=selected_family, runtime_metadata=None)
        emit_summary(
            "System-level production forecasts completed.",
            {"forecast_rows": len(outputs["production_forecasts"]), "family": args.family, "output_root": config.output_root},
        )
        return

    if args.command == "evaluate":
        metrics_path = directories["backtests"] / "system_level_fold_metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Expected backtest metrics at {metrics_path}. Run `backtest` first.")
        metrics = pd.read_csv(metrics_path, parse_dates=["train_start", "train_end", "test_start", "test_end"])
        summary = summarize_backtest_metrics(metrics)
        recommendations = build_recommendation_table(summary)
        write_dataframe(summary, directories["metrics"] / "system_level_model_comparison.csv")
        write_dataframe(recommendations, directories["metrics"] / "system_level_recommended_models.csv")
        plot_model_comparison(summary, directories["figures"] / "system_level_model_comparison.png")
        emit_summary("System-level evaluation completed.", {"rows": len(summary), "output_root": config.output_root})
        return

    if args.command == "run-all":
        summary = run_system_level_pipeline(config, progress=progress)
        emit_summary("System-level forecasting pipeline completed.", summary)
        return

    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
