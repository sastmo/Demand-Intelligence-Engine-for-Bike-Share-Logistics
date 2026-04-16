from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from metro_bike_share_forecasting.station_level.diagnosis import StationDiagnosisConfig
from metro_bike_share_forecasting.station_level.diagnosis.pipeline import build_station_level_diagnosis
from metro_bike_share_forecasting.station_level.forecasting import (
    load_station_level_config,
    run_station_level_pipeline,
)
from metro_bike_share_forecasting.system_level import load_system_level_config, run_system_level_pipeline
from metro_bike_share_forecasting.system_level.diagnosis import DiagnosticConfig, run_forecasting_diagnostics


def _system_diagnosis_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, str | None]:
    if args.synthetic_demo:
        timestamps = pd.date_range("2021-01-01", periods=365, freq="D")
        index = np.arange(len(timestamps))
        values = 120 + 0.18 * index + 18 * np.sin(2 * np.pi * index / 7) + 7 * np.cos(2 * np.pi * index / 30)
        return pd.DataFrame({"timestamp": timestamps, "value": values}), "timestamp"

    dataset_path = Path(args.dataset or "data/processed/daily_aggregate.csv.gz")
    frame = pd.read_csv(dataset_path, low_memory=False)
    time_col = args.time_col or "bucket_start"
    if args.level == "system" and {"segment_type", "segment_id"}.issubset(frame.columns):
        segment_type = args.segment_type or "system_total"
        segment_id = args.segment_id or "all"
        frame = frame.loc[
            (frame["segment_type"].astype(str) == segment_type)
            & (frame["segment_id"].astype(str) == segment_id)
        ].copy()
    if args.start_date:
        frame = frame.loc[pd.to_datetime(frame[time_col]) >= pd.Timestamp(args.start_date)].copy()
    if args.end_date:
        frame = frame.loc[pd.to_datetime(frame[time_col]) <= pd.Timestamp(args.end_date)].copy()
    return frame, time_col


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
    diagnose.add_argument("--input", default="data/interim/station_level/station_daily.csv")
    diagnose.add_argument("--date-col", default="date")
    diagnose.add_argument("--station-col", default="station_id")
    diagnose.add_argument("--n-clusters", type=int, default=6)

    forecast = subparsers.add_parser("forecast")
    forecast.add_argument("--level", choices=["system", "station"], required=True)
    forecast.add_argument("--model", default="all")
    forecast.add_argument("--config", default=None)
    forecast.add_argument("--tune", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.task == "diagnose" and args.level == "system":
        frame, time_col = _system_diagnosis_frame(args)
        config = DiagnosticConfig(
            series_name=args.segment_id or "system_total",
            target_col=args.target_col or "trip_count",
            time_col=time_col,
            frequency=args.frequency,
            output_root=Path(args.output_root or "diagnosis/system_level_analysis/outputs"),
            clean_output=True,
        )
        result = run_forecasting_diagnostics(frame, config)
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
        summary = run_system_level_pipeline(config)
        print("system forecast complete")
        for key, value in summary.items():
            print(f"{key}: {value}")
        return

    if args.task == "forecast" and args.level == "station":
        if str(args.model).lower() != "all":
            raise SystemExit("Station forecast CLI currently supports only --model all.")
        config = load_station_level_config(args.config or "configs/station_level/config.yaml")
        summary = run_station_level_pipeline(config, model=args.model, tune=args.tune)
        print("station forecast complete")
        for key, value in summary.items():
            print(f"{key}: {value}")
        return

    raise SystemExit("Unsupported command.")


if __name__ == "__main__":
    main()
