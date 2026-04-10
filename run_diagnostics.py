from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from forecasting_diagnostics import DiagnosticConfig, run_forecasting_diagnostics


DEFAULT_OUTPUT_ROOT = Path("outputs/diagnostics/tmp")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run forecasting diagnostics and save outputs into outputs/diagnostics/tmp.")
    parser.add_argument("dataset", nargs="?", help="Path to a CSV dataset.")
    parser.add_argument("--target-col", default="value", help="Target column name.")
    parser.add_argument("--time-col", default=None, help="Time column name. Omit when using a datetime index.")
    parser.add_argument("--use-index", action="store_true", help="Treat the first CSV column as the datetime index.")
    parser.add_argument("--frequency", default=None, help="Optional logical frequency label such as daily or hourly.")
    parser.add_argument("--series-name", default=None, help="Optional series name for the report.")
    parser.add_argument(
        "--seasonal-periods",
        nargs="*",
        type=int,
        default=None,
        help="Optional candidate seasonal periods, for example --seasonal-periods 7 30 365",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Diagnostics output root.")
    parser.add_argument("--synthetic-demo", action="store_true", help="Run the diagnostics on a built-in synthetic seasonal series.")
    return parser


def _build_synthetic_demo() -> pd.DataFrame:
    timestamps = pd.date_range("2021-01-01", periods=365, freq="D")
    index = np.arange(len(timestamps))
    values = (
        120
        + 0.18 * index
        + 18 * np.sin(2 * np.pi * index / 7)
        + 7 * np.cos(2 * np.pi * index / 30)
        + np.where(index > 210, 22, 0)
    )
    return pd.DataFrame({"timestamp": timestamps, "value": values})


def _load_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, str | None]:
    if args.synthetic_demo:
        return _build_synthetic_demo(), "timestamp"
    if not args.dataset:
        raise SystemExit("Provide a dataset path or use --synthetic-demo.")

    dataset_path = Path(args.dataset)
    if args.use_index:
        frame = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
        return frame, None
    frame = pd.read_csv(dataset_path)
    return frame, args.time_col


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    frame, resolved_time_col = _load_dataset(args)
    output_root = Path(args.output_root)
    series_name = args.series_name or ("synthetic_demo" if args.synthetic_demo else Path(args.dataset).stem)
    config = DiagnosticConfig(
        series_name=series_name,
        target_col=args.target_col,
        time_col=resolved_time_col,
        frequency=args.frequency,
        candidate_periods=tuple(args.seasonal_periods or ()),
        output_root=output_root,
        clean_output=True,
    )
    result = run_forecasting_diagnostics(frame, config)

    print("Forecasting diagnostics completed.")
    print(f"Output root: {result.output_root}")
    print(f"Figures: {result.figures_dir}")
    print(f"Tables: {result.tables_dir}")
    print(f"Report: {result.report_path}")
    print("Top recommendations:")
    for item in result.summary.get("recommendations", [])[:5]:
        print(f"- {item}")


if __name__ == "__main__":
    main()
