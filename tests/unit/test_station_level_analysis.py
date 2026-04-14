from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "src"))

from metro_bike_share_forecasting.station_level.diagnosis.config import StationDiagnosisConfig
from metro_bike_share_forecasting.station_level.diagnosis.pipeline import build_station_level_diagnosis


class StationLevelDiagnosisTests(unittest.TestCase):
    def _build_station_frame(self) -> pd.DataFrame:
        dates = pd.date_range("2023-01-01", periods=420, freq="D")
        rng = np.random.default_rng(42)
        rows: list[dict[str, object]] = []

        profiles = {
            "A": lambda i, d: 42 + 6 * np.sin(2 * np.pi * (d.dayofweek / 7)) + (6 if d.dayofweek < 5 else -4),
            "B": lambda i, d: 28 + 12 * np.sin(2 * np.pi * (i / 30)),
            "C": lambda i, d: 5 if d.dayofweek < 5 else 14,
            "D": lambda i, d: 0 if i % 3 else 3,
            "E": lambda i, d: 20 + (18 if i % 45 == 0 else 0),
            "F": lambda i, d: 9 + 0.02 * i,
            "G": lambda i, d: 16 + 4 * np.cos(2 * np.pi * (i / 14)),
            "H": lambda i, d: 12 + (7 if d.month in {6, 7, 8} else 0),
        }

        for station_id, generator in profiles.items():
            for i, date_value in enumerate(dates):
                level = max(0.0, float(generator(i, date_value)) + float(rng.normal(0, 1.25)))
                rows.append(
                    {
                        "date": date_value,
                        "station_id": station_id,
                        "target": round(level, 2),
                    }
                )

        frame = pd.DataFrame(rows)
        short_history_dates = pd.date_range("2024-01-01", periods=45, freq="D")
        short_history = pd.DataFrame(
            {
                "date": short_history_dates,
                "station_id": "Z",
                "target": [0, 0, 1, 0, 0] * 9,
            }
        )
        return pd.concat([frame, short_history], ignore_index=True)

    def test_station_level_diagnosis_builds_revised_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "station_daily.csv"
            output_dir = root / "station_level_analysis" / "outputs"

            frame = self._build_station_frame()
            frame.to_csv(input_path, index=False)

            written = build_station_level_diagnosis(
                input_path=input_path,
                date_col="date",
                station_col="station_id",
                target_col="target",
                config=StationDiagnosisConfig(
                    output_root=output_dir,
                    top_n=3,
                    n_clusters=3,
                    cluster_k_values=(2, 3),
                    expected_station_count=8,
                ),
            )

            required_keys = {
                "station_inventory",
                "station_summary_csv",
                "station_summary_parquet",
                "station_category_summary",
                "station_cluster_profile",
                "cluster_model_selection",
                "summary_with_clusters",
                "report",
                "category_count_bar_chart",
                "history_days_histogram",
                "cluster_profile_heatmap",
            }
            self.assertTrue(required_keys.issubset(written.keys()))
            for key in required_keys:
                self.assertTrue(Path(written[key]).exists(), msg=f"Missing output for {key}")

            summary = pd.read_csv(written["station_summary_csv"])
            self.assertIn("station_category", summary.columns)
            self.assertIn("history_group", summary.columns)
            self.assertNotIn("cluster_label", summary.columns)

            summary_with_clusters = pd.read_csv(written["summary_with_clusters"])
            self.assertIn("cluster_label", summary_with_clusters.columns)
            self.assertIn("metrics_limited_by_sparsity", summary_with_clusters.columns)
            self.assertEqual(summary_with_clusters["station_id"].nunique(), 9)

            inventory = pd.read_csv(written["station_inventory"])
            self.assertIn("is_zero_almost_always", inventory.columns)
            self.assertIn("appears_active_recently", inventory.columns)

            cluster_selection = pd.read_csv(written["cluster_model_selection"])
            self.assertIn("selected", cluster_selection.columns)
            self.assertGreaterEqual(len(cluster_selection), 1)

            report_text = Path(written["report"]).read_text()
            self.assertIn("Station Universe Validation", report_text)
            self.assertIn("Final Recommendation", report_text)


if __name__ == "__main__":
    unittest.main()
