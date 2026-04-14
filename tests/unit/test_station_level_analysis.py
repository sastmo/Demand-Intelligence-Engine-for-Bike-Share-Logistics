from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "src"))

from metro_bike_share_forecasting.station_level.diagnosis.config import StationDiagnosisConfig
from metro_bike_share_forecasting.station_level.diagnosis.pipeline import build_station_level_diagnosis


class StationLevelDiagnosisTests(unittest.TestCase):
    def test_station_level_diagnosis_builds_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "station_daily.csv"
            output_dir = root / "station_level_analysis" / "outputs"

            frame = pd.DataFrame(
                {
                    "date": pd.date_range("2024-01-01", periods=21, freq="D").tolist() * 2,
                    "station_id": ["A"] * 21 + ["B"] * 21,
                    "target": [10, 12, 11, 0, 0, 5, 7, 11, 13, 12, 3, 0, 0, 4, 6, 12, 13, 11, 3, 0, 5]
                    + [1, 0, 0, 0, 2, 0, 0, 1, 0, 0, 3, 0, 0, 0, 2, 0, 0, 1, 0, 0, 2],
                }
            )
            frame.to_csv(input_path, index=False)

            written = build_station_level_diagnosis(
                input_path=input_path,
                date_col="date",
                station_col="station_id",
                target_col="target",
                config=StationDiagnosisConfig(output_root=output_dir, top_n=2, n_clusters=2),
            )

            self.assertTrue(Path(written["station_summary_csv"]).exists())
            self.assertTrue(Path(written["station_summary_parquet"]).exists())
            self.assertTrue(Path(written["station_category_summary"]).exists())
            self.assertTrue(Path(written["station_cluster_profile"]).exists())
            self.assertTrue(Path(written["summary_with_clusters"]).exists())
            self.assertTrue(Path(written["report"]).exists())

            summary = pd.read_csv(written["station_summary_csv"])
            self.assertIn("station_category", summary.columns)
            self.assertIn("cluster_label", summary.columns)
            self.assertIn("lag1_autocorr", summary.columns)
            self.assertIn("correlation_with_system_total", summary.columns)
            self.assertEqual(summary["station_id"].nunique(), 2)


if __name__ == "__main__":
    unittest.main()
