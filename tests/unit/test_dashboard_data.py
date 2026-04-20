from __future__ import annotations

import tempfile
import unittest

import pandas as pd

from dashboard.data import prediction_column, station_forecast_chart_frame


class DashboardDataTests(unittest.TestCase):
    def test_dashboard_data_import_resolves_from_canonical_package(self) -> None:
        self.assertTrue(callable(prediction_column))

    def test_prediction_column_prefers_point_forecast(self) -> None:
        frame = pd.DataFrame({"prediction": [1.0], "point_forecast": [2.0]})
        self.assertEqual(prediction_column(frame), "point_forecast")

    def test_station_forecast_chart_frame_merges_history_and_forecast(self) -> None:
        observed = pd.DataFrame(
            {
                "station_id": ["3001", "3001", "3001"],
                "date": pd.to_datetime(["2024-12-29", "2024-12-30", "2024-12-31"]),
                "target": [4.0, 5.0, 6.0],
            }
        )
        future = pd.DataFrame(
            {
                "station_id": ["3001", "3001"],
                "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                "prediction": [7.0, 8.0],
                "lower_80": [5.0, 6.0],
                "upper_80": [9.0, 10.0],
                "model_name": ["deepar", "deepar"],
                "horizon": [7, 7],
            }
        )

        chart = station_forecast_chart_frame(observed, future, "3001", "deepar", 7, history_days=10)
        self.assertEqual(len(chart), 5)
        self.assertIn("observed", chart.columns)
        self.assertIn("forecast", chart.columns)
        self.assertEqual(chart["forecast"].dropna().tolist(), [7.0, 8.0])


if __name__ == "__main__":
    unittest.main()
