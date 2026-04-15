from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def detect_anomalies(values: pd.Series, window: int, threshold: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    adaptive_window = max(int(window), 5)
    rolling_median = numeric.rolling(adaptive_window, center=True, min_periods=max(3, adaptive_window // 2)).median()
    rolling_median = rolling_median.bfill().ffill()
    residual = numeric - rolling_median
    mad = residual.abs().rolling(adaptive_window, center=True, min_periods=max(3, adaptive_window // 2)).median()
    mad = mad.replace(0, np.nan).bfill().ffill()
    robust_score = 0.6745 * residual / mad
    outlier_flag = robust_score.abs() > threshold

    detail = pd.DataFrame(
        {
            "rolling_median": rolling_median,
            "residual": residual,
            "robust_score": robust_score.fillna(0.0),
            "outlier_flag": outlier_flag.fillna(False),
        }
    )
    summary = {
        "outlier_count": int(outlier_flag.fillna(False).sum()),
        "outlier_share": float(outlier_flag.fillna(False).mean()),
    }
    return detail, summary
