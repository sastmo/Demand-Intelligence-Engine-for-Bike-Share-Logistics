from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _centered_rolling_mad(numeric: pd.Series, adaptive_window: int) -> tuple[pd.Series, pd.Series]:
    rolling_median = numeric.rolling(adaptive_window, center=True, min_periods=max(3, adaptive_window // 2)).median()
    rolling_median = rolling_median.bfill().ffill()
    residual = numeric - rolling_median
    mad = residual.abs().rolling(adaptive_window, center=True, min_periods=max(3, adaptive_window // 2)).median()
    mad = mad.replace(0, np.nan).bfill().ffill()
    return rolling_median, mad


def _causal_rolling_mad(numeric: pd.Series, adaptive_window: int) -> tuple[pd.Series, pd.Series]:
    rolling_median = numeric.shift(1).rolling(adaptive_window, min_periods=max(3, adaptive_window // 2)).median()
    rolling_median = rolling_median.combine_first(numeric.expanding(min_periods=1).median().shift(1))
    residual = numeric - rolling_median
    mad = residual.abs().shift(1).rolling(adaptive_window, min_periods=max(3, adaptive_window // 2)).median()
    mad = mad.combine_first(residual.abs().expanding(min_periods=1).median().shift(1))
    mad = mad.replace(0, np.nan)
    return rolling_median, mad


def detect_anomalies(
    values: pd.Series,
    window: int,
    threshold: float,
    method: str = "retrospective_centered_mad",
    is_imputed: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    adaptive_window = max(int(window), 5)
    imputed = is_imputed.astype(bool).reindex(numeric.index).fillna(False) if is_imputed is not None else pd.Series(False, index=numeric.index)

    if method == "retrospective_centered_mad":
        rolling_median, mad = _centered_rolling_mad(numeric, adaptive_window)
        anomaly_scope = "retrospective_offline"
    elif method == "causal_rolling_mad":
        rolling_median, mad = _causal_rolling_mad(numeric, adaptive_window)
        anomaly_scope = "causal_monitoring"
    else:
        raise ValueError(f"Unsupported anomaly method: {method}")

    residual = numeric - rolling_median
    robust_score = 0.6745 * residual / mad
    raw_outlier_flag = robust_score.abs() > threshold
    suppressed_imputed_flag = raw_outlier_flag.fillna(False) & imputed
    outlier_flag = raw_outlier_flag.fillna(False) & ~imputed

    detail = pd.DataFrame(
        {
            "rolling_median": rolling_median,
            "residual": residual,
            "robust_score": robust_score.fillna(0.0),
            "raw_outlier_flag": raw_outlier_flag.fillna(False),
            "suppressed_imputed_anomaly_flag": suppressed_imputed_flag,
            "outlier_flag": outlier_flag,
            "imputed_flag": imputed,
        }
    )
    summary = {
        "anomaly_method": method,
        "anomaly_scope": anomaly_scope,
        "outlier_count": int(outlier_flag.fillna(False).sum()),
        "outlier_share": float(outlier_flag.fillna(False).mean()),
        "suppressed_imputed_anomaly_count": int(suppressed_imputed_flag.sum()),
    }
    return detail, summary
