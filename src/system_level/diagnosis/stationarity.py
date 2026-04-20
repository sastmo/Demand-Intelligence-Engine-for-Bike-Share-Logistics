from __future__ import annotations

import warnings
from typing import Any

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def run_stationarity_checks(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(numeric) < 12:
        return {
            "adf_pvalue": None,
            "kpss_pvalue": None,
            "adf_status": "not_run_insufficient_history",
            "kpss_status": "not_run_insufficient_history",
            "stationarity_test_status": "insufficient_history",
            "stationarity_assessment": "insufficient_history",
        }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf_pvalue = float(adfuller(numeric, autolag="AIC")[1])
        adf_status = "ok"
    except Exception:
        adf_pvalue = None
        adf_status = "failed"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_pvalue = float(kpss(numeric, regression="c", nlags="auto")[1])
        kpss_status = "ok"
    except Exception:
        kpss_pvalue = None
        kpss_status = "failed"

    if adf_pvalue is not None and kpss_pvalue is not None:
        if adf_pvalue <= 0.05 and kpss_pvalue > 0.05:
            assessment = "screening_suggests_stationary"
        elif adf_pvalue > 0.05 and kpss_pvalue <= 0.05:
            assessment = "screening_suggests_nonstationary"
        else:
            assessment = "mixed_or_uncertain"
        stationarity_test_status = "ok"
    elif adf_pvalue is not None:
        assessment = "screening_suggests_stationary" if adf_pvalue <= 0.05 else "screening_suggests_nonstationary"
        stationarity_test_status = "partial"
    else:
        assessment = "unknown"
        stationarity_test_status = "unavailable"

    return {
        "adf_pvalue": adf_pvalue,
        "kpss_pvalue": kpss_pvalue,
        "adf_status": adf_status,
        "kpss_status": kpss_status,
        "stationarity_test_status": stationarity_test_status,
        "stationarity_assessment": assessment,
    }
