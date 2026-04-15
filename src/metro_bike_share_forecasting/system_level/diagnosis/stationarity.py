from __future__ import annotations

import warnings
from typing import Any

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def run_stationarity_checks(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(numeric) < 12:
        return {"adf_pvalue": None, "kpss_pvalue": None, "stationarity_assessment": "insufficient_history"}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf_pvalue = float(adfuller(numeric, autolag="AIC")[1])
    except Exception:
        adf_pvalue = None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_pvalue = float(kpss(numeric, regression="c", nlags="auto")[1])
    except Exception:
        kpss_pvalue = None

    if adf_pvalue is not None and kpss_pvalue is not None:
        if adf_pvalue <= 0.05 and kpss_pvalue > 0.05:
            assessment = "likely_stationary"
        elif adf_pvalue > 0.05 and kpss_pvalue <= 0.05:
            assessment = "likely_nonstationary"
        else:
            assessment = "mixed_signal"
    elif adf_pvalue is not None:
        assessment = "likely_stationary" if adf_pvalue <= 0.05 else "likely_nonstationary"
    else:
        assessment = "unknown"

    return {
        "adf_pvalue": adf_pvalue,
        "kpss_pvalue": kpss_pvalue,
        "stationarity_assessment": assessment,
    }
