from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


def summarize_distribution(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if numeric.empty:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "zero_share": None,
            "skewness": None,
            "kurtosis": None,
            "is_count_like": False,
            "is_non_negative": False,
            "is_intermittent_like": False,
        }

    zero_share = float((numeric == 0).mean())
    count_like = bool(np.allclose(numeric, np.round(numeric)) and (numeric >= 0).all())
    return {
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "std": float(numeric.std(ddof=0)),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "zero_share": zero_share,
        "skewness": float(skew(numeric, bias=False)) if len(numeric) > 2 else None,
        "kurtosis": float(kurtosis(numeric, fisher=True, bias=False)) if len(numeric) > 3 else None,
        "is_count_like": count_like,
        "is_non_negative": bool((numeric >= 0).all()),
        "is_intermittent_like": bool(zero_share >= 0.3),
    }
