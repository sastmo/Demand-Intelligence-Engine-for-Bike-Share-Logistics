from __future__ import annotations

from typing import Iterable

import pandas as pd


def validate_required_columns(frame: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Raise a clear error when expected columns are missing."""

    missing = sorted(set(required_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")
