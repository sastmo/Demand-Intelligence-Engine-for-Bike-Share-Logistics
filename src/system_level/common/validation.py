from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


KNOWN_FUTURE_PREFIXES = ("known_", "forecast_", "planned_", "calendar_", "event_")


def validate_known_future_external_frame(
    frame: pd.DataFrame,
    date_column: str = "date",
    allowed_prefixes: Iterable[str] = KNOWN_FUTURE_PREFIXES,
) -> list[str]:
    if frame.empty:
        return []
    if date_column not in frame.columns:
        raise ValueError(f"Known-future feature frame must contain `{date_column}`.")

    allowed = tuple(str(prefix) for prefix in allowed_prefixes)
    feature_columns = [column for column in frame.columns if column != date_column]
    invalid = [
        column
        for column in feature_columns
        if not any(str(column).startswith(prefix) for prefix in allowed)
    ]
    if invalid:
        raise ValueError(
            "External forecast features must be explicitly named as known-future signals. "
            f"Rename these columns to use one of {allowed}: {sorted(invalid)}"
        )
    return feature_columns


def assert_known_future_feature_coverage(
    frame: pd.DataFrame,
    feature_columns: list[str],
    date_column: str = "date",
) -> None:
    if frame.empty or not feature_columns:
        return
    missing = frame.loc[frame[feature_columns].isna().any(axis=1), date_column]
    if missing.empty:
        return
    sample = ", ".join(pd.to_datetime(missing).dt.strftime("%Y-%m-%d").head(3).tolist())
    raise ValueError(
        "Known-future external features must be available for every modeled date. "
        f"Missing values were found on dates including: {sample}"
    )


def time_ordered_validation_split(
    frame: pd.DataFrame,
    date_column: str,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame.copy(), frame.copy()
    cutoff = pd.to_datetime(frame[date_column]).max() - pd.Timedelta(days=int(horizon))
    train = frame.loc[pd.to_datetime(frame[date_column]) <= cutoff].copy()
    valid = frame.loc[pd.to_datetime(frame[date_column]) > cutoff].copy()
    if train.empty or valid.empty:
        return frame.copy(), pd.DataFrame(columns=frame.columns)
    return train, valid
