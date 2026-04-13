from __future__ import annotations

import os
import warnings
from typing import Callable

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - optional dependency
    lgb = None

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - optional dependency
    xgb = None

from sklearn.ensemble import HistGradientBoostingRegressor

from metro_bike_share_forecasting.system_level.config import SystemLevelConfig
from metro_bike_share_forecasting.system_level.features import (
    build_future_dates,
    build_known_future_features,
    build_system_level_features,
    known_future_feature_columns,
    ml_feature_columns,
)


def _clip_nonnegative(values: np.ndarray | pd.Series) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=float), 0.0)


def _fallback_prediction(train_frame: pd.DataFrame, horizon: int, label: str) -> pd.DataFrame:
    value = float(train_frame["target"].iloc[-1])
    dates = build_future_dates(train_frame["date"].iloc[-1], horizon)
    return pd.DataFrame({"date": dates["date"], "prediction": np.repeat(max(value, 0.0), horizon), "model_name": label})


def _seasonal_fallback_prediction(train_frame: pd.DataFrame, horizon: int, label: str) -> pd.DataFrame:
    if len(train_frame) >= 7:
        return seasonal_naive_forecast(train_frame, horizon, 7, label)
    return _fallback_prediction(train_frame, horizon, label)


def _fit_statsmodels_safely(fit_callable):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        fitted = fit_callable()
    return fitted


def _prediction_upper_bound(train_frame: pd.DataFrame) -> float:
    series = train_frame["target"].astype(float).dropna()
    if series.empty:
        return 1_000.0
    recent = series.tail(min(len(series), 90))
    recent_std = float(recent.std(ddof=0)) if len(recent) > 1 else 0.0
    return float(
        max(
            series.max() * 3.0,
            series.quantile(0.99) * 4.0 if len(series) >= 10 else series.max() * 3.0,
            recent.mean() + 10.0 * max(recent_std, 1.0),
            1_000.0,
        )
    )


def _finalize_predictions(
    train_frame: pd.DataFrame,
    future_dates: pd.DataFrame,
    predictions: np.ndarray | pd.Series,
    model_name: str,
    fallback_label: str,
) -> pd.DataFrame:
    values = np.asarray(predictions, dtype=float)
    if (not np.isfinite(values).all()) or (len(values) > 0 and values.max() > _prediction_upper_bound(train_frame)):
        return _seasonal_fallback_prediction(train_frame, len(future_dates), fallback_label)
    values = _clip_nonnegative(values)
    return pd.DataFrame({"date": future_dates["date"], "prediction": values, "model_name": model_name})


def _stable_dynamic_feature_columns(frame: pd.DataFrame) -> list[str]:
    raw_calendar = {"day_of_week", "week_of_year", "month", "quarter", "year", "day_of_month", "day_of_year", "season_code"}
    known_columns = known_future_feature_columns(frame)
    external_numeric = [
        column
        for column in known_columns
        if column not in raw_calendar
        and not column.startswith("weekly_")
        and not column.startswith("yearly_")
        and column not in {"is_weekend", "is_holiday"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    selected = [
        column
        for column in known_columns
        if column in {"is_weekend", "is_holiday"} or column.startswith("weekly_") or column.startswith("yearly_")
    ]
    return sorted(dict.fromkeys(selected + external_numeric))


def naive_forecast(train_frame: pd.DataFrame, horizon: int, config: SystemLevelConfig, external_features: pd.DataFrame) -> pd.DataFrame:
    return _fallback_prediction(train_frame, horizon, "naive")


def seasonal_naive_forecast(train_frame: pd.DataFrame, horizon: int, season_length: int, label: str) -> pd.DataFrame:
    values = train_frame["target"].astype(float).tolist()
    future_dates = build_future_dates(train_frame["date"].iloc[-1], horizon)
    predictions: list[float] = []
    season = min(season_length, len(values)) if values else 1
    for _ in range(horizon):
        prediction = float(values[-season] if len(values) >= season else values[-1])
        prediction = max(prediction, 0.0)
        predictions.append(prediction)
        values.append(prediction)
    return pd.DataFrame({"date": future_dates["date"], "prediction": predictions, "model_name": label})


def ets_forecast(train_frame: pd.DataFrame, horizon: int, config: SystemLevelConfig, external_features: pd.DataFrame) -> pd.DataFrame:
    series = train_frame["target"].astype(float)
    future_dates = build_future_dates(train_frame["date"].iloc[-1], horizon)
    try:
        seasonal = "add" if len(series) >= 14 else None
        seasonal_periods = 7 if seasonal else None
        model = ExponentialSmoothing(
            series,
            trend="add",
            damped_trend=True,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        )
        fitted = model.fit(optimized=True, use_brute=False)
        predictions = fitted.forecast(horizon)
    except Exception:
        return _seasonal_fallback_prediction(train_frame, horizon, "ets_fallback")
    return _finalize_predictions(train_frame, future_dates, predictions, "ets", "ets_fallback")


def sarimax_dynamic_forecast(
    train_frame: pd.DataFrame,
    horizon: int,
    config: SystemLevelConfig,
    external_features: pd.DataFrame,
) -> pd.DataFrame:
    model_frame = build_system_level_features(train_frame.copy(), config, external_features)
    exog_columns = _stable_dynamic_feature_columns(model_frame)
    exog_train = model_frame[exog_columns].fillna(0.0).astype(float)
    future_dates = build_future_dates(train_frame["date"].iloc[-1], horizon)
    exog_future = build_known_future_features(future_dates, config, external_features)[exog_columns].fillna(0.0).astype(float)

    try:
        fitted = _fit_statsmodels_safely(
            lambda: SARIMAX(
                train_frame["target"].astype(float),
                order=(0, 1, 1),
                seasonal_order=(0, 1, 1, 7),
                exog=exog_train,
                trend="n",
                enforce_stationarity=True,
                enforce_invertibility=True,
            ).fit(disp=False)
        )
        predictions = fitted.forecast(steps=horizon, exog=exog_future)
    except Exception:
        return _seasonal_fallback_prediction(train_frame, horizon, "sarimax_dynamic_fallback")

    return _finalize_predictions(
        train_frame,
        future_dates,
        predictions,
        "sarimax_dynamic",
        "sarimax_dynamic_fallback",
    )


def fourier_dynamic_regression_forecast(
    train_frame: pd.DataFrame,
    horizon: int,
    config: SystemLevelConfig,
    external_features: pd.DataFrame,
) -> pd.DataFrame:
    model_frame = build_system_level_features(train_frame.copy(), config, external_features)
    exog_columns = _stable_dynamic_feature_columns(model_frame)
    model_frame = model_frame.copy()
    model_frame["trend_index"] = np.linspace(0.0, 1.0, len(model_frame))
    exog_columns = exog_columns + ["trend_index"]
    train_design = sm.add_constant(model_frame[exog_columns].fillna(0.0).astype(float), has_constant="add")

    future_dates = build_future_dates(train_frame["date"].iloc[-1], horizon)
    future_known = build_known_future_features(future_dates, config, external_features)
    future_known["trend_index"] = np.linspace(1.0, 1.0 + horizon / max(len(model_frame), 1), horizon)
    future_design = sm.add_constant(future_known[exog_columns].fillna(0.0).astype(float), has_constant="add")
    future_design = future_design.reindex(columns=train_design.columns, fill_value=0.0)

    try:
        fitted = sm.OLS(train_frame["target"].astype(float), train_design).fit()
        predictions = fitted.predict(future_design)
    except Exception:
        return _seasonal_fallback_prediction(train_frame, horizon, "fourier_dynamic_regression_fallback")

    return _finalize_predictions(
        train_frame,
        future_dates,
        predictions,
        "fourier_dynamic_regression",
        "fourier_dynamic_regression_fallback",
    )


def unobserved_components_forecast(
    train_frame: pd.DataFrame,
    horizon: int,
    config: SystemLevelConfig,
    external_features: pd.DataFrame,
) -> pd.DataFrame:
    model_frame = build_system_level_features(train_frame.copy(), config, external_features)
    exog_columns = _stable_dynamic_feature_columns(model_frame)
    exog_train = model_frame[exog_columns].fillna(0.0).astype(float)
    future_dates = build_future_dates(train_frame["date"].iloc[-1], horizon)
    exog_future = build_known_future_features(future_dates, config, external_features)[exog_columns].fillna(0.0).astype(float)

    freq_terms = [{"period": 7, "harmonics": 3}]
    if len(train_frame) >= 365:
        freq_terms.append({"period": 30, "harmonics": 2})

    try:
        fitted = _fit_statsmodels_safely(
            lambda: UnobservedComponents(
                train_frame["target"].astype(float),
                level="local linear trend",
                seasonal=7,
                freq_seasonal=freq_terms,
                exog=exog_train,
            ).fit(disp=False)
        )
        predictions = fitted.forecast(steps=horizon, exog=exog_future)
    except Exception:
        return _seasonal_fallback_prediction(train_frame, horizon, "unobserved_components_fallback")

    return _finalize_predictions(
        train_frame,
        future_dates,
        predictions,
        "unobserved_components",
        "unobserved_components_fallback",
    )


def tree_ml_forecast(
    train_frame: pd.DataFrame,
    horizon: int,
    config: SystemLevelConfig,
    external_features: pd.DataFrame,
) -> pd.DataFrame:
    feature_frame = build_system_level_features(train_frame.copy(), config, external_features)
    feature_columns = ml_feature_columns(feature_frame)
    training_rows = feature_frame.dropna(subset=feature_columns).copy()
    training_rows = training_rows.dropna(subset=["target"])
    future_dates = build_future_dates(train_frame["date"].iloc[-1], horizon)

    if training_rows.empty:
        return _fallback_prediction(train_frame, horizon, "tree_boosting_fallback")

    if lgb is not None:
        estimator = lgb.LGBMRegressor(
            objective="regression",
            learning_rate=0.05,
            n_estimators=300,
            max_depth=6,
            random_state=42,
            verbosity=-1,
        )
        model_name = "lightgbm"
    elif xgb is not None:
        estimator = xgb.XGBRegressor(
            objective="reg:squarederror",
            learning_rate=0.05,
            n_estimators=300,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        model_name = "xgboost"
    else:
        estimator = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_depth=6,
            max_iter=300,
            random_state=42,
        )
        model_name = "tree_boosting_fallback"

    estimator.fit(training_rows[feature_columns].fillna(0.0), training_rows["target"].astype(float))

    working_target = train_frame[["date", "target", "missing_period_flag", "series_scope"]].copy()
    predictions: list[float] = []
    for future_date in future_dates["date"]:
        candidate = pd.concat(
            [
                working_target,
                pd.DataFrame(
                    {
                        "date": [future_date],
                        "target": [np.nan],
                        "missing_period_flag": [0],
                        "series_scope": ["system_level"],
                    }
                ),
            ],
            ignore_index=True,
        )
        candidate_features = build_system_level_features(candidate, config, external_features)
        row = candidate_features.iloc[[-1]].copy()
        prediction = float(estimator.predict(row[feature_columns].fillna(0.0))[0])
        prediction = max(prediction, 0.0)
        predictions.append(prediction)
        working_target = candidate.copy()
        working_target.loc[working_target.index[-1], "target"] = prediction

    return pd.DataFrame({"date": future_dates["date"], "prediction": predictions, "model_name": model_name})


MODEL_REGISTRY: dict[str, Callable[[pd.DataFrame, int, SystemLevelConfig, pd.DataFrame], pd.DataFrame]] = {
    "naive": naive_forecast,
    "seasonal_naive_7": lambda train_frame, horizon, config, external_features: seasonal_naive_forecast(
        train_frame, horizon, 7, "seasonal_naive_7"
    ),
    "seasonal_naive_30": lambda train_frame, horizon, config, external_features: seasonal_naive_forecast(
        train_frame, horizon, 30, "seasonal_naive_30"
    ),
    "ets": ets_forecast,
    "sarimax_dynamic": sarimax_dynamic_forecast,
    "fourier_dynamic_regression": fourier_dynamic_regression_forecast,
    "unobserved_components": unobserved_components_forecast,
    "tree_boosting": tree_ml_forecast,
}
