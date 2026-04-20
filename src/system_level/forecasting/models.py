from __future__ import annotations

import os
import time
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
except Exception:  # pragma: no cover - optional dependency may fail during native library load
    lgb = None

try:
    import xgboost as xgb
except Exception:  # pragma: no cover - optional dependency may fail during native library load
    xgb = None

from sklearn.ensemble import HistGradientBoostingRegressor

from system_level.forecasting.config import SystemLevelConfig
from system_level.forecasting.features import (
    build_future_dates,
    build_known_future_features,
    build_system_level_features,
    known_future_feature_columns,
    ml_feature_columns,
)


MODEL_DIAGNOSTIC_COLUMNS = [
    "fit_success",
    "fallback_triggered",
    "fallback_reason",
    "exception_type",
    "warning_count",
    "n_train",
    "n_exog",
    "condition_number",
    "model_runtime_seconds",
    "selected_spec",
    "used_exog_columns",
]


def system_model_runtime_report(config: SystemLevelConfig) -> pd.DataFrame:
    baselines_enabled = getattr(config, "baselines_enabled", {})
    classical_enabled = getattr(config, "classical_enabled", {})
    ml_enabled = getattr(config, "ml_enabled", {})
    rows: list[dict[str, object]] = []

    if baselines_enabled.get("naive", False):
        rows.append(
            {
                "model_name": "naive",
                "family": "baseline",
                "implementation": "baseline_rule",
                "experimental": False,
                "native_backend_available": None,
                "tuning_strategy": "none",
                "note": "Last-observation carry-forward benchmark.",
            }
        )
    if baselines_enabled.get("seasonal_naive_7", False):
        rows.append(
            {
                "model_name": "seasonal_naive_7",
                "family": "baseline",
                "implementation": "seasonal_rule",
                "experimental": False,
                "native_backend_available": None,
                "tuning_strategy": "none",
                "note": "Seven-day seasonal naive benchmark.",
            }
        )
    if baselines_enabled.get("seasonal_naive_30", False):
        rows.append(
            {
                "model_name": "seasonal_naive_30",
                "family": "baseline",
                "implementation": "seasonal_rule",
                "experimental": False,
                "native_backend_available": None,
                "tuning_strategy": "none",
                "note": "Thirty-day seasonal naive benchmark.",
            }
        )
    if classical_enabled.get("ets", False):
        rows.append(
            {
                "model_name": "ets",
                "family": "classical",
                "implementation": "holt_winters_exponential_smoothing",
                "experimental": False,
                "native_backend_available": None,
                "tuning_strategy": "heuristic_specification",
                "note": "Single ETS family with guarded fallback; not grid searched.",
            }
        )
    if classical_enabled.get("sarimax_dynamic", False):
        rows.append(
            {
                "model_name": "sarimax_dynamic",
                "family": "classical",
                "implementation": "sarimax_candidate_search",
                "experimental": False,
                "native_backend_available": None,
                "tuning_strategy": "small_candidate_search",
                "note": "Small guarded SARIMAX candidate set with plausibility checks.",
            }
        )
    if classical_enabled.get("fourier_dynamic_regression", False):
        rows.append(
            {
                "model_name": "fourier_dynamic_regression",
                "family": "classical",
                "implementation": "ols_dynamic_regression",
                "experimental": False,
                "native_backend_available": None,
                "tuning_strategy": "fixed_specification",
                "note": "OLS dynamic regression with known-future features; no hyperparameter search.",
            }
        )
    if classical_enabled.get("unobserved_components", False):
        rows.append(
            {
                "model_name": "unobserved_components",
                "family": "classical",
                "implementation": "structural_time_series",
                "experimental": False,
                "native_backend_available": None,
                "tuning_strategy": "heuristic_specification",
                "note": "Structural model with guarded seasonal terms; not a broad spec search.",
            }
        )
    if ml_enabled.get("tree_boosting", False):
        if lgb is not None:
            implementation = "lightgbm"
            native_backend_available = True
        elif xgb is not None:
            implementation = "xgboost"
            native_backend_available = True
        else:
            implementation = "hist_gradient_boosting"
            native_backend_available = False
        rows.append(
            {
                "model_name": "tree_boosting",
                "family": "ml",
                "implementation": implementation,
                "experimental": False,
                "native_backend_available": native_backend_available,
                "tuning_strategy": "fixed_defaults",
                "note": "Recursive tree model with leakage-safe lag features; no hyperparameter tuning yet.",
            }
        )
    return pd.DataFrame(rows)


def system_model_runtime_notes(config: SystemLevelConfig) -> list[str]:
    ml_enabled = getattr(config, "ml_enabled", {})
    classical_enabled = getattr(config, "classical_enabled", {})
    notes: list[str] = []
    if ml_enabled.get("tree_boosting", False):
        if lgb is None and xgb is None:
            notes.append(
                "Native LightGBM/XGBoost backends are unavailable. `tree_boosting` will use the sklearn HistGradientBoosting fallback."
            )
        elif lgb is None:
            notes.append(
                "Native LightGBM is unavailable. `tree_boosting` will prefer XGBoost if importable."
            )
        elif xgb is None:
            notes.append(
                "Native XGBoost is unavailable. `tree_boosting` will prefer LightGBM if importable."
            )
        notes.append(
            "System-level ML currently uses fixed tree parameters with recursive forecasting. It is leakage-safe, but not hyperparameter tuned."
        )
    if any(classical_enabled.values()):
        notes.append(
            "System-level classical models use guarded heuristic specifications and small candidate searches rather than a large automated order search."
        )
    return notes


def _clip_nonnegative(values: np.ndarray | pd.Series) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=float), 0.0)


def _fallback_prediction(train_frame: pd.DataFrame, horizon: int, label: str) -> pd.DataFrame:
    value = float(train_frame["target"].iloc[-1])
    dates = build_future_dates(train_frame["date"].iloc[-1], horizon)
    return pd.DataFrame({"date": dates["date"], "prediction": np.repeat(max(value, 0.0), horizon), "model_name": label})


def _attach_diagnostics(frame: pd.DataFrame, diagnostics: dict[str, object]) -> pd.DataFrame:
    tagged = frame.copy()
    for column in MODEL_DIAGNOSTIC_COLUMNS:
        tagged[column] = diagnostics.get(column)
    return tagged


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


def _drop_constant_columns(frame: pd.DataFrame) -> pd.DataFrame:
    keep_columns = []
    for column in frame.columns:
        series = frame[column]
        if series.nunique(dropna=False) > 1:
            keep_columns.append(column)
    return frame[keep_columns].copy()


def _standardize_continuous_columns(
    train_frame: pd.DataFrame,
    future_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_scaled = train_frame.copy()
    future_scaled = future_frame.copy()
    for column in train_scaled.columns:
        unique_values = train_scaled[column].dropna().unique()
        if len(unique_values) <= 2:
            continue
        std = float(train_scaled[column].std(ddof=0))
        if not np.isfinite(std) or std == 0.0:
            continue
        mean = float(train_scaled[column].mean())
        train_scaled[column] = (train_scaled[column] - mean) / std
        future_scaled[column] = (future_scaled[column] - mean) / std
    return train_scaled, future_scaled


def _drop_highly_collinear_columns(frame: pd.DataFrame, threshold: float = 0.98) -> tuple[pd.DataFrame, list[str]]:
    if frame.shape[1] <= 1:
        return frame.copy(), []
    corr = frame.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    dropped = [column for column in upper.columns if any(upper[column] > threshold)]
    return frame.drop(columns=dropped, errors="ignore").copy(), dropped


def _prepare_sarimax_exog(
    train_frame: pd.DataFrame,
    future_dates: pd.DataFrame,
    config: SystemLevelConfig,
    external_features: pd.DataFrame,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, list[str], float | None]:
    model_frame = build_system_level_features(train_frame.copy(), config, external_features)
    exog_columns = _sarimax_feature_columns(model_frame)
    if not exog_columns:
        return None, None, [], None

    exog_train = model_frame[exog_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float).reset_index(drop=True)
    future_known = build_known_future_features(future_dates, config, external_features)
    exog_future = (
        future_known.reindex(columns=exog_columns, fill_value=0.0)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .astype(float)
        .reset_index(drop=True)
    )

    exog_train = _drop_constant_columns(exog_train)
    exog_future = exog_future.reindex(columns=exog_train.columns, fill_value=0.0)
    if exog_train.empty:
        return None, None, [], None

    exog_train, exog_future = _standardize_continuous_columns(exog_train, exog_future)
    exog_train, dropped_columns = _drop_highly_collinear_columns(exog_train)
    exog_future = exog_future.reindex(columns=exog_train.columns, fill_value=0.0)

    used_columns = exog_train.columns.tolist()
    if not used_columns:
        return None, None, [], None

    try:
        condition_number = float(np.linalg.cond(exog_train.to_numpy(dtype=float)))
    except Exception:
        condition_number = None
    if condition_number is not None and not np.isfinite(condition_number):
        condition_number = None

    return exog_train, exog_future, used_columns, condition_number


def _sarimax_training_frame(train_frame: pd.DataFrame, max_history: int = 1095) -> pd.DataFrame:
    ordered = train_frame.sort_values("date").copy()
    if len(ordered) > max_history:
        ordered = ordered.tail(max_history).copy()
    return ordered.reset_index(drop=True)


def _sarimax_prediction_is_implausible(train_frame: pd.DataFrame, predictions: pd.Series | np.ndarray, horizon: int) -> tuple[bool, str | None]:
    values = np.asarray(predictions, dtype=float)
    if not np.isfinite(values).all():
        return True, "non_finite_forecast_values"
    if (values < 0.0).any():
        return True, "negative_forecast_values"

    upper_bound = _prediction_upper_bound(train_frame)
    if len(values) > 0 and values.max() > upper_bound:
        return True, "forecast_exceeds_plausible_upper_bound"

    recent = train_frame["target"].astype(float).dropna().tail(min(len(train_frame), 30))
    if recent.empty:
        return False, None

    recent_mean = float(recent.mean())
    recent_max = float(recent.max())
    if horizon >= 30 and len(values) > 1:
        end_value = float(values[-1])
        if end_value > max(recent_max * 1.6, recent_mean * 1.75):
            return True, "implausible_upward_drift"
        if float(np.nanmax(values) - np.nanmin(values)) > max(recent_mean * 1.0, recent_max * 0.75):
            return True, "excessive_medium_horizon_volatility"
    return False, None


def _sarimax_spec_label(order: tuple[int, int, int], seasonal_order: tuple[int, int, int, int], trend: str | None) -> str:
    return f"order={order}; seasonal_order={seasonal_order}; trend={trend or 'n'}"


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


def _sarimax_feature_columns(frame: pd.DataFrame) -> list[str]:
    known_columns = known_future_feature_columns(frame)
    external_numeric = [
        column
        for column in known_columns
        if not column.startswith("weekly_")
        and not column.startswith("yearly_")
        and column not in {"day_of_week", "week_of_year", "month", "quarter", "year", "day_of_month", "day_of_year", "season_code"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    selected = [
        column
        for column in known_columns
        if column in {"is_weekend", "is_holiday"} or column.startswith("yearly_")
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
        seasonal_periods = int(config.mase_season_length)
        seasonal = "add" if seasonal_periods > 1 and len(series) >= 2 * seasonal_periods else None
        seasonal_periods = seasonal_periods if seasonal else None
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
    sarimax_train = _sarimax_training_frame(train_frame)
    future_dates = build_future_dates(sarimax_train["date"].iloc[-1], horizon)
    diagnostics: dict[str, object] = {
        "fit_success": False,
        "fallback_triggered": True,
        "fallback_reason": "",
        "exception_type": "",
        "warning_count": 0,
        "n_train": int(len(sarimax_train)),
        "n_exog": 0,
        "condition_number": None,
        "model_runtime_seconds": None,
        "selected_spec": "",
        "used_exog_columns": "",
    }

    try:
        exog_train, exog_future, used_columns, condition_number = _prepare_sarimax_exog(
            sarimax_train, future_dates, config, external_features
        )
    except Exception as exc:
        diagnostics.update(
            {
                "fallback_reason": "exogenous_preparation_failed",
                "exception_type": type(exc).__name__,
            }
        )
        return _attach_diagnostics(_seasonal_fallback_prediction(sarimax_train, horizon, "sarimax_dynamic_fallback"), diagnostics)

    diagnostics["n_exog"] = len(used_columns)
    diagnostics["condition_number"] = condition_number
    diagnostics["used_exog_columns"] = "|".join(used_columns)
    seasonal_period = int(config.mase_season_length)

    if seasonal_period > 1:
        candidate_specs = [
            ((0, 1, 1), (0, 1, 1, seasonal_period), None),
            ((1, 0, 1), (0, 1, 1, seasonal_period), None),
            ((1, 1, 1), (1, 0, 1, seasonal_period), None),
            ((0, 1, 1), (0, 1, 1, seasonal_period), "c"),
        ]
    else:
        candidate_specs = [
            ((0, 1, 1), (0, 0, 0, 0), None),
            ((1, 0, 1), (0, 0, 0, 0), None),
            ((1, 1, 1), (0, 0, 0, 0), None),
            ((0, 1, 1), (0, 0, 0, 0), "c"),
        ]

    best_candidate: dict[str, object] | None = None
    rejection_reasons: list[str] = []

    for order, seasonal_order, trend in candidate_specs:
        spec_label = _sarimax_spec_label(order, seasonal_order, trend)
        started_at = time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                fitted = SARIMAX(
                    sarimax_train["target"].astype(float).reset_index(drop=True),
                    order=order,
                    seasonal_order=seasonal_order,
                    exog=exog_train,
                    trend="n" if trend is None else trend,
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                ).fit(disp=False, maxiter=50)
            runtime_seconds = time.perf_counter() - started_at

            mle_retvals = getattr(fitted, "mle_retvals", None)
            if isinstance(mle_retvals, dict) and not bool(mle_retvals.get("converged", True)):
                rejection_reasons.append(f"{spec_label}:not_converged")
                continue

            params = np.asarray(getattr(fitted, "params", []), dtype=float)
            if params.size > 0 and not np.isfinite(params).all():
                rejection_reasons.append(f"{spec_label}:non_finite_parameters")
                continue

            predictions = fitted.forecast(steps=horizon, exog=exog_future)
            is_implausible, reason = _sarimax_prediction_is_implausible(sarimax_train, predictions, horizon)
            if is_implausible:
                rejection_reasons.append(f"{spec_label}:{reason}")
                continue

            aic = float(getattr(fitted, "aic", np.inf))
            warning_count = len(caught_warnings)
            candidate = {
                "spec_label": spec_label,
                "predictions": np.asarray(predictions, dtype=float),
                "aic": aic,
                "warning_count": warning_count,
                "runtime_seconds": runtime_seconds,
            }
            if best_candidate is None or candidate["aic"] < best_candidate["aic"]:
                best_candidate = candidate
            if warning_count == 0:
                break
        except Exception as exc:
            rejection_reasons.append(f"{spec_label}:{type(exc).__name__}")

    if best_candidate is None:
        diagnostics.update(
            {
                "fallback_reason": ";".join(rejection_reasons[-5:]) if rejection_reasons else "no_valid_sarimax_candidate",
                "exception_type": "CandidateSearchFailed",
            }
        )
        return _attach_diagnostics(_seasonal_fallback_prediction(sarimax_train, horizon, "sarimax_dynamic_fallback"), diagnostics)

    diagnostics.update(
        {
            "fit_success": True,
            "fallback_triggered": False,
            "fallback_reason": "",
            "exception_type": "",
            "warning_count": int(best_candidate["warning_count"]),
            "model_runtime_seconds": float(best_candidate["runtime_seconds"]),
            "selected_spec": str(best_candidate["spec_label"]),
        }
    )
    frame = pd.DataFrame(
        {
            "date": future_dates["date"],
            "prediction": _clip_nonnegative(best_candidate["predictions"]),
            "model_name": "sarimax_dynamic",
        }
    )
    return _attach_diagnostics(frame, diagnostics)


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

    seasonal_period = int(config.mase_season_length)
    freq_terms = []
    if str(config.frequency).strip().lower() in {"d", "day", "daily"} and len(train_frame) >= 180:
        freq_terms.append({"period": 30, "harmonics": 2})

    try:
        uc_kwargs: dict[str, object] = {
            "endog": train_frame["target"].astype(float),
            "level": "local linear trend",
            "exog": exog_train,
        }
        if seasonal_period > 1 and len(train_frame) >= 2 * seasonal_period:
            uc_kwargs["seasonal"] = seasonal_period
        if freq_terms:
            uc_kwargs["freq_seasonal"] = freq_terms

        fitted = _fit_statsmodels_safely(lambda: UnobservedComponents(**uc_kwargs).fit(disp=False))
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
