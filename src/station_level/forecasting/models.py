from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover - optional backend may fail during native library load
    lgb = None

try:
    import xgboost as xgb
except Exception:  # pragma: no cover - optional backend may fail during native library load
    xgb = None

import torch
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

from system_level.common.validation import time_ordered_validation_split
from station_level.forecasting.config import StationLevelForecastConfig
from station_level.forecasting.features import (
    BASE_STATION_FEATURE_COLUMNS,
    build_future_station_rows,
    build_station_feature_frame,
    history_lookup,
    station_start_dates,
    training_rows,
)


CAT_FEATURES = ("station_id", "station_category", "cluster_label")
DEEPAR_SCALE_COLUMNS = (
    "target",
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_28",
)
MODEL_DIAGNOSTIC_COLUMNS = [
    "implementation",
    "tuned",
    "selected_params",
]


def _clip(values: np.ndarray | pd.Series) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=float), 0.0)


def _z_value(level: float) -> float:
    if level == 0.80:
        return 1.2815515655446004
    return 1.959963984540054


def _active_feature_columns(frame: pd.DataFrame, config: StationLevelForecastConfig) -> list[str]:
    columns = BASE_STATION_FEATURE_COLUMNS.copy()
    if config.include_category_feature and "station_category" in frame.columns:
        columns.append("station_category")
    if config.include_cluster_feature and "cluster_label" in frame.columns:
        columns.append("cluster_label")
    return [column for column in columns if column in frame.columns]


def _encode_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
    metadata: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    encoded = frame[feature_columns].copy()
    metadata = {} if metadata is None else dict(metadata)
    categorical_maps = dict(metadata.get("categorical_maps", {}))

    for column in [name for name in CAT_FEATURES if name in encoded.columns]:
        if column not in categorical_maps:
            values = pd.Index(encoded[column].fillna("unknown").astype(str).unique().tolist())
            categorical_maps[column] = {value: index for index, value in enumerate(sorted(values), start=1)}
        encoded[column] = encoded[column].fillna("unknown").astype(str).map(categorical_maps[column]).fillna(0).astype(int)

    for column in encoded.columns:
        encoded[column] = pd.to_numeric(encoded[column], errors="coerce")
    encoded = encoded.fillna(0.0)
    metadata["categorical_maps"] = categorical_maps
    metadata["columns"] = encoded.columns.tolist()
    return encoded, metadata


def _validation_split(training_data: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    return time_ordered_validation_split(training_data, "date", horizon)


def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def _tree_candidate_params(model_key: str) -> list[dict[str, Any]]:
    if model_key == "xgboost":
        return [
            {"learning_rate": 0.05, "max_depth": 4, "n_estimators": 250},
            {"learning_rate": 0.03, "max_depth": 6, "n_estimators": 350},
        ]
    return [
        {"learning_rate": 0.05, "max_depth": 6, "n_estimators": 250},
        {"learning_rate": 0.03, "max_depth": 8, "n_estimators": 350},
    ]


def _build_tree_estimator(model_key: str, objective: str, quantile: float | None, params: dict[str, Any], random_state: int):
    learning_rate = float(params.get("learning_rate", 0.05))
    max_depth = int(params.get("max_depth", 6))
    n_estimators = int(params.get("n_estimators", 250))
    if model_key == "lgbm" and lgb is not None:
        if objective == "point":
            return lgb.LGBMRegressor(
                objective="regression",
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                verbosity=-1,
            ), "lightgbm"
        return lgb.LGBMRegressor(
            objective="quantile",
            alpha=float(quantile),
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            verbosity=-1,
        ), "lightgbm"
    if model_key == "xgboost" and xgb is not None:
        return xgb.XGBRegressor(
            objective="reg:squarederror",
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
        ), "xgboost"
    if model_key == "lgbm":
        loss = "squared_error" if objective == "point" else "quantile"
        kwargs = {"loss": loss, "learning_rate": learning_rate, "max_depth": max_depth, "max_iter": n_estimators, "random_state": random_state}
        if objective != "point":
            kwargs["quantile"] = float(quantile)
        return HistGradientBoostingRegressor(**kwargs), "hist_gradient_boosting"
    loss = "squared_error" if objective == "point" else "quantile"
    kwargs = {"loss": loss, "learning_rate": learning_rate, "max_depth": max_depth, "n_estimators": n_estimators, "random_state": random_state}
    if objective != "point":
        kwargs["alpha"] = float(quantile)
    return GradientBoostingRegressor(**kwargs), "gradient_boosting"


def _station_tuning_strategy(config: StationLevelForecastConfig) -> str:
    if not bool(getattr(config, "tune_enabled", False)):
        return "fixed_defaults"
    return "single_validation_split_small_grid"


def station_model_runtime_report(config: StationLevelForecastConfig) -> pd.DataFrame:
    baselines_enabled = getattr(config, "baselines_enabled", {})
    tree_enabled = getattr(config, "tree_enabled", {})
    deepar_enabled = bool(getattr(config, "deepar_enabled", False))
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
    if tree_enabled.get("lgbm", False):
        rows.append(
            {
                "model_name": "lgbm",
                "family": "tree",
                "implementation": "lightgbm" if lgb is not None else "hist_gradient_boosting",
                "experimental": False,
                "native_backend_available": lgb is not None,
                "tuning_strategy": _station_tuning_strategy(config),
                "note": "Native LightGBM if available; otherwise sklearn fallback.",
            }
        )
    if tree_enabled.get("xgboost", False):
        rows.append(
            {
                "model_name": "xgboost",
                "family": "tree",
                "implementation": "xgboost" if xgb is not None else "gradient_boosting",
                "experimental": False,
                "native_backend_available": xgb is not None,
                "tuning_strategy": _station_tuning_strategy(config),
                "note": "Native XGBoost if available; otherwise sklearn fallback.",
            }
        )
    if deepar_enabled:
        rows.append(
            {
                "model_name": "deepar",
                "family": "deep",
                "implementation": "global_neural_mlp",
                "experimental": True,
                "native_backend_available": True,
                "tuning_strategy": _station_tuning_strategy(config),
                "note": "Experimental global neural benchmark; not a canonical recurrent DeepAR implementation.",
            }
        )
    return pd.DataFrame(rows)


def _select_tree_params(
    training_data: pd.DataFrame,
    feature_columns: list[str],
    config: StationLevelForecastConfig,
    model_key: str,
    tune: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    candidates = _tree_candidate_params(model_key)
    if not tune:
        default = candidates[0]
        return default, pd.DataFrame([{"model_name": model_key, "candidate_rank": 1, "selected": True, **default, "validation_mae": np.nan}])

    train_split, valid_split = _validation_split(training_data, horizon=max(config.forecast_horizons))
    if valid_split.empty:
        default = candidates[0]
        return default, pd.DataFrame([{"model_name": model_key, "candidate_rank": 1, "selected": True, **default, "validation_mae": np.nan}])

    train_X, metadata = _encode_features(train_split, feature_columns)
    valid_X, _ = _encode_features(valid_split, feature_columns, metadata)
    train_y = train_split["target"].astype(float).to_numpy()
    valid_y = valid_split["target"].astype(float).to_numpy()

    rows: list[dict[str, Any]] = []
    best_params = candidates[0]
    best_score = float("inf")
    best_rank = 1
    for rank, params in enumerate(candidates, start=1):
        estimator, implementation = _build_tree_estimator(model_key, "point", None, params, config.random_state)
        estimator.fit(train_X, train_y)
        preds = estimator.predict(valid_X)
        score = _mae(valid_y, preds)
        rows.append(
            {
                "model_name": model_key,
                "candidate_rank": rank,
                "implementation": implementation,
                "learning_rate": params["learning_rate"],
                "max_depth": params["max_depth"],
                "n_estimators": params["n_estimators"],
                "validation_mae": score,
                "selected": False,
            }
        )
        if score < best_score:
            best_score = score
            best_params = params
            best_rank = rank

    tuning = pd.DataFrame(rows)
    tuning.loc[tuning["candidate_rank"] == best_rank, "selected"] = True
    return best_params, tuning


@dataclass
class TreeModelArtifact:
    model_name: str
    implementation: str
    point_estimator: Any
    lower_estimator: Any | None
    upper_estimator: Any | None
    feature_columns: list[str]
    encoder_metadata: dict[str, Any]
    selected_params: dict[str, Any]
    tuned: bool


class DeepARLikeNetwork(torch.nn.Module):
    def __init__(self, n_stations: int, n_numeric: int, embedding_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(max(n_stations + 1, 2), embedding_dim)
        self.network = torch.nn.Sequential(
            torch.nn.Linear(n_numeric + embedding_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
        )
        self.mean_head = torch.nn.Linear(hidden_size, 1)
        self.scale_head = torch.nn.Linear(hidden_size, 1)

    def forward(self, station_codes: torch.Tensor, numeric_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.embedding(station_codes)
        hidden = self.network(torch.cat([numeric_features, embedding], dim=1))
        mean = self.mean_head(hidden).squeeze(-1)
        scale = torch.nn.functional.softplus(self.scale_head(hidden).squeeze(-1)) + 1e-3
        return mean, scale


@dataclass
class DeepARArtifact:
    model_name: str
    network: DeepARLikeNetwork
    feature_columns: list[str]
    encoder_metadata: dict[str, Any]
    numeric_columns: list[str]
    numeric_means: dict[str, float]
    numeric_stds: dict[str, float]
    station_scales: dict[str, float]
    tuned: bool
    selected_params: dict[str, Any]


def fit_tree_model(
    training_data: pd.DataFrame,
    config: StationLevelForecastConfig,
    model_key: str,
    tune: bool,
) -> tuple[TreeModelArtifact, pd.DataFrame]:
    feature_columns = _active_feature_columns(training_data, config)
    train_frame = training_rows(training_data).dropna(subset=feature_columns + ["target"]).copy()
    if train_frame.empty:
        raise ValueError(f"No training rows available for {model_key}.")

    selected_params, tuning_results = _select_tree_params(train_frame, feature_columns, config, model_key, tune)
    train_X, metadata = _encode_features(train_frame, feature_columns)
    train_y = train_frame["target"].astype(float).to_numpy()

    point_estimator, implementation = _build_tree_estimator(model_key, "point", None, selected_params, config.random_state)
    if model_key == "xgboost" and implementation == "xgboost":
        lower_estimator = None
        upper_estimator = None
    else:
        lower_estimator, _ = _build_tree_estimator(model_key, "quantile", 0.10, selected_params, config.random_state)
        upper_estimator, _ = _build_tree_estimator(model_key, "quantile", 0.90, selected_params, config.random_state)
    point_estimator.fit(train_X, train_y)
    if lower_estimator is not None and upper_estimator is not None:
        lower_estimator.fit(train_X, train_y)
        upper_estimator.fit(train_X, train_y)

    return (
        TreeModelArtifact(
            model_name=model_key,
            implementation=implementation,
            point_estimator=point_estimator,
            lower_estimator=lower_estimator,
            upper_estimator=upper_estimator,
            feature_columns=feature_columns,
            encoder_metadata=metadata,
            selected_params=selected_params,
            tuned=tune,
        ),
        tuning_results,
    )


def _split_station_and_numeric(encoded: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    station_column = "station_id"
    numeric_columns = [column for column in encoded.columns if column != station_column]
    station_codes = encoded[station_column].astype(int).to_numpy()
    numeric = encoded[numeric_columns].astype(float)
    return station_codes, numeric.to_numpy(dtype=np.float32), numeric_columns


def _standardize_numeric(frame: pd.DataFrame, means: dict[str, float] | None = None, stds: dict[str, float] | None = None):
    standardized = frame.copy()
    means = {} if means is None else dict(means)
    stds = {} if stds is None else dict(stds)
    for column in standardized.columns:
        if means == {} or column not in means:
            means[column] = float(standardized[column].mean())
            std = float(standardized[column].std(ddof=0))
            stds[column] = std if np.isfinite(std) and std > 0 else 1.0
        standardized[column] = (standardized[column] - means[column]) / stds[column]
    return standardized, means, stds


def _deepar_station_scales(train_frame: pd.DataFrame) -> dict[str, float]:
    grouped = (
        train_frame.loc[train_frame["target"].notna()]
        .groupby("station_id")["target"]
        .apply(lambda series: max(float(series.abs().mean()), 1.0))
    )
    return {str(station_id): float(scale) for station_id, scale in grouped.items()}


def _apply_deepar_station_scaling(
    frame: pd.DataFrame,
    station_scales: dict[str, float],
) -> pd.DataFrame:
    scaled = frame.copy()
    if scaled.empty:
        return scaled
    scale_values = scaled["station_id"].astype(str).map(station_scales).fillna(1.0).astype(float)
    for column in [name for name in DEEPAR_SCALE_COLUMNS if name in scaled.columns]:
        scaled[column] = pd.to_numeric(scaled[column], errors="coerce") / scale_values
    return scaled


def _deepar_candidates() -> list[dict[str, Any]]:
    return [
        {"hidden_size": 32, "learning_rate": 0.001},
        {"hidden_size": 48, "learning_rate": 0.0007},
    ]


def _fit_deepar_candidate(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    feature_columns: list[str],
    config: StationLevelForecastConfig,
    params: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    artifact = fit_deepar_model(train_frame, config, tune=False, override_params=params)[0]
    if valid_frame.empty:
        return float("inf"), params
    predictions = predict_with_deepar(artifact, train_frame, valid_frame["date"].drop_duplicates().sort_values(), sorted(valid_frame["station_id"].astype(str).unique().tolist()), config, None)
    scored = valid_frame[["station_id", "date", "target"]].merge(predictions, on=["station_id", "date"], how="left")
    return _mae(scored["target"].to_numpy(dtype=float), scored["prediction"].to_numpy(dtype=float)), params


def fit_deepar_model(
    training_data: pd.DataFrame,
    config: StationLevelForecastConfig,
    tune: bool,
    override_params: dict[str, Any] | None = None,
) -> tuple[DeepARArtifact, pd.DataFrame]:
    feature_columns = _active_feature_columns(training_data, config)
    train_frame = training_rows(training_data).dropna(subset=feature_columns + ["target"]).copy()
    if train_frame.empty:
        raise ValueError("No training rows available for deepar.")

    if override_params is not None:
        selected = dict(override_params)
        tuning_results = pd.DataFrame([{"model_name": "deepar", "candidate_rank": 1, "selected": True, **selected, "validation_mae": np.nan}])
    elif tune:
        split_train, split_valid = _validation_split(train_frame, horizon=max(config.forecast_horizons))
        rows: list[dict[str, Any]] = []
        best_score = float("inf")
        selected = _deepar_candidates()[0]
        for rank, params in enumerate(_deepar_candidates(), start=1):
            score, _ = _fit_deepar_candidate(split_train if not split_train.empty else train_frame, split_valid, feature_columns, config, params)
            rows.append({"model_name": "deepar", "candidate_rank": rank, "validation_mae": score, "selected": False, **params})
            if score < best_score:
                best_score = score
                selected = params
        tuning_results = pd.DataFrame(rows)
        tuning_results.loc[tuning_results["validation_mae"] == tuning_results["validation_mae"].min(), "selected"] = True
    else:
        selected = {"hidden_size": config.deepar_hidden_size, "learning_rate": config.deepar_learning_rate}
        tuning_results = pd.DataFrame([{"model_name": "deepar", "candidate_rank": 1, "selected": True, "validation_mae": np.nan, **selected}])

    station_scales = _deepar_station_scales(train_frame)
    scaled_train_frame = _apply_deepar_station_scaling(train_frame, station_scales)
    encoded, metadata = _encode_features(scaled_train_frame, feature_columns)
    station_codes, numeric_matrix, numeric_columns = _split_station_and_numeric(encoded)
    numeric_frame = pd.DataFrame(numeric_matrix, columns=numeric_columns)
    numeric_frame, means, stds = _standardize_numeric(numeric_frame)

    torch.manual_seed(config.random_state)
    network = DeepARLikeNetwork(
        n_stations=max(metadata["categorical_maps"]["station_id"].values(), default=0) + 1,
        n_numeric=len(numeric_columns),
        embedding_dim=config.deepar_embedding_dim,
        hidden_size=int(selected["hidden_size"]),
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=float(selected["learning_rate"]))
    target = torch.tensor(scaled_train_frame["target"].astype(float).to_numpy(dtype=np.float32))
    station_tensor = torch.tensor(station_codes, dtype=torch.long)
    numeric_tensor = torch.tensor(numeric_frame.to_numpy(dtype=np.float32), dtype=torch.float32)
    batch_size = min(config.deepar_batch_size, len(train_frame))

    network.train()
    for _ in range(config.deepar_epochs):
        order = torch.randperm(len(train_frame))
        for start in range(0, len(train_frame), batch_size):
            batch_idx = order[start : start + batch_size]
            mean, scale = network(station_tensor[batch_idx], numeric_tensor[batch_idx])
            loss = torch.mean(torch.log(scale) + 0.5 * ((target[batch_idx] - mean) / scale) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return (
        DeepARArtifact(
            model_name="deepar",
            network=network.eval(),
            feature_columns=feature_columns,
            encoder_metadata=metadata,
            numeric_columns=numeric_columns,
            numeric_means=means,
            numeric_stds=stds,
            station_scales=station_scales,
            tuned=tune,
            selected_params=selected,
        ),
        tuning_results,
    )


def predict_with_tree(
    artifact: TreeModelArtifact,
    train_panel: pd.DataFrame,
    forecast_dates: pd.DatetimeIndex,
    station_ids: list[str],
    config: StationLevelForecastConfig,
    slice_lookup: pd.DataFrame | None,
) -> pd.DataFrame:
    history_by_station = history_lookup(train_panel)
    station_start_by_station = station_start_dates(train_panel)
    predictions: list[pd.DataFrame] = []

    for forecast_date in forecast_dates:
        rows = build_future_station_rows(
            pd.Timestamp(forecast_date),
            station_ids,
            history_by_station,
            station_start_by_station,
            config,
            slice_lookup,
        )
        encoded, _ = _encode_features(rows, artifact.feature_columns, artifact.encoder_metadata)
        point = _clip(artifact.point_estimator.predict(encoded))
        batch = rows[["station_id", "date"]].copy()
        batch["prediction"] = point
        if artifact.lower_estimator is not None and artifact.upper_estimator is not None:
            lower = _clip(artifact.lower_estimator.predict(encoded))
            upper = np.maximum(point, artifact.upper_estimator.predict(encoded))
            upper = np.maximum(upper, lower)
            batch["lower_80"] = lower
            batch["upper_80"] = upper
        batch["model_name"] = artifact.model_name
        batch["implementation"] = artifact.implementation
        batch["tuned"] = artifact.tuned
        batch["selected_params"] = str(artifact.selected_params)
        predictions.append(batch)

        for station_id, value in zip(batch["station_id"], point):
            station_history = history_by_station.setdefault(str(station_id), pd.Series(dtype=float))
            station_history.loc[pd.Timestamp(forecast_date)] = float(value)
            history_by_station[str(station_id)] = station_history

    return pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()


def predict_with_deepar(
    artifact: DeepARArtifact,
    train_panel: pd.DataFrame,
    forecast_dates: pd.DatetimeIndex,
    station_ids: list[str],
    config: StationLevelForecastConfig,
    slice_lookup: pd.DataFrame | None,
) -> pd.DataFrame:
    history_by_station = history_lookup(train_panel)
    station_start_by_station = station_start_dates(train_panel)
    predictions: list[pd.DataFrame] = []
    z_80 = _z_value(0.80)

    for forecast_date in forecast_dates:
        rows = build_future_station_rows(
            pd.Timestamp(forecast_date),
            station_ids,
            history_by_station,
            station_start_by_station,
            config,
            slice_lookup,
        )
        scaled_rows = _apply_deepar_station_scaling(rows, artifact.station_scales)
        encoded, _ = _encode_features(scaled_rows, artifact.feature_columns, artifact.encoder_metadata)
        station_codes, numeric_matrix, _ = _split_station_and_numeric(encoded)
        numeric_frame = pd.DataFrame(numeric_matrix, columns=artifact.numeric_columns)
        numeric_frame, _, _ = _standardize_numeric(numeric_frame, artifact.numeric_means, artifact.numeric_stds)
        with torch.no_grad():
            mean, scale = artifact.network(
                torch.tensor(station_codes, dtype=torch.long),
                torch.tensor(numeric_frame.to_numpy(dtype=np.float32), dtype=torch.float32),
            )
        station_scale = rows["station_id"].astype(str).map(artifact.station_scales).fillna(1.0).to_numpy(dtype=np.float32)
        point = _clip(mean.numpy() * station_scale)
        std = scale.numpy() * station_scale
        lower = _clip(point - z_80 * std)
        upper = point + z_80 * std
        batch = rows[["station_id", "date"]].copy()
        batch["prediction"] = point
        batch["lower_80"] = lower
        batch["upper_80"] = upper
        batch["model_name"] = "deepar"
        batch["implementation"] = "global_neural_mlp"
        batch["tuned"] = artifact.tuned
        batch["selected_params"] = str(artifact.selected_params)
        predictions.append(batch)

        for station_id, value in zip(batch["station_id"], point):
            station_history = history_by_station.setdefault(str(station_id), pd.Series(dtype=float))
            station_history.loc[pd.Timestamp(forecast_date)] = float(value)
            history_by_station[str(station_id)] = station_history

    return pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()


def _baseline_history_value(history: pd.Series, lookup_date: pd.Timestamp) -> float | None:
    value = history.get(lookup_date, np.nan)
    return float(value) if pd.notna(value) else None


def predict_naive(
    train_panel: pd.DataFrame,
    forecast_dates: pd.DatetimeIndex,
    station_ids: list[str],
    config: StationLevelForecastConfig,
) -> pd.DataFrame:
    history_by_station = history_lookup(train_panel)
    rows: list[pd.DataFrame] = []
    for forecast_date in forecast_dates:
        batch_rows = []
        for station_id in station_ids:
            history = history_by_station.get(station_id, pd.Series(dtype=float)).dropna()
            value = float(history.iloc[-1]) if not history.empty else 0.0
            batch_rows.append({"station_id": station_id, "date": pd.Timestamp(forecast_date), "prediction": max(value, 0.0)})
            history_by_station.setdefault(station_id, history).loc[pd.Timestamp(forecast_date)] = max(value, 0.0)
        batch = pd.DataFrame(batch_rows)
        batch["model_name"] = "naive"
        batch["implementation"] = "baseline_rule"
        rows.append(batch)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def predict_seasonal_naive_7(
    train_panel: pd.DataFrame,
    forecast_dates: pd.DatetimeIndex,
    station_ids: list[str],
    config: StationLevelForecastConfig,
) -> pd.DataFrame:
    history_by_station = history_lookup(train_panel)
    rows: list[pd.DataFrame] = []
    for forecast_date in forecast_dates:
        batch_rows = []
        for station_id in station_ids:
            history = history_by_station.get(station_id, pd.Series(dtype=float))
            seasonal_value = _baseline_history_value(history, pd.Timestamp(forecast_date) - pd.Timedelta(days=7))
            if seasonal_value is None:
                observed = history.dropna()
                seasonal_value = float(observed.iloc[-1]) if not observed.empty else 0.0
            prediction = max(float(seasonal_value), 0.0)
            batch_rows.append({"station_id": station_id, "date": pd.Timestamp(forecast_date), "prediction": prediction})
            history_by_station.setdefault(station_id, history).loc[pd.Timestamp(forecast_date)] = prediction
        batch = pd.DataFrame(batch_rows)
        batch["model_name"] = "seasonal_naive_7"
        batch["implementation"] = "seasonal_rule"
        rows.append(batch)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def station_model_runtime_notes(config: StationLevelForecastConfig) -> list[str]:
    tree_enabled = getattr(config, "tree_enabled", {})
    deepar_enabled = bool(getattr(config, "deepar_enabled", False))
    notes: list[str] = []
    if tree_enabled.get("lgbm", False) and lgb is None:
        notes.append(
            "Native LightGBM is unavailable. `lgbm` will use the sklearn HistGradientBoosting fallback. "
            "On macOS, install `libomp` and reinstall `lightgbm` if you want the native backend."
        )
    if tree_enabled.get("xgboost", False) and xgb is None:
        notes.append(
            "Native XGBoost is unavailable. `xgboost` will use the sklearn GradientBoosting fallback. "
            "On macOS, install `libomp` and reinstall `xgboost` if you want the native backend."
        )
    if deepar_enabled:
        notes.append(
            "`deepar` currently maps to the local `global_neural_mlp` implementation. It is a lightweight global neural "
            "benchmark, not a canonical recurrent DeepAR, so treat it as experimental."
        )
        notes.append(
            "`deepar_context_length` is currently a config placeholder and is not driving a true sequence encoder. "
            "The current network learns from engineered lag features rather than an autoregressive recurrent context window."
        )
    if not bool(getattr(config, "tune_enabled", False)):
        notes.append(
            "Hyperparameter tuning is disabled in the current station-level config. Current model comparisons use fixed/default settings."
        )
    else:
        notes.append(
            "Station-level tuning is currently lightweight: two candidate parameter sets on one time-ordered validation split. "
            "It is not a nested rolling-origin search."
        )
    return notes
