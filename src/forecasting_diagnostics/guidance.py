from __future__ import annotations

from typing import Any


def build_model_guidance(summary: dict[str, Any]) -> tuple[list[str], list[str], list[str], dict[str, float]]:
    scores = {
        "ETS / exponential smoothing": 0.0,
        "ARIMA / SARIMA": 0.0,
        "Fourier-based regression or dynamic harmonic regression": 0.0,
        "TBATS / multi-seasonal state space": 0.0,
        "SARIMAX or ML lag-feature models with exogenous signals": 0.0,
        "Probabilistic / count-aware methods": 0.0,
        "Intermittent-demand methods": 0.0,
    }

    trend_strength = summary.get("trend_strength") or 0.0
    seasonal_strength = summary.get("seasonal_strength") or 0.0
    lag1 = summary.get("lag1_autocorrelation") or 0.0
    seasonal_lag = summary.get("seasonal_lag_autocorrelation") or 0.0
    level_shift_count = summary.get("level_shift_count") or 0
    multiple_seasonalities = bool(summary.get("multiple_seasonalities_detected"))
    is_count_like = bool(summary.get("is_count_like"))
    is_intermittent = bool(summary.get("is_intermittent_like"))
    outlier_share = summary.get("outlier_share") or 0.0
    stationarity = summary.get("stationarity_assessment")
    seasonal_naive_mae = summary.get("seasonal_naive_mae")
    naive_mae = summary.get("naive_mae")

    if trend_strength >= 0.45 and seasonal_strength >= 0.35 and not multiple_seasonalities:
        scores["ETS / exponential smoothing"] += 3.0
    if lag1 >= 0.45 or stationarity in {"mixed_signal", "likely_stationary"}:
        scores["ARIMA / SARIMA"] += 2.0
    if seasonal_lag >= 0.35:
        scores["ARIMA / SARIMA"] += 1.5
    if multiple_seasonalities:
        scores["Fourier-based regression or dynamic harmonic regression"] += 3.0
        scores["TBATS / multi-seasonal state space"] += 3.0
    elif seasonal_strength >= 0.35:
        scores["Fourier-based regression or dynamic harmonic regression"] += 2.0
    if level_shift_count >= 1 or stationarity == "likely_nonstationary":
        scores["SARIMAX or ML lag-feature models with exogenous signals"] += 2.0
    if is_count_like:
        scores["Probabilistic / count-aware methods"] += 2.0
    if is_intermittent:
        scores["Intermittent-demand methods"] += 3.0
    if outlier_share >= 0.02:
        scores["Probabilistic / count-aware methods"] += 1.0
        scores["SARIMAX or ML lag-feature models with exogenous signals"] += 1.0
    if seasonal_naive_mae is not None and naive_mae is not None and seasonal_naive_mae < naive_mae:
        scores["ETS / exponential smoothing"] += 1.0
        scores["ARIMA / SARIMA"] += 1.0

    ranked_methods = [name for name, score in sorted(scores.items(), key=lambda item: item[1], reverse=True) if score > 0]

    recommendations: list[str] = []
    risks: list[str] = []
    if multiple_seasonalities:
        recommendations.append("Multiple seasonal cycles appear present, so include harmonic terms or a multi-seasonal model instead of only one seasonal lag.")
    if level_shift_count >= 1:
        recommendations.append("Detected level shifts suggest intervention variables, regime-aware features, or rolling retraining should be part of the modeling strategy.")
        risks.append("Structural breaks can invalidate backtests if folds do not span multiple regimes.")
    if stationarity == "likely_nonstationary":
        recommendations.append("The series looks nonstationary, so differencing, local trend components, or explicit trend features are more defensible than a fixed-level assumption.")
        risks.append("Training on one long pooled history may wash out recent behavior if the level keeps moving.")
    if seasonal_lag >= 0.35:
        recommendations.append("Strong seasonal autocorrelation means a seasonal-naive benchmark is meaningful and lag-based models should include the seasonal lag explicitly.")
    if is_count_like:
        recommendations.append("Because the target behaves like non-negative counts, count-aware probabilistic models can produce more defensible intervals than Gaussian assumptions.")
    if is_intermittent:
        recommendations.append("Zero mass is high enough that intermittent-demand methods or zero-aware probabilistic models should be considered.")
        risks.append("Sparse zeros can make average-based baselines look better than they really are on peaks.")
    if outlier_share >= 0.02:
        recommendations.append("Anomalies are frequent enough that interval calibration and robust residual diagnostics should be reviewed before trusting uncertainty estimates.")
        risks.append("Outliers may destabilize both ARIMA-style residual assumptions and naive interval heuristics.")
    if not recommendations:
        recommendations.append("Start with seasonal baselines, then compare one interpretable statistical model and one lag-feature model on strict rolling validation.")

    return ranked_methods, recommendations, risks, scores


def build_insights(summary: dict[str, Any]) -> list[str]:
    insights: list[str] = []
    trend_strength = summary.get("trend_strength")
    seasonal_strength = summary.get("seasonal_strength")
    lag1 = summary.get("lag1_autocorrelation")
    seasonal_lag = summary.get("seasonal_lag_autocorrelation")
    dominant_periods = summary.get("dominant_periods", [])
    stationarity = summary.get("stationarity_assessment")
    level_shift_count = summary.get("level_shift_count", 0)
    missing_periods = summary.get("missing_periods", 0)

    if trend_strength is not None and trend_strength >= 0.6:
        insights.append("Trend strength is high, so a fixed-level forecasting assumption would be too weak.")
    elif trend_strength is not None and trend_strength >= 0.3:
        insights.append("Trend strength is moderate, which supports rolling retraining or explicit trend components.")
    if seasonal_strength is not None and seasonal_strength >= 0.6:
        insights.append("Seasonality is strong and should be encoded explicitly with seasonal lags, seasonal baselines, or harmonic terms.")
    if lag1 is not None and lag1 >= 0.65:
        insights.append("Lag-1 autocorrelation is very high, so recent memory is a strong forecasting signal.")
    if seasonal_lag is not None and seasonal_lag >= 0.35:
        insights.append("Seasonal autocorrelation is material, making seasonal-naive a real benchmark rather than a weak baseline.")
    if dominant_periods:
        insights.append(f"Frequency-domain peaks suggest repeating cycles near {', '.join(str(period) for period in dominant_periods[:3])} periods.")
    if stationarity == "likely_nonstationary":
        insights.append("Stationarity tests suggest the mean level is not stable, which supports differencing or trend/regime handling.")
    elif stationarity == "mixed_signal":
        insights.append("Stationarity tests are mixed, which is typical when trend and seasonality are both present.")
    if level_shift_count:
        insights.append("Detected level shifts imply regime-aware modeling and time-aware validation matter here.")
    if missing_periods:
        insights.append("Missing timestamps were reconstructed only for diagnostics, so production features should preserve completeness flags.")
    return insights
