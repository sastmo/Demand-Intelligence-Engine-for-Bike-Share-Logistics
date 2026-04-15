from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import periodogram
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from metro_bike_share_forecasting.system_level.diagnosis.config import DiagnosticConfig


def save_series_plot(prepared: pd.DataFrame, trend_component: pd.Series | None, config: DiagnosticConfig, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(prepared["timestamp"], prepared["value_filled"], color="#1f77b4", linewidth=1.2, label="observed")
    if trend_component is not None:
        ax.plot(prepared["timestamp"], trend_component, color="#188054", linewidth=2.0, label="trend")
    for event in config.events:
        ax.axvline(event.timestamp, linestyle="--", color=event.color, alpha=0.75)
        ax.text(event.timestamp, ax.get_ylim()[1], event.label, rotation=90, va="top", ha="right", fontsize=8, color=event.color)
    ax.set_title("Time Series Overview")
    ax.set_ylabel("Value")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_gap_plot(prepared: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(prepared["timestamp"], prepared["missing_period_flag"], color="#d62728", linewidth=1.0)
    ax.set_title("Missing timestamp indicator")
    ax.set_ylabel("Missing")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_rolling_stats_plot(prepared: pd.DataFrame, window: int, path: Path) -> None:
    rolling_mean = prepared["value_filled"].rolling(window, min_periods=max(2, window // 2)).mean()
    rolling_var = prepared["value_filled"].rolling(window, min_periods=max(2, window // 2)).var()
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(prepared["timestamp"], prepared["value_filled"], color="#1f77b4", linewidth=1.0, label="value")
    axes[0].plot(prepared["timestamp"], rolling_mean, color="#188054", linewidth=1.8, label=f"{window}-period mean")
    axes[0].legend(loc="upper left")
    axes[0].set_title("Rolling mean")
    axes[1].plot(prepared["timestamp"], rolling_var, color="#ff7f0e", linewidth=1.5)
    axes[1].set_title("Rolling variance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_decomposition_plot(decomposition: Any, prepared: pd.DataFrame, path: Path) -> None:
    if decomposition is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Not enough history for seasonal decomposition.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return
    try:
        fig = decomposition.plot()
        fig.set_size_inches(12, 8)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(prepared["timestamp"], prepared["value_filled"])
        ax.set_title("Decomposition output unavailable")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def save_profile_plot(profiles: dict[str, pd.DataFrame], path: Path) -> None:
    frames_to_plot = [name for name in ("weekday_profile", "intraday_profile", "monthly_profile") if name in profiles]
    if not frames_to_plot:
        return
    fig, axes = plt.subplots(len(frames_to_plot), 1, figsize=(12, 4 * len(frames_to_plot)))
    if len(frames_to_plot) == 1:
        axes = [axes]
    for axis, name in zip(axes, frames_to_plot):
        frame = profiles[name]
        if name == "weekday_profile":
            axis.plot(frame["weekday"].astype(str), frame["average_value"], marker="o")
            axis.set_title("Average by weekday")
        elif name == "intraday_profile":
            for weekend_label, subset in frame.groupby("weekend"):
                axis.plot(subset["hour"], subset["average_value"], marker="o", label=str(weekend_label))
            axis.legend(loc="upper left")
            axis.set_title("Average hourly profile")
        else:
            axis.plot(frame["month"].astype(str), frame["average_value"], marker="o")
            axis.set_title("Average by month")
        axis.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_acf_pacf(values: pd.Series, max_lags: int, acf_path: Path, pacf_path: Path) -> None:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    max_lags = min(max_lags, max(1, len(numeric) // 2 - 1))
    if len(numeric) < 8 or max_lags < 1:
        for path, title in ((acf_path, "Autocorrelation"), (pacf_path, "Partial Autocorrelation")):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, f"Not enough history for {title.lower()}.", ha="center", va="center")
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    plot_acf(numeric, lags=max_lags, ax=ax)
    ax.set_title("Autocorrelation")
    fig.tight_layout()
    fig.savefig(acf_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    plot_pacf(numeric, lags=max_lags, ax=ax, method="ywm")
    ax.set_title("Partial autocorrelation")
    fig.tight_layout()
    fig.savefig(pacf_path, dpi=150)
    plt.close(fig)


def save_periodogram_plot(values: pd.Series, path: Path) -> None:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(10, 4))
    if len(numeric) >= 8:
        frequencies, power = periodogram(numeric, detrend="linear", scaling="spectrum")
        ax.plot(frequencies, power, linewidth=1.2)
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Power")
    else:
        ax.text(0.5, 0.5, "Not enough history for a periodogram.", ha="center", va="center")
        ax.axis("off")
    ax.set_title("Periodogram")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_distribution_plot(values: pd.Series, path: Path) -> None:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if not numeric.empty:
        axes[0].hist(numeric, bins=min(40, max(10, len(numeric) // 8)), color="#188054", alpha=0.8)
        axes[0].set_title("Distribution")
        axes[1].boxplot(numeric, vert=True)
        axes[1].set_title("Boxplot")
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "No numeric values available.", ha="center", va="center")
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_outlier_plot(prepared: pd.DataFrame, outlier_detail: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(prepared["timestamp"], prepared["value_filled"], color="#1f77b4", linewidth=1.1, label="value")
    flagged = outlier_detail["outlier_flag"].fillna(False).to_numpy()
    if flagged.any():
        ax.scatter(prepared.loc[flagged, "timestamp"], prepared.loc[flagged, "value_filled"], color="#d62728", s=18, label="anomaly", zorder=3)
    ax.set_title("Outlier and anomaly candidates")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
