"""Scoring functions for a twin's forecast. Pure numpy, no side effects."""

from __future__ import annotations

import numpy as np


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE as a fraction in [0, 2]. Robust to counts near zero."""
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    denom = np.abs(a) + np.abs(b)
    denom = np.where(denom == 0.0, 1.0, denom)
    return float(np.mean(2.0 * np.abs(a - b) / denom))


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    """Mean absolute scaled error: forecast MAE over the in-sample naive-step MAE."""
    y_train = np.asarray(y_train, dtype=float)
    if y_train.size < 2:
        return float("nan")
    scale = np.mean(np.abs(np.diff(y_train)))
    if scale == 0.0:
        return float("nan")
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / scale)


def log_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = np.log(np.clip(np.asarray(y_true, dtype=float), 1e-9, None))
    b = np.log(np.clip(np.asarray(y_pred, dtype=float), 1e-9, None))
    return float(np.sqrt(np.mean((a - b) ** 2)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    if ss_tot == 0.0:
        return float("nan")
    return float(1.0 - np.sum((a - b) ** 2) / ss_tot)


def aic_from_residuals(residuals: np.ndarray, n_params: int) -> float:
    """Gaussian-likelihood AIC from residuals (here: residuals of log counts)."""
    r = np.asarray(residuals, dtype=float)
    n = r.size
    sse = float(np.sum(r**2))
    if n == 0 or sse <= 0.0:
        return float("nan")
    return n * np.log(sse / n) + 2 * n_params


def implied_doubling_time_h(
    predict_fn, params: dict, *, horizon_h: float = 12.0, points: int = 400
) -> float:
    """Max instantaneous doubling time implied by the model over an early window."""
    t_s = np.linspace(0.0, horizon_h * 3600.0, points)
    y = np.asarray(predict_fn(params, t_s), dtype=float)
    y = np.clip(y, 1e-9, None)
    ln_y = np.log(y)
    dt_h = np.diff(t_s) / 3600.0
    mu = np.diff(ln_y) / dt_h  # per hour
    mu_max = float(np.max(mu)) if mu.size else 0.0
    if mu_max <= 1e-6:
        return float("inf")
    return float(np.log(2.0) / mu_max)


def plausibility(value: float, lo: float, hi: float) -> float:
    """1.0 inside [lo, hi]; smooth Gaussian-ish decay outside. Range (0, 1]."""
    if not np.isfinite(value):
        return 0.0
    if lo <= value <= hi:
        return 1.0
    width = max(hi - lo, 1e-9)
    dist = (lo - value) if value < lo else (value - hi)
    return float(np.exp(-((dist / width) ** 2)))
