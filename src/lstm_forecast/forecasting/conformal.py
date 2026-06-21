"""Split-conformal prediction intervals.

Given residuals on a held-out calibration set, conformal prediction yields intervals with
finite-sample marginal coverage guarantees that hold regardless of the model's internal
assumptions about its inputs or error distribution.
"""

from __future__ import annotations

import numpy as np


def conformal_quantile(residuals: np.ndarray, alpha: float = 0.1) -> float:
    """Return the conformal radius for a symmetric ``(1 - alpha)`` interval.

    Uses the finite-sample-corrected ``(1-alpha)(1 + 1/n)`` empirical quantile of absolute
    calibration residuals.
    """
    res = np.abs(np.asarray(residuals, dtype=float).ravel())
    n = res.size
    if n == 0:
        raise ValueError("Need at least one calibration residual.")
    level = min(1.0, (1 - alpha) * (1 + 1 / n))
    return float(np.quantile(res, level, method="higher"))


def conformal_intervals(
    point: np.ndarray,
    residuals: np.ndarray,
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Static conformal intervals around a point forecast.

    Every horizon step gets the same radius (the calibration-set conformal quantile),
    producing a "static" interval (constant width across the horizon).
    Returns ``(lower, upper)``.
    """
    point = np.asarray(point, dtype=float).ravel()
    radius = conformal_quantile(residuals, alpha=alpha)
    return point - radius, point + radius
