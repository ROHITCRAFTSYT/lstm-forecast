"""Rolling-origin backtesting and horizon-aware (dynamic) conformal intervals.

Backtesting refits the forecaster at successively later cutoffs and records the residual at
each horizon step. The stacked residuals form a matrix of shape ``(n_windows, horizon)``.
Taking a per-step quantile of the absolute residuals yields **dynamic** intervals whose
width grows with the forecast step (horizon-aware probabilistic forecasting).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class BacktestResult:
    """Outputs of a rolling-origin backtest."""

    residual_matrix: np.ndarray  # (n_windows, horizon) signed residuals (actual - pred)
    actuals: np.ndarray  # (n_windows, horizon)
    preds: np.ndarray  # (n_windows, horizon)
    cutoffs: list[int]  # training-end positions used for each window

    @property
    def n_windows(self) -> int:
        return self.residual_matrix.shape[0]

    @property
    def horizon(self) -> int:
        return self.residual_matrix.shape[1]

    def step_rmse(self) -> np.ndarray:
        """Per-horizon-step RMSE across windows."""
        return np.sqrt(np.mean(self.residual_matrix**2, axis=0))


def backtest(
    fit_predict_fn: Callable[[np.ndarray], np.ndarray],
    y: np.ndarray,
    *,
    horizon: int,
    n_windows: int = 10,
    step: int = 1,
    min_train: int | None = None,
) -> BacktestResult:
    """Run an expanding-window rolling-origin backtest.

    Parameters
    ----------
    fit_predict_fn:
        Callable mapping a training series (1-D) to a length-``horizon`` point forecast.
        It is invoked once per window — for an LSTM this means a full refit per cutoff.
    y:
        Full observed series.
    horizon:
        Forecast horizon per window.
    n_windows:
        Number of backtest cutoffs (≥ 10 recommended for 90% dynamic intervals).
    step:
        Spacing between consecutive backtest cutoffs.
    min_train:
        Minimum training length for the earliest window. Defaults to half the series.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    if min_train is None:
        min_train = max(horizon * 2, n // 2)

    # Latest cutoff leaves room for one full horizon of actuals.
    last_cutoff = n - horizon
    cutoffs = [last_cutoff - i * step for i in range(n_windows)]
    cutoffs = [c for c in cutoffs if c >= min_train]
    cutoffs = sorted(set(cutoffs))
    if not cutoffs:
        raise ValueError(
            f"No valid backtest windows: series length {n} too short for horizon={horizon}, "
            f"n_windows={n_windows}, step={step}, min_train={min_train}."
        )

    residuals, actuals_all, preds_all = [], [], []
    for cutoff in cutoffs:
        train = y[:cutoff]
        actual = y[cutoff : cutoff + horizon]
        pred = np.asarray(fit_predict_fn(train), dtype=float).ravel()[:horizon]
        if pred.size < horizon:  # forecaster returned short — pad with last value
            pred = np.concatenate([pred, np.full(horizon - pred.size, pred[-1])])
        residuals.append(actual - pred)
        actuals_all.append(actual)
        preds_all.append(pred)

    return BacktestResult(
        residual_matrix=np.vstack(residuals),
        actuals=np.vstack(actuals_all),
        preds=np.vstack(preds_all),
        cutoffs=cutoffs,
    )


def dynamic_intervals(
    point: np.ndarray,
    residual_matrix: np.ndarray,
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-step conformal intervals from a backtest residual matrix.

    The radius at horizon step ``k`` is the finite-sample-corrected ``(1-alpha)`` quantile
    of ``|residual_matrix[:, k]|``, so intervals widen with the horizon. Returns
    ``(lower, upper)`` aligned to ``point``.
    """
    point = np.asarray(point, dtype=float).ravel()
    res = np.abs(np.asarray(residual_matrix, dtype=float))
    n_windows, horizon = res.shape
    if point.size != horizon:
        raise ValueError(f"point length {point.size} != residual horizon {horizon}")
    level = min(1.0, (1 - alpha) * (1 + 1 / n_windows))
    radius = np.quantile(res, level, axis=0, method="higher")
    return point - radius, point + radius
