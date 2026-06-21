"""Metrics correctness, conformal coverage, and backtest residual-matrix shape."""

from __future__ import annotations

import numpy as np

from lstm_forecast.evaluation.metrics import coverage, mae, rmse, smape
from lstm_forecast.forecasting.backtest import backtest, dynamic_intervals
from lstm_forecast.forecasting.conformal import conformal_intervals, conformal_quantile


def test_metric_values():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.0, 2.0, 3.0, 5.0])
    assert rmse(y, p) == 0.5
    assert mae(y, p) == 0.25
    assert smape(y, y) == 0.0


def test_conformal_coverage_approximates_nominal():
    rng = np.random.default_rng(0)
    cal = rng.normal(0, 1, size=2000)  # calibration residuals
    radius = conformal_quantile(cal, alpha=0.1)
    test = rng.normal(0, 1, size=5000)
    cov = coverage(test, np.full_like(test, -radius), np.full_like(test, radius))
    assert 0.86 <= cov <= 0.94  # ~0.90 nominal


def test_conformal_intervals_symmetric():
    point = np.array([10.0, 11.0, 12.0])
    res = np.array([-1.0, 0.5, -0.3, 0.8, 1.2])
    lo, hi = conformal_intervals(point, res, alpha=0.1)
    np.testing.assert_allclose(hi - point, point - lo)
    assert (hi > point).all() and (lo < point).all()


def test_backtest_residual_matrix_shape_and_dynamic_intervals():
    # Deterministic forecaster: predict the last training value repeated (naive).
    y = np.cumsum(np.random.default_rng(0).normal(0, 1, size=200)) + 100

    def naive_fit_predict(train):
        return np.full(8, train[-1])

    bt = backtest(naive_fit_predict, y, horizon=8, n_windows=10, step=2)
    assert bt.residual_matrix.shape == (bt.n_windows, 8)
    assert bt.horizon == 8

    point = np.full(8, y[-1])
    lo, hi = dynamic_intervals(point, bt.residual_matrix, alpha=0.1)
    assert lo.shape == (8,) and hi.shape == (8,)
    # Dynamic intervals should be non-degenerate.
    assert (hi - lo > 0).all()
