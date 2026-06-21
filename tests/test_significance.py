"""Diebold-Mariano significance test."""

from __future__ import annotations

import numpy as np
import pytest

from lstm_forecast.evaluation.significance import diebold_mariano


def test_dm_detects_clearly_better_model():
    rng = np.random.default_rng(0)
    # Model B has much smaller errors than model A → B significantly better.
    errors_a = rng.normal(0, 3.0, size=60)
    errors_b = rng.normal(0, 0.5, size=60)
    res = diebold_mariano(errors_a, errors_b)
    assert res.better == "b"
    assert res.significant
    assert res.p_value < 0.05


def test_dm_tie_for_equal_errors():
    rng = np.random.default_rng(1)
    e = rng.normal(0, 1.0, size=60)
    res = diebold_mariano(e, e.copy())
    # Identical errors → zero differential → not significant.
    assert res.better == "tie"
    assert not res.significant


def test_dm_shape_mismatch_raises():
    with pytest.raises(ValueError):
        diebold_mariano(np.zeros(5), np.zeros(6))


def test_dm_too_few_points_is_nan():
    res = diebold_mariano(np.array([0.1, 0.2]), np.array([0.3, 0.4]))
    assert np.isnan(res.statistic)
    assert res.better == "tie"
