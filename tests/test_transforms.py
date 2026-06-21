"""Round-trip and leakage-safety tests for reversible transforms."""

from __future__ import annotations

import numpy as np
import pytest

from lstm_forecast.transforms import (
    DeseasonTransform,
    DetrendTransform,
    DifferenceTransform,
    LogTransform,
    RobustScaleTransform,
    StandardScaleTransform,
    Transformer,
    default_finance_transformer,
)

ALL_SINGLE = [
    LogTransform,
    DetrendTransform,
    StandardScaleTransform,
    RobustScaleTransform,
]


@pytest.fixture
def series():
    t = np.arange(120)
    return 50 + 0.3 * t + 4 * np.sin(2 * np.pi * t / 12) + 1.0


@pytest.mark.parametrize("cls", ALL_SINGLE)
def test_single_transform_round_trip(series, cls):
    t = np.arange(len(series))
    tr = cls()
    z = tr.fit_transform(series, t)
    back = tr.inverse_transform(z, t)
    np.testing.assert_allclose(back, series, rtol=1e-5, atol=1e-5)


def test_deseason_round_trip(series):
    t = np.arange(len(series))
    tr = DeseasonTransform(period=12)
    z = tr.fit_transform(series, t)
    back = tr.inverse_transform(z, t)
    np.testing.assert_allclose(back, series, rtol=1e-5, atol=1e-5)


def test_difference_round_trip_contiguous(series):
    # Fit on a training prefix (sets the anchor), then round-trip an arbitrary block.
    train, train_t = series[:100], np.arange(100)
    block, block_t = series[100:], np.arange(100, len(series))
    tr = DifferenceTransform().fit(train, train_t)
    z = tr.transform(block, block_t)
    recon = tr.inverse_transform(z, block_t)
    np.testing.assert_allclose(recon, block, rtol=1e-6)


def test_transformer_stack_round_trip(series):
    t = np.arange(len(series))
    tr = Transformer([DetrendTransform(1), DeseasonTransform(12), RobustScaleTransform()])
    z = tr.fit_transform(series, t)
    back = tr.inverse_transform(z, t)
    np.testing.assert_allclose(back, series, rtol=1e-5, atol=1e-5)


def test_inverse_at_future_positions(series):
    """Detrend/deseason must invert correctly beyond the training window."""
    n = len(series)
    train, train_t = series[:100], np.arange(100)
    tr = Transformer([DetrendTransform(1), DeseasonTransform(12)])
    tr.fit(train, train_t)
    # Transform full series; invert the future slice at its true positions.
    z_full = tr.transform(series, np.arange(n))
    future_t = np.arange(100, n)
    back_future = tr.inverse_transform(z_full[100:], future_t)
    np.testing.assert_allclose(back_future, series[100:], rtol=1e-5, atol=1e-5)


def test_default_finance_transformer_shapes(series):
    transformer, reverter = default_finance_transformer(seasonal_period=12)
    t = np.arange(len(series))
    z = transformer.fit_transform(series, t)
    assert z.shape == series.shape
    np.testing.assert_allclose(reverter.inverse_transform(z, t), series, rtol=1e-5, atol=1e-5)


def test_log_transform_rejects_nonpositive():
    tr = LogTransform()
    with pytest.raises(ValueError):
        tr.fit(np.array([1.0, -2.0, 3.0]), np.arange(3))
