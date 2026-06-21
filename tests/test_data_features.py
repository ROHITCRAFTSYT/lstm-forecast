"""Tests for data loaders and feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lstm_forecast.data import add_finance_features, fourier_terms, load_synthetic_prices
from lstm_forecast.data.features import calendar_features, log_returns, macd, rsi


def test_synthetic_prices_schema():
    df = load_synthetic_prices(n=100, seed=1)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 100
    assert isinstance(df.index, pd.DatetimeIndex)
    assert (df["high"] >= df["low"]).all()
    assert (df["close"] > 0).all()


def test_synthetic_prices_deterministic():
    a = load_synthetic_prices(n=50, seed=42)
    b = load_synthetic_prices(n=50, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_rsi_bounds(prices):
    r = rsi(prices["close"]).dropna()
    assert (r >= 0).all() and (r <= 100).all()


def test_macd_columns(prices):
    out = macd(prices["close"])
    assert set(out.columns) == {"macd", "macd_signal", "macd_hist"}


def test_log_returns_no_lookahead(prices):
    lr = log_returns(prices["close"])
    # First value must be NaN (no prior price); causal.
    assert np.isnan(lr.iloc[0])


def test_fourier_terms_shape(prices):
    ft = fourier_terms(prices.index, period=5.0, n_harmonics=2)
    assert ft.shape == (len(prices), 4)
    assert (ft.abs() <= 1.0 + 1e-9).all().all()


def test_calendar_features_bounded(prices):
    cf = calendar_features(prices.index)
    assert (cf.abs() <= 1.0 + 1e-9).all().all()


def test_add_finance_features_drops_nan_and_keeps_close(prices):
    feat = add_finance_features(prices, fourier_periods=(5.0,))
    assert "close" in feat.columns
    assert "rsi" in feat.columns and "macd" in feat.columns
    assert not feat.isna().any().any()
    assert len(feat) < len(prices)  # warmup rows dropped
