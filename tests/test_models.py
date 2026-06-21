"""Model core: windowing, shapes, and an overfit-tiny-batch sanity check."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lstm_forecast.models import LSTMForecaster, Trainer, TrainerConfig, make_windows
from lstm_forecast.models.dataset import last_input_window
from lstm_forecast.models.heads import pinball_loss


def test_make_windows_shapes():
    feats = np.random.default_rng(0).normal(size=(60, 3)).astype(np.float32)
    x, y = make_windows(feats, lags=10, horizon=5)
    assert x.shape == (60 - 10 - 5 + 1, 10, 3)
    assert y.shape == (60 - 10 - 5 + 1, 5)


def test_make_windows_too_short():
    feats = np.zeros((5, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        make_windows(feats, lags=10, horizon=5)


def test_last_input_window():
    feats = np.arange(20, dtype=np.float32).reshape(10, 2)
    w = last_input_window(feats, lags=3)
    assert w.shape == (1, 3, 2)
    np.testing.assert_array_equal(w[0], feats[-3:])


def test_forward_shapes_point_and_quantile():
    x = torch.randn(4, 8, 2)
    point = LSTMForecaster(n_features=2, horizon=3)
    assert point(x).shape == (4, 3, 1)
    quant = LSTMForecaster(n_features=2, horizon=3, quantiles=[0.1, 0.5, 0.9])
    assert quant(x).shape == (4, 3, 3)


def test_pinball_loss_nonnegative():
    preds = torch.randn(5, 3, 3)
    target = torch.randn(5, 3)
    q = torch.tensor([0.1, 0.5, 0.9])
    assert pinball_loss(preds, target, q).item() >= 0


def test_trainer_overfits_tiny_batch():
    """A high-capacity model on a few samples should drive training loss far down."""
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(40, 1)).astype(np.float32)
    x, y = make_windows(feats, lags=6, horizon=2)
    model = LSTMForecaster(n_features=1, horizon=2, hidden_size=64, use_attention=True)
    cfg = TrainerConfig(epochs=200, batch_size=64, lr=5e-3, val_fraction=0.0, patience=200)
    Trainer(model, cfg).fit(x, y)
    final = cfg.history[-1]["train_loss"]
    assert final < 0.5 * cfg.history[0]["train_loss"]


def test_mc_dropout_samples_shape():
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(40, 1)).astype(np.float32)
    x, y = make_windows(feats, lags=6, horizon=2)
    model = LSTMForecaster(n_features=1, horizon=2, dropout=0.3)
    trainer = Trainer(model, TrainerConfig(epochs=5, val_fraction=0.0)).fit(x, y)
    samples = trainer.predict_mc_dropout(x[:3], n_samples=10)
    assert samples.shape == (10, 3, 2)
