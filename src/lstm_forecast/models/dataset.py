"""Sliding-window supervised datasets for direct multi-horizon forecasting.

Given a feature matrix whose column 0 is the (transformed) forecasting target and whose
remaining columns are exogenous/engineered features, :func:`make_windows` produces
``(X, Y)`` where ``X`` has shape ``(n_samples, lags, n_features)`` and ``Y`` has shape
``(n_samples, horizon)`` — i.e. the model predicts the whole horizon at once (direct
multi-step), which avoids recursive error accumulation.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def make_windows(
    features: np.ndarray,
    *,
    lags: int,
    horizon: int,
    target_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build supervised sliding windows for direct multi-horizon training.

    Parameters
    ----------
    features:
        2-D array ``(T, F)``; column ``target_idx`` is the target.
    lags:
        Length of the input window (lookback).
    horizon:
        Number of steps ahead to predict.
    target_idx:
        Index of the target column in ``features``.
    """
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D (T, F); got shape {features.shape}")
    t_len = features.shape[0]
    n_samples = t_len - lags - horizon + 1
    if n_samples <= 0:
        raise ValueError(
            f"Not enough observations ({t_len}) for lags={lags} and horizon={horizon}. "
            f"Need at least lags + horizon = {lags + horizon}."
        )
    target = features[:, target_idx]
    x_list = np.empty((n_samples, lags, features.shape[1]), dtype=np.float32)
    y_list = np.empty((n_samples, horizon), dtype=np.float32)
    for i in range(n_samples):
        x_list[i] = features[i : i + lags]
        y_list[i] = target[i + lags : i + lags + horizon]
    return x_list, y_list


def last_input_window(features: np.ndarray, *, lags: int) -> np.ndarray:
    """Return the final ``lags`` rows as a single inference window ``(1, lags, F)``."""
    features = np.asarray(features, dtype=np.float32)
    if features.shape[0] < lags:
        raise ValueError(f"Need at least {lags} rows, got {features.shape[0]}.")
    return features[-lags:][None, :, :]


class WindowDataset(Dataset):
    """Torch dataset wrapping pre-built ``(X, Y)`` window arrays."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]
