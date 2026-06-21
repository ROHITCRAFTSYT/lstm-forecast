"""Embed time-series windows so shape-similar windows are nearby in vector space.

We z-normalise each window (subtract mean, divide by std) and use the normalised values
directly as the embedding. Euclidean distance on z-normalised windows is monotonic in
shape dissimilarity, so nearest neighbours are "the same pattern at a different level/scale".
"""

from __future__ import annotations

import numpy as np


def znorm_window(window: np.ndarray) -> np.ndarray:
    """Z-normalise a 1-D window (zero mean, unit std)."""
    window = np.asarray(window, dtype=np.float32)
    mean = window.mean()
    std = window.std()
    if std == 0:
        return np.zeros_like(window)
    return (window - mean) / std


def embed_windows(series: np.ndarray, window_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Build z-normalised embeddings of every length-``window_len`` window with a successor.

    Returns
    -------
    embeddings:
        ``(n_windows, window_len)`` z-normalised window vectors.
    next_values:
        ``(n_windows,)`` the raw series value immediately following each window — the
        "what happened next" target used to summarise analogs.
    """
    series = np.asarray(series, dtype=np.float32).ravel()
    n = series.size
    n_windows = n - window_len  # need one value after each window
    if n_windows <= 0:
        raise ValueError(f"series length {n} too short for window_len {window_len}")
    emb = np.empty((n_windows, window_len), dtype=np.float32)
    nxt = np.empty(n_windows, dtype=np.float32)
    for i in range(n_windows):
        w = series[i : i + window_len]
        emb[i] = znorm_window(w)
        nxt[i] = series[i + window_len]
    return emb, nxt
