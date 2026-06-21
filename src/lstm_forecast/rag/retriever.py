"""Turn retrieved analog windows into per-timestep feature channels for the model.

For each position ``t`` the retriever takes the window ending at ``t``, finds the ``k``
most shape-similar historical windows, and summarises what *followed* them into two
channels: the mean next-move and the dispersion across analogs (regime agreement). These
channels are concatenated to the model's input features. Self-matches (distance ~0) are
dropped to avoid trivial leakage of the actual next value.
"""

from __future__ import annotations

import numpy as np

from lstm_forecast.rag.embedder import embed_windows, znorm_window
from lstm_forecast.rag.store import AnalogStore

_SELF_MATCH_EPS = 1e-6


class AnalogRetriever:
    """Builds analog feature channels from a fixed reference (training) series."""

    def __init__(self, store: AnalogStore, window_len: int, k: int = 8) -> None:
        self.store = store
        self.window_len = window_len
        self.k = k

    @property
    def n_channels(self) -> int:
        return 2  # analog-mean delta + analog-std

    def feature_channels(self, series_transformed: np.ndarray) -> np.ndarray | None:
        """Return ``(T, 2)`` analog channels aligned to ``series_transformed``.

        Channel 0: mean of analog "next values" minus the window's last value (expected move).
        Channel 1: std of analog next values (analog disagreement / regime uncertainty).
        Positions without a full lookback window are zero-filled.
        """
        series = np.asarray(series_transformed, dtype=np.float32).ravel()
        t_len = series.size
        out = np.zeros((t_len, 2), dtype=np.float32)
        if self.store.size == 0:
            return out

        for t in range(self.window_len - 1, t_len):
            w = series[t - self.window_len + 1 : t + 1]
            q = znorm_window(w)
            _idx, dists, payloads = self.store.search(q, self.k + 1)
            d = dists[0]
            p = payloads[0]
            # Drop a self-match (near-zero distance) if present.
            keep = d > _SELF_MATCH_EPS
            p = p[keep][: self.k] if keep.any() else p[: self.k]
            if p.size == 0:
                continue
            last_val = float(w[-1])
            out[t, 0] = float(np.mean(p)) - last_val
            out[t, 1] = float(np.std(p))
        return out


def build_analog_retriever(
    reference_series_transformed: np.ndarray,
    *,
    window_len: int = 21,
    k: int = 8,
) -> AnalogRetriever:
    """Build an :class:`AnalogRetriever` from a transformed reference (training) series."""
    embeddings, next_values = embed_windows(reference_series_transformed, window_len)
    store = AnalogStore().build(embeddings, next_values)
    return AnalogRetriever(store, window_len=window_len, k=k)
