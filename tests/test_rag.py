"""RAG embedder, store and retriever tests (NumPy fallback path)."""

from __future__ import annotations

import numpy as np

from lstm_forecast.rag import AnalogStore, build_analog_retriever, embed_windows, znorm_window


def test_znorm_window():
    w = np.array([1.0, 2.0, 3.0])
    z = znorm_window(w)
    assert abs(z.mean()) < 1e-6
    assert abs(z.std() - 1.0) < 1e-6


def test_znorm_constant_window_is_zero():
    np.testing.assert_array_equal(znorm_window(np.array([5.0, 5.0, 5.0])), np.zeros(3))


def test_embed_windows_shapes():
    series = np.arange(50, dtype=float)
    emb, nxt = embed_windows(series, window_len=10)
    assert emb.shape == (40, 10)
    assert nxt.shape == (40,)


def test_store_search_returns_k():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(100, 8)).astype(np.float32)
    payload = rng.normal(size=100).astype(np.float32)
    store = AnalogStore().build(emb, payload)
    idx, dist, _pay = store.search(emb[0], k=5)
    assert idx.shape == (1, 5)
    assert dist[0, 0] <= dist[0, -1]  # sorted ascending
    assert idx[0, 0] == 0  # nearest to itself


def test_retriever_feature_channels_shape_and_causality():
    series = np.cumsum(np.random.default_rng(1).normal(0, 1, size=200)).astype(np.float32)
    retr = build_analog_retriever(series, window_len=15, k=5)
    ch = retr.feature_channels(series)
    assert ch.shape == (series.size, 2)
    # Early positions without a full window are zero-filled.
    assert np.allclose(ch[: 15 - 1], 0.0)


def test_retriever_changes_with_input():
    a = np.cumsum(np.random.default_rng(1).normal(0, 1, size=150)).astype(np.float32)
    retr = build_analog_retriever(a, window_len=12, k=4)
    b = a * 1.5 + 3.0
    cha = retr.feature_channels(a)
    chb = retr.feature_channels(b)
    # Different inputs should generally produce different channel values.
    assert not np.allclose(cha, chb)
