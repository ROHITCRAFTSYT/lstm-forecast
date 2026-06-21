"""Retrieval-augmented forecasting: find analog historical windows to condition the model.

This is the genuinely novel piece relative to a vanilla LSTM. We index z-normalised
historical windows and, at each timestep, retrieve the most shape-similar past windows and
summarise *what historically followed them* into extra feature channels — a non-parametric
memory of recurring market regimes.
"""

from __future__ import annotations

from lstm_forecast.rag.embedder import embed_windows, znorm_window
from lstm_forecast.rag.retriever import AnalogRetriever, build_analog_retriever
from lstm_forecast.rag.store import AnalogStore

__all__ = [
    "AnalogRetriever",
    "AnalogStore",
    "build_analog_retriever",
    "embed_windows",
    "znorm_window",
]
