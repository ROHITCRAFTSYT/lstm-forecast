"""Bonus — Retrieval-augmented forecasting + Claude AI (beyond the article).

Builds an analog-window index from the training data, conditions the LSTM on retrieved
regimes, then generates a natural-language insight (Claude if ANTHROPIC_API_KEY is set,
otherwise a deterministic template).

    python examples/06_rag_and_ai.py AAPL
"""

from __future__ import annotations

import sys

import numpy as np

from lstm_forecast import Forecaster, Pipeline
from lstm_forecast.ai import generate_insights, suggest_tuning
from lstm_forecast.data import load_prices
from lstm_forecast.rag import build_analog_retriever
from lstm_forecast.transforms import default_finance_transformer


def main(ticker: str = "AAPL") -> None:
    df = load_prices(ticker, allow_synthetic_fallback=True)
    f = Forecaster(y=df["close"], current_dates=df.index, future_dates=21, test_length=42)

    transformer, reverter = default_finance_transformer(seasonal_period=5)
    # Build the analog index from the transformed *training* portion only (leakage-safe).
    split = f.y.size - f.test_length
    transformer.fit(f.y[:split], np.arange(split))
    ref = transformer.transform(f.y[:split], np.arange(split))
    f.attach_retriever(build_analog_retriever(ref, window_len=21, k=8))

    pipe = Pipeline(transformer=transformer, reverter=reverter)
    result = pipe.fit_predict(f, lags=21, epochs=60)

    print(f"\n=== {ticker} — RAG-conditioned benchmark ===")
    print(result.metrics_frame().round(4).to_string())

    print("\n=== LLM-assisted tuning suggestion (Claude or default grid) ===")
    sug = suggest_tuning(f.y)
    for c in sug.candidates:
        print(f"  lags={c.lags} hidden={c.hidden_size} layers={c.num_layers} "
              f"dropout={c.dropout} — {c.rationale}")

    print("\n=== AI insights ===")
    print(generate_insights(result, label=ticker))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "AAPL")
