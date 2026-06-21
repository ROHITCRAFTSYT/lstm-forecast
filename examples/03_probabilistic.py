"""Capability 3 — Probabilistic forecasting (article §3, upgraded).

Static split-conformal intervals with a coverage check on the held-out test set. Conformal
calibration gives marginal coverage guarantees regardless of the model's error distribution.

    python examples/03_probabilistic.py AAPL
"""

from __future__ import annotations

import sys

from lstm_forecast import Forecaster, Pipeline
from lstm_forecast.data import load_prices
from lstm_forecast.transforms import default_finance_transformer


def main(ticker: str = "AAPL") -> None:
    df = load_prices(ticker, allow_synthetic_fallback=True)
    f = Forecaster(y=df["close"], current_dates=df.index, future_dates=21, test_length=60)
    transformer, reverter = default_finance_transformer(seasonal_period=5)
    pipe = Pipeline(transformer=transformer, reverter=reverter)
    result = pipe.fit_predict(f, lags=21, epochs=60, alpha=0.1)

    print(f"\n=== {ticker} — 90% conformal interval coverage on the test set ===")
    print(f"coverage   : {result.interval.get('coverage'):.3f}  (nominal 0.90)")
    print(f"mean width : {result.interval.get('mean_width'):.4f}")
    print("\nForecast with static conformal intervals:")
    for d, p, lo, hi in zip(
        result.future_dates, result.point, result.lower, result.upper, strict=False
    ):
        print(f"  {str(d)[:10]}  {p:10.3f}  [{lo:10.3f}, {hi:10.3f}]")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "AAPL")
