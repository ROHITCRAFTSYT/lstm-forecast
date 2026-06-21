"""Univariate forecasting — a single price series.

Reversible transforms, conformal intervals, and an honest benchmark against simple
baselines. Runs offline (synthetic fallback) if no data provider is present.

    python examples/01_univariate.py AAPL
"""

from __future__ import annotations

import sys

from lstm_forecast import Forecaster, Pipeline
from lstm_forecast.data import load_prices
from lstm_forecast.transforms import default_finance_transformer


def main(ticker: str = "AAPL") -> None:
    df = load_prices(ticker, allow_synthetic_fallback=True)

    f = Forecaster(
        y=df["close"], current_dates=df.index, future_dates=21, test_length=42, name="lstm"
    )
    transformer, reverter = default_finance_transformer(seasonal_period=5)
    pipe = Pipeline(transformer=transformer, reverter=reverter)
    result = pipe.fit_predict(f, lags=21, hidden_size=64, epochs=60, alpha=0.1)

    print(f"\n=== {ticker} — test-set benchmark (RMSE-sorted) ===")
    print(result.metrics_frame().round(4).to_string())
    print("\n=== 21-step forecast (90% interval) ===")
    for d, p, lo, hi in zip(
        result.future_dates, result.point, result.lower, result.upper, strict=False
    ):
        print(f"  {str(d)[:10]}  {p:10.3f}  [{lo:10.3f}, {hi:10.3f}]")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "AAPL")
