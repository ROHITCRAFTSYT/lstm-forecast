"""Capability 4 — Dynamic probabilistic forecasting (article §4, upgraded).

A rolling-origin backtest builds a residual matrix of shape (n_windows, horizon); a per-step
quantile of the absolute residuals yields **horizon-aware** intervals that widen with the
forecast step. Backtesting refits the model at each cutoff, so this is the slow path.

    python examples/04_dynamic_probabilistic.py AAPL
"""

from __future__ import annotations

import sys

import numpy as np

from lstm_forecast import Forecaster, Pipeline
from lstm_forecast.data import load_prices
from lstm_forecast.transforms import default_finance_transformer


def main(ticker: str = "AAPL") -> None:
    df = load_prices(ticker, allow_synthetic_fallback=True)
    f = Forecaster(y=df["close"], current_dates=df.index, future_dates=15, test_length=30)
    transformer, reverter = default_finance_transformer(seasonal_period=5)
    pipe = Pipeline(transformer=transformer, reverter=reverter)

    result = pipe.fit_predict(
        f, lags=21, epochs=40, alpha=0.1, run_backtest=True, backtest_windows=10
    )

    bt = result.backtest_result
    print(f"\n=== {ticker} — backtest residual matrix ===")
    print(f"shape (n_windows, horizon) = {bt.residual_matrix.shape}")
    print("per-step RMSE across windows:", np.round(bt.step_rmse(), 4))
    print("\nDynamic intervals (note how width grows with the horizon):")
    for k, (d, p, lo, hi) in enumerate(
        zip(result.future_dates, result.point, result.lower, result.upper, strict=False)
    ):
        print(f"  step {k + 1:2d} {str(d)[:10]}  {p:10.3f}  width={hi - lo:8.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "AAPL")
