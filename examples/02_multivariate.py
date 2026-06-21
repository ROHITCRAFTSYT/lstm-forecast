"""Capability 2 — Multivariate forecasting (article §2, upgraded).

Engineered finance features (RSI/MACD/Bollinger/volatility/Fourier) are added as exogenous
inputs. The univariate vs multivariate comparison shows whether the extra signal helps.

    python examples/02_multivariate.py AAPL
"""

from __future__ import annotations

import sys

from lstm_forecast import Forecaster, Pipeline
from lstm_forecast.data import add_finance_features, load_prices
from lstm_forecast.transforms import default_finance_transformer


def _run(target, dates, exog, label):
    f = Forecaster(y=target, current_dates=dates, future_dates=21, test_length=42, exog=exog)
    transformer, reverter = default_finance_transformer(seasonal_period=5)
    pipe = Pipeline(transformer=transformer, reverter=reverter)
    res = pipe.fit_predict(f, lags=21, epochs=60)
    rmse = res.metrics["lstm"]["rmse"]
    print(f"{label:14s} test RMSE = {rmse:.4f}")
    return res


def main(ticker: str = "AAPL") -> None:
    df = load_prices(ticker, allow_synthetic_fallback=True)
    feat = add_finance_features(df, fourier_periods=(5.0,))

    print(f"\n=== {ticker} — univariate vs multivariate ===")
    _run(feat["close"], feat.index, None, "univariate")
    _run(feat["close"], feat.index, feat.drop(columns=["close"]), "multivariate")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "AAPL")
