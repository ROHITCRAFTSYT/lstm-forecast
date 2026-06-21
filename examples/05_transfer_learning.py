"""Transfer learning — reuse a fitted model on a related series.

Train a model on a source series, then forecast a *different* (related) series with no
retraining via ``transfer_predict`` (the "new data" and "related series" scenarios).

    python examples/05_transfer_learning.py AAPL MSFT
"""

from __future__ import annotations

import sys

from lstm_forecast import Forecaster, Pipeline
from lstm_forecast.data import load_prices
from lstm_forecast.transforms import default_finance_transformer


def main(source: str = "AAPL", target: str = "MSFT") -> None:
    src_df = load_prices(source, allow_synthetic_fallback=True)
    tgt_df = load_prices(target, allow_synthetic_fallback=True)

    # 1. Fit the source model.
    src = Forecaster(y=src_df["close"], current_dates=src_df.index, future_dates=21, test_length=42)
    transformer, reverter = default_finance_transformer(seasonal_period=5)
    Pipeline(transformer=transformer, reverter=reverter).fit_predict(src, lags=21, epochs=60)
    print(f"Source ({source}) trained. Test RMSE = {src.result.metrics['lstm']['rmse']:.4f}")

    # 2. Transfer to the target — no retraining.
    tgt = Forecaster(y=tgt_df["close"], current_dates=tgt_df.index, future_dates=21, test_length=42)
    result = tgt.transfer_predict(transfer_from=src)

    print(f"\n=== Transfer forecast for {target} (model trained on {source}) ===")
    for d, p, lo, hi in zip(
        result.future_dates, result.point, result.lower, result.upper, strict=False
    ):
        print(f"  {str(d)[:10]}  {p:10.3f}  [{lo:10.3f}, {hi:10.3f}]")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0] if args else "AAPL", args[1] if len(args) > 1 else "MSFT")
