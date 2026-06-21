# Model card — lstm-forecast

## Overview
A custom PyTorch LSTM (with optional attention and quantile heads) for univariate and
multivariate financial time-series forecasting, with conformal prediction intervals,
retrieval-augmented conditioning, and an optional Claude-powered analysis layer.

## Intended use
- Research and engineering on time-series forecasting methods.
- Educational exploration of LSTMs, conformal prediction, RAG, and baseline benchmarking.
- A starting framework for forecasting pipelines on your own series.

## Out of scope / non-goals
- **Not financial advice.** Do not use forecasts to make investment decisions.
- No live brokerage/trading execution, order routing, or money movement.
- No real-time tick streaming.
- **No guarantee of market alpha.** Financial returns are close to a random walk; a model
  that ties or marginally beats the naive baseline is a normal, honest outcome.

## Training data
- Daily OHLCV from yfinance/Stooq (with the `data` extra), user-supplied CSV, or a built-in
  deterministic synthetic generator used for offline runs and tests.
- No data is bundled or redistributed; users fetch their own.

## Evaluation
Every run reports the model **and** baselines (Naive, Drift, Seasonal-Naive, ARIMA, ETS) on
a held-out test set across RMSE, MAE, MAPE, sMAPE, R² and MASE. Interval quality is measured
by empirical coverage vs nominal and mean interval width. "Outperformance" is therefore
*measured per series*, never assumed.

## Uncertainty quantification
- **Static** intervals: split-conformal, finite-sample-corrected, marginal coverage.
- **Dynamic** intervals: per-horizon-step quantiles of a rolling-origin backtest residual
  matrix. Coverage guarantees are *marginal* and assume the calibration data is exchangeable
  with the future — an assumption that can break under regime shifts.

## Limitations & risks
- LSTMs can regress toward the mean on noisy series; delta-mode mitigates but does not
  eliminate this.
- Conformal coverage degrades under distribution shift (market regime changes).
- RAG analogs are shape-based and may retrieve spurious matches in low-signal regimes.
- ARIMA/ETS baselines fall back to naive if statsmodels fails to converge.
- The AI layer can produce plausible-but-wrong narratives; it is explanatory, not a source
  of ground truth. It never sees more than the numeric forecast summary and retrieved docs.

## Ethical considerations
Financial forecasting tools can encourage overconfidence. Outputs always include
uncertainty intervals and an explicit not-financial-advice disclaimer in the API
description, dashboard, insights text, and this card.

## Reproducibility
Global seeding (`LSTM_FORECAST_SEED`, default 20) covers Python/NumPy/torch. Synthetic data
is deterministic. GPU nondeterminism may cause minor variation on CUDA devices.
