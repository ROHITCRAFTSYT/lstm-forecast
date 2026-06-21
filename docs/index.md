# lstm-forecast

Production-grade **LSTM time-series forecasting for finance**, enhanced with
**retrieval-augmented forecasting (RAG)**, **Claude-powered insights**, and **rigorous
probabilistic intervals**.

It reimagines the classic scalecast "LSTM for Time Series" walkthrough as a custom PyTorch
system you can ship: a clean library, a REST API, and a dashboard — Dockerized, tested,
and documented.

!!! warning "Not financial advice"
    This is a research/engineering framework. Forecasts are uncertain and markets are not
    guaranteed to be predictable. See the [model card](model_card.md).

## The five capabilities (all upgraded)

| Capability | Upgrade |
| --- | --- |
| Univariate | PyTorch LSTM **+ attention**, reversible transforms |
| Multivariate | engineered finance features as exogenous inputs |
| Probabilistic | **split-conformal** intervals with coverage guarantees |
| Dynamic probabilistic | **backtest residual matrix** → horizon-aware intervals |
| Transfer learning | `transfer_predict` — reuse a fitted model, no retraining |

## Beyond the article

- 🔎 **RAG** — index analog historical windows and condition the model on recurring regimes.
- 🤖 **Claude** — NL insights, a grounded chat assistant, and LLM-assisted tuning.
- 📊 **Honest benchmarking** — every run is scored against Naive / Drift / Seasonal-Naive /
  ARIMA / ETS on a held-out test set.

Start with the [Quickstart](quickstart.md), then read the [Architecture](architecture.md).
