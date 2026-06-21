# lstm-forecast

Production-grade **LSTM time-series forecasting for finance**, with
**retrieval-augmented forecasting (RAG)**, **LLM-powered insights**, and **rigorous
probabilistic intervals**.

A custom PyTorch system you can ship: a clean library, a REST API, and a dashboard —
Dockerized, tested, and documented.

!!! warning "Not financial advice"
    This is a research/engineering framework. Forecasts are uncertain and markets are not
    guaranteed to be predictable. See the [model card](model_card.md).

## Forecasting capabilities

| Capability | What it does |
| --- | --- |
| Univariate | PyTorch LSTM **+ attention**, reversible transforms |
| Multivariate | engineered finance features as exogenous inputs |
| Probabilistic | **split-conformal** intervals with coverage guarantees |
| Dynamic probabilistic | **backtest residual matrix** → horizon-aware intervals |
| Transfer learning | `transfer_predict` — reuse a fitted model, no retraining |

## What sets it apart

- 🔎 **RAG** — index analog historical windows and condition the model on recurring regimes.
- 🤖 **LLM AI** — NL insights, a grounded chat assistant, and LLM-assisted tuning (any provider).
- 📊 **Honest benchmarking** — every run is scored against Naive / Drift / Seasonal-Naive /
  ARIMA / ETS on a held-out test set, with a Diebold–Mariano significance test.

Start with the [Quickstart](quickstart.md), then read the [Architecture](architecture.md).
