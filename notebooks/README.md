# Notebooks

Each forecasting capability has a **runnable script** under
[`examples/`](../examples) — maintained, CI-friendly, and offline by default:

| Capability | Script |
| --- | --- |
| Univariate forecasting | `examples/01_univariate.py` |
| Multivariate forecasting | `examples/02_multivariate.py` |
| Probabilistic (conformal) | `examples/03_probabilistic.py` |
| Dynamic probabilistic (backtest) | `examples/04_dynamic_probabilistic.py` |
| Transfer learning | `examples/05_transfer_learning.py` |
| RAG + LLM AI + tuning | `examples/06_rag_and_ai.py` |

`quickstart.ipynb` in this folder is an interactive end-to-end notebook. To turn any
example script into a notebook, install [jupytext](https://jupytext.readthedocs.io) and run
`jupytext --to notebook examples/01_univariate.py`.

All examples run offline (synthetic data fallback) — no API key or network required.
