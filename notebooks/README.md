# Notebooks

The five capabilities from the source article each have a **runnable script** under
[`examples/`](../examples) — these are the maintained, CI-friendly equivalents of the
article's notebook sections:

| # | Capability | Script |
| --- | --- | --- |
| 1 | Univariate forecasting | `examples/01_univariate.py` |
| 2 | Multivariate forecasting | `examples/02_multivariate.py` |
| 3 | Probabilistic (conformal) | `examples/03_probabilistic.py` |
| 4 | Dynamic probabilistic (backtest) | `examples/04_dynamic_probabilistic.py` |
| 5 | Transfer learning | `examples/05_transfer_learning.py` |
| + | RAG + Claude AI | `examples/06_rag_and_ai.py` |

`quickstart.ipynb` in this folder is an interactive end-to-end walkthrough. To turn any
example script into a notebook, install [jupytext](https://jupytext.readthedocs.io) and run
`jupytext --to notebook examples/01_univariate.py`.

All examples run offline (synthetic data fallback) — no API key or network required.
