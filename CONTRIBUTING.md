# Contributing to lstm-forecast

Thanks for your interest in improving the project! This guide covers the development setup
and the conventions the codebase follows.

## Development setup

```bash
git clone https://github.com/your-org/lstm-forecast
cd lstm-forecast
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[all]"
pre-commit install
```

## The quality gate

All three must pass before a PR is merged (CI enforces them):

```bash
ruff check .      # lint + import order
mypy src          # static types
pytest            # tests (offline: synthetic data, NumPy RAG fallback, mocked Claude)
```

`ruff format .` formats the code. `pre-commit run --all-files` runs the full hook set.

## Conventions

- **Core library stays framework-agnostic.** No FastAPI/Streamlit imports inside
  `src/lstm_forecast/{data,transforms,models,forecasting,rag,evaluation}`. Serving code
  lives in `api/` and `dashboard/` and only *consumes* the core.
- **No look-ahead.** Features and transforms must be causal; transforms are fit on the
  training split only. New transforms must round-trip (see `tests/test_transforms.py`) and
  invert correctly at future positions.
- **AI is always optional.** Every AI code path must degrade gracefully without
  `ANTHROPIC_API_KEY`. Tests must not require a live key — mock the client.
- **Benchmark honestly.** New model features should be evaluated against the existing
  baselines on a held-out test set; don't hardcode "it wins".
- **Type everything in `src/`.** Public functions get type hints and docstrings.

## Adding a transform / baseline / metric

- A new `SeriesTransform` implements `fit/transform/inverse_transform(y, t)` and gets a
  round-trip + future-position test.
- A new baseline implements the `BaseForecaster` protocol and is added to
  `baseline_registry`.
- A new metric goes in `evaluation/metrics.py` with a correctness test.

## Reporting issues

Open a GitHub issue with a minimal reproduction (the synthetic loader,
`load_synthetic_prices`, is handy for repros that don't need network access).
