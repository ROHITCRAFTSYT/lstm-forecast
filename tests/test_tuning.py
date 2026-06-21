"""Cross-validated tuning and AI-suggestion → spec conversion."""

from __future__ import annotations

import numpy as np

from lstm_forecast import Forecaster
from lstm_forecast.ai.tuner import CandidateConfig, TuningSuggestion
from lstm_forecast.forecasting.forecaster import ModelSpec
from lstm_forecast.forecasting.tuning import specs_from_suggestion, walk_forward_cv


def _structured(n=240, seed=0):
    t = np.arange(n)
    rng = np.random.default_rng(seed)
    return 100 + 0.1 * t + 5 * np.sin(2 * np.pi * t / 20) + rng.normal(0, 0.5, n)


def test_specs_from_suggestion():
    sug = TuningSuggestion(
        candidates=[
            CandidateConfig(lags=10, hidden_size=32, num_layers=1, dropout=0.0),
            CandidateConfig(lags=20, hidden_size=64, num_layers=2, dropout=0.1),
        ]
    )
    specs = specs_from_suggestion(sug)
    assert len(specs) == 2
    assert specs[0].lags == 10 and specs[1].hidden_size == 64


def test_specs_from_empty_suggestion_returns_base():
    specs = specs_from_suggestion(TuningSuggestion(), base=ModelSpec(lags=7))
    assert len(specs) == 1 and specs[0].lags == 7


def test_walk_forward_cv_returns_finite():
    y = _structured()
    score = walk_forward_cv(
        y, spec=ModelSpec(lags=12, hidden_size=16, epochs=20), k=2, val_length=20, cv_epochs=15
    )
    assert np.isfinite(score)
    assert score > 0


def test_walk_forward_cv_inf_when_too_short():
    score = walk_forward_cv(np.arange(30.0), spec=ModelSpec(lags=20, epochs=5), k=3)
    assert score == float("inf")


def test_forecaster_tune_selects_and_sets_spec():
    y = _structured()
    f = Forecaster(y=y, future_dates=10, test_length=20)
    specs = [
        ModelSpec(lags=8, hidden_size=12, epochs=15),
        ModelSpec(lags=16, hidden_size=24, epochs=15),
    ]
    report = f.tune(specs, k=2, val_length=20, cv_epochs=12)
    assert report["best_spec"] is not None
    assert f.spec is report["best_spec"]
    assert len(report["results"]) == 2
    assert all(np.isfinite(r["cv_rmse"]) for r in report["results"])
