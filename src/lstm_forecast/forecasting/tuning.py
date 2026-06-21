"""Cross-validated hyperparameter tuning — closes the "LLM proposes, data decides" loop.

The AI tuner (:mod:`lstm_forecast.ai.tuner`) suggests a candidate grid; this module
*evaluates* that grid with walk-forward cross-validation and picks the winner. Tuning is
therefore data-driven: Claude only narrows the search space.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from lstm_forecast.ai.tuner import TuningSuggestion
    from lstm_forecast.forecasting.forecaster import ModelSpec


def specs_from_suggestion(
    suggestion: TuningSuggestion,
    *,
    base: ModelSpec | None = None,
) -> list[ModelSpec]:
    """Convert an AI :class:`TuningSuggestion` into concrete :class:`ModelSpec` candidates."""
    from lstm_forecast.forecasting.forecaster import ModelSpec

    base = base or ModelSpec()
    specs: list[ModelSpec] = []
    for c in suggestion.candidates:
        specs.append(
            replace(
                base,
                lags=c.lags,
                hidden_size=c.hidden_size,
                num_layers=c.num_layers,
                dropout=c.dropout,
                use_attention=c.use_attention,
            )
        )
    return specs or [base]


def walk_forward_cv(
    y: np.ndarray,
    *,
    spec: ModelSpec,
    exog: np.ndarray | None = None,
    transformer_factory=None,
    retriever=None,
    k: int = 3,
    val_length: int | None = None,
    cv_epochs: int = 40,
    seed: int = 20,
    device: str = "auto",
) -> float:
    """Mean RMSE of ``spec`` across ``k`` rolling-origin validation folds.

    Each fold trains on an expanding prefix and validates on the next ``val_length`` steps.
    Epochs are capped at ``cv_epochs`` to keep the search affordable. Returns ``inf`` if no
    fold is feasible (series too short for the spec).
    """
    from lstm_forecast.forecasting.forecaster import Forecaster

    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    val = val_length or max(spec.lags, n // 10)
    cv_spec = replace(spec, epochs=min(spec.epochs, cv_epochs))

    scores: list[float] = []
    for i in range(k, 0, -1):
        end = n - (i - 1) * val
        if end > n:
            end = n
        if end < val + cv_spec.lags + 5:
            continue
        ex = exog[:end] if exog is not None else None
        f = Forecaster(
            y=y[:end],
            future_dates=val,
            test_length=val,
            exog=pd.DataFrame(ex) if ex is not None else None,
            seed=seed,
            device=device,
        )
        if transformer_factory is not None:
            tr, rv = transformer_factory()
            f.attach_transformer(tr, rv)
        if retriever is not None:
            f.attach_retriever(retriever)
        try:
            res = f.fit_predict(cv_spec, benchmark=False)
        except ValueError:
            continue
        rmse = res.metrics.get(f.name, {}).get("rmse")
        if rmse is not None and np.isfinite(rmse):
            scores.append(float(rmse))
    return float(np.mean(scores)) if scores else float("inf")


def _transformer_factory_from(transformer):
    """Build a factory that yields fresh (deep-copied, re-fittable) transformer stacks."""
    if transformer is None:
        return None

    def factory():
        tr = copy.deepcopy(transformer)
        return tr, tr.reverter()

    return factory
