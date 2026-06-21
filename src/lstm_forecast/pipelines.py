"""Compose transforms + forecasting + reversion into a single call.

A thin, typed wrapper around :class:`Forecaster`: the transform/revert steps are handled by
attaching a fitted-on-train transformer to the forecaster (which then applies and inverts it
at the right positions internally), so a whole run is one ``fit_predict`` call.
"""

from __future__ import annotations

from lstm_forecast.forecasting.forecaster import Forecaster, ForecastResult, ModelSpec
from lstm_forecast.transforms.pipeline import Reverter, Transformer


class Pipeline:
    """Wire a transformer (+reverter) and optional RAG retriever into a Forecaster run."""

    def __init__(
        self,
        *,
        transformer: Transformer | None = None,
        reverter: Reverter | None = None,
        retriever: object | None = None,
    ) -> None:
        self.transformer = transformer
        self.reverter = reverter
        self.retriever = retriever

    def fit_predict(
        self,
        f: Forecaster,
        *,
        spec: ModelSpec | None = None,
        alpha: float = 0.1,
        run_backtest: bool = False,
        backtest_windows: int = 10,
        benchmark: bool = True,
        **spec_overrides: object,
    ) -> ForecastResult:
        """Attach pipeline components to ``f`` and run the forecast.

        Convenience: pass model hyperparameters either as a :class:`ModelSpec` via ``spec``
        or as keyword overrides (e.g. ``lags=21, hidden_size=64``).
        """
        if self.transformer is not None:
            f.attach_transformer(self.transformer, self.reverter)
        if self.retriever is not None:
            f.attach_retriever(self.retriever)  # type: ignore[arg-type]

        effective_spec = spec or ModelSpec()
        for key, value in spec_overrides.items():
            if not hasattr(effective_spec, key):
                raise TypeError(f"Unknown ModelSpec field: {key!r}")
            setattr(effective_spec, key, value)

        return f.fit_predict(
            effective_spec,
            alpha=alpha,
            run_backtest=run_backtest,
            backtest_windows=backtest_windows,
            benchmark=benchmark,
        )
