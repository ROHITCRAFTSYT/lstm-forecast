"""lstm-forecast: production-grade LSTM time-series forecasting with RAG and Claude AI.

The public surface mirrors the ergonomics of the scalecast library referenced in the
project's design doc, but the forecasting core is a custom PyTorch model with attention
and probabilistic (quantile + conformal) heads, plus optional retrieval-augmented
forecasting and Claude-powered insights.

Typical usage::

    from lstm_forecast import Forecaster, Pipeline
    from lstm_forecast.transforms import default_finance_transformer
    from lstm_forecast.data import load_prices

    series = load_prices("AAPL")
    f = Forecaster(y=series["close"], current_dates=series.index, future_dates=21, test_length=42)
    transformer, reverter = default_finance_transformer(seasonal_period=5)
    pipe = Pipeline(transformer=transformer, reverter=reverter)
    f = pipe.fit_predict(f, lags=21)
    print(f.results.metrics_frame())
"""

from __future__ import annotations

from lstm_forecast.config import Settings, get_settings
from lstm_forecast.forecasting.forecaster import Forecaster, ForecastResult
from lstm_forecast.pipelines import Pipeline

__all__ = [
    "ForecastResult",
    "Forecaster",
    "Pipeline",
    "Settings",
    "__version__",
    "get_settings",
]

__version__ = "0.2.0"
