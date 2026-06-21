"""FastAPI service exposing the forecasting library over HTTP."""

from __future__ import annotations

__all__ = ["create_app"]


def create_app():
    from lstm_forecast.api.main import create_app as _create_app

    return _create_app()
