"""Transfer-learning endpoint: fit on a source series, forecast a target series."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from lstm_forecast.api import service
from lstm_forecast.api.schemas import (
    ForecastRequest,
    ForecastResponse,
    IntervalPoint,
    TransferRequest,
)

router = APIRouter(tags=["transfer"])


@router.post("/transfer", response_model=ForecastResponse)
def transfer(req: TransferRequest) -> ForecastResponse:
    """Train a model on ``source`` and apply it to ``target`` with no retraining."""
    source_req = ForecastRequest(
        series=req.source,
        horizon=req.horizon,
        test_length=req.test_length,
        lags=req.lags,
        epochs=req.epochs,
    )
    try:
        source_f, _ = service.run_forecast(source_req)
        target_f = service.build_forecaster(
            ForecastRequest(
                series=req.target,
                horizon=req.horizon,
                test_length=req.test_length,
                lags=req.lags,
                epochs=req.epochs,
            )
        )
        result = target_f.transfer_predict(transfer_from=source_f, future_dates=req.horizon)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    points = [
        IntervalPoint(date=str(d)[:10], point=float(p), lower=float(lo), upper=float(hi))
        for d, p, lo, hi in zip(
            result.future_dates, result.point, result.lower, result.upper, strict=False
        )
    ]
    return ForecastResponse(
        name="lstm-transfer",
        horizon=req.horizon,
        alpha=result.alpha,
        forecast=points,
        metrics={},
        interval={},
        best_model=None,
    )
