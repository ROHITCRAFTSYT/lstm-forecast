"""Output heads and probabilistic losses for the forecasting model."""

from __future__ import annotations

import torch
import torch.nn as nn


class PointHead(nn.Module):
    """Maps the encoder summary vector to a single value per horizon step."""

    def __init__(self, in_features: int, horizon: int) -> None:
        super().__init__()
        self.horizon = horizon
        self.proj = nn.Linear(in_features, horizon)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # (batch, horizon, 1)
        return self.proj(h).unsqueeze(-1)


class QuantileHead(nn.Module):
    """Maps the encoder summary to ``len(quantiles)`` values per horizon step."""

    def __init__(self, in_features: int, horizon: int, n_quantiles: int) -> None:
        super().__init__()
        self.horizon = horizon
        self.n_quantiles = n_quantiles
        self.proj = nn.Linear(in_features, horizon * n_quantiles)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out = self.proj(h)  # (batch, horizon * n_quantiles)
        return out.view(out.shape[0], self.horizon, self.n_quantiles)


def pinball_loss(
    preds: torch.Tensor,
    target: torch.Tensor,
    quantiles: torch.Tensor,
) -> torch.Tensor:
    """Quantile (pinball) loss averaged over batch, horizon and quantiles.

    Parameters
    ----------
    preds:
        ``(batch, horizon, n_quantiles)`` predicted quantiles.
    target:
        ``(batch, horizon)`` observed values.
    quantiles:
        1-D tensor of quantile levels in ``(0, 1)``.
    """
    target = target.unsqueeze(-1)  # (batch, horizon, 1)
    errors = target - preds  # (batch, horizon, n_quantiles)
    q = quantiles.view(1, 1, -1)
    loss = torch.maximum(q * errors, (q - 1) * errors)
    return loss.mean()
