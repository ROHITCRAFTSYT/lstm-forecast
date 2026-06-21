"""LSTM encoder with optional additive attention and a point or quantile head.

The model encodes a ``(batch, lags, n_features)`` window with a (multi-layer) LSTM, forms
a summary vector — either the last hidden state or an attention-weighted average over all
time steps — and projects it to the forecast horizon via the chosen head.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lstm_forecast.models.heads import PointHead, QuantileHead


class AdditiveAttention(nn.Module):
    """Bahdanau-style additive attention pooling over LSTM outputs."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # outputs: (batch, time, hidden)
        scores = self.score(outputs).squeeze(-1)  # (batch, time)
        weights = torch.softmax(scores, dim=1)  # (batch, time)
        context = torch.bmm(weights.unsqueeze(1), outputs).squeeze(1)  # (batch, hidden)
        return context, weights


class LSTMForecaster(nn.Module):
    """Direct multi-horizon LSTM forecaster.

    Parameters
    ----------
    n_features:
        Number of input channels (1 for univariate, more with exogenous/RAG features).
    horizon:
        Forecast horizon (number of steps predicted at once).
    hidden_size, num_layers, dropout:
        LSTM capacity and regularisation.
    use_attention:
        If True, pool LSTM outputs with additive attention; else use the last hidden state.
    quantiles:
        If provided, use a quantile head producing one value per quantile per step;
        otherwise a point head. Quantiles must be sorted and within ``(0, 1)``.
    """

    def __init__(
        self,
        *,
        n_features: int,
        horizon: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        use_attention: bool = True,
        quantiles: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.horizon = horizon
        self.use_attention = use_attention
        self.quantiles = list(quantiles) if quantiles else None

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head_dropout = nn.Dropout(dropout)
        self.attention = AdditiveAttention(hidden_size) if use_attention else None

        if self.quantiles is not None:
            self.head: nn.Module = QuantileHead(hidden_size, horizon, len(self.quantiles))
        else:
            self.head = PointHead(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(batch, horizon, n_outputs)`` where n_outputs is 1 or len(quantiles)."""
        outputs, (h_n, _c_n) = self.lstm(x)
        if self.attention is not None:
            summary, _ = self.attention(outputs)
        else:
            summary = h_n[-1]  # last layer's final hidden state
        summary = self.head_dropout(summary)
        return self.head(summary)
