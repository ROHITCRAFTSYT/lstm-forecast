"""Training loop with time-aware validation split, early stopping, and MC-dropout."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lstm_forecast.models.dataset import WindowDataset
from lstm_forecast.models.heads import pinball_loss
from lstm_forecast.models.lstm import LSTMForecaster


@dataclass
class TrainerConfig:
    """Hyperparameters for :class:`Trainer`."""

    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    val_fraction: float = 0.2
    patience: int = 10
    grad_clip: float = 1.0
    device: str = "cpu"
    verbose: bool = False
    quantiles: list[float] | None = None
    history: list[dict[str, float]] = field(default_factory=list)


class Trainer:
    """Trains an :class:`LSTMForecaster` and produces point/quantile/MC-dropout forecasts."""

    def __init__(self, model: LSTMForecaster, config: TrainerConfig) -> None:
        self.model = model.to(config.device)
        self.config = config
        self._quantile_tensor: torch.Tensor | None = None
        if config.quantiles is not None:
            self._quantile_tensor = torch.tensor(
                config.quantiles, dtype=torch.float32, device=config.device
            )

    def _loss(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self._quantile_tensor is not None:
            return pinball_loss(preds, target, self._quantile_tensor)
        # Point head: preds is (batch, horizon, 1)
        return nn.functional.mse_loss(preds.squeeze(-1), target)

    def _split(self, x: np.ndarray, y: np.ndarray) -> tuple[WindowDataset, WindowDataset | None]:
        n = x.shape[0]
        n_val = round(n * self.config.val_fraction)
        if n_val < 1 or n - n_val < 1:
            return WindowDataset(x, y), None
        # Time order preserved: earliest samples train, latest validate.
        return (
            WindowDataset(x[:-n_val], y[:-n_val]),
            WindowDataset(x[-n_val:], y[-n_val:]),
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> Trainer:
        cfg = self.config
        train_ds, val_ds = self._split(x, y)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        val_loader = (
            DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False) if val_ds else None
        )

        optim = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        best_loss = float("inf")
        best_state = copy.deepcopy(self.model.state_dict())
        epochs_no_improve = 0

        for epoch in range(cfg.epochs):
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(cfg.device), yb.to(cfg.device)
                optim.zero_grad()
                loss = self._loss(self.model(xb), yb)
                loss.backward()
                if cfg.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                optim.step()
                train_loss += loss.item() * xb.shape[0]
            train_loss /= max(len(train_ds), 1)

            val_loss = self._evaluate(val_loader) if val_loader else train_loss
            cfg.history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            if cfg.verbose:
                print(f"epoch {epoch:3d}  train={train_loss:.5f}  val={val_loss:.5f}")

            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= cfg.patience:
                    if cfg.verbose:
                        print(f"Early stopping at epoch {epoch} (best val={best_loss:.5f}).")
                    break

        self.model.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(self.config.device), yb.to(self.config.device)
            total += self._loss(self.model(xb), yb).item() * xb.shape[0]
            n += xb.shape[0]
        return total / max(n, 1)

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return raw model output ``(n, horizon, n_outputs)`` for input windows ``x``."""
        self.model.eval()
        xb = torch.as_tensor(x, dtype=torch.float32, device=self.config.device)
        return self.model(xb).cpu().numpy()

    @torch.no_grad()
    def predict_point(self, x: np.ndarray) -> np.ndarray:
        """Return a point forecast ``(n, horizon)``.

        For a quantile model the median quantile is used as the point estimate.
        """
        out = self.predict(x)
        if self._quantile_tensor is not None:
            qs = np.asarray(self.config.quantiles)
            median_idx = int(np.argmin(np.abs(qs - 0.5)))
            return out[:, :, median_idx]
        return out[:, :, 0]

    @torch.no_grad()
    def predict_mc_dropout(self, x: np.ndarray, n_samples: int = 50) -> np.ndarray:
        """Monte-Carlo dropout samples ``(n_samples, n, horizon)`` for epistemic uncertainty."""
        self.model.train()  # keep dropout active
        xb = torch.as_tensor(x, dtype=torch.float32, device=self.config.device)
        samples = []
        for _ in range(n_samples):
            out = self.model(xb).cpu().numpy()
            point = out[:, :, 0] if self._quantile_tensor is None else out[:, :, out.shape[2] // 2]
            samples.append(point)
        self.model.eval()
        return np.stack(samples, axis=0)
