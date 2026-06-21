"""Compose reversible transforms into a ``Transformer`` and its bound ``Reverter``.

The :class:`Transformer` applies an ordered list of :class:`SeriesTransform` ops, fit on
training data only. The :class:`Reverter` shares the *same fitted state* and inverts them
in reverse order — so a forecast produced in transformed space can be returned to the
original scale, including at future positions beyond the training window.
"""

from __future__ import annotations

import numpy as np

from lstm_forecast.transforms.ops import (
    DeseasonTransform,
    DetrendTransform,
    RobustScaleTransform,
    SeriesTransform,
)


class Transformer:
    """An ordered, fittable sequence of reversible series transforms."""

    def __init__(self, transforms: list[SeriesTransform] | None = None) -> None:
        self.transforms: list[SeriesTransform] = list(transforms or [])
        self.fitted_ = False

    def fit(self, y: np.ndarray, t: np.ndarray | None = None) -> Transformer:
        """Fit each transform in sequence on the (progressively transformed) training data."""
        y = np.asarray(y, dtype=float)
        t = np.arange(len(y)) if t is None else np.asarray(t)
        cur = y
        for tr in self.transforms:
            cur = tr.fit_transform(cur, t)
        self.fitted_ = True
        return self

    def transform(self, y: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        t = np.arange(len(y)) if t is None else np.asarray(t)
        cur = y
        for tr in self.transforms:
            cur = tr.transform(cur, t)
        return cur

    def fit_transform(self, y: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        self.fit(y, t)
        return self.transform(y, t)

    def inverse_transform(self, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Invert all transforms in reverse order at positions ``t``."""
        if not self.fitted_:
            raise RuntimeError("Transformer must be fit before inverse_transform.")
        cur = np.asarray(y, dtype=float)
        t = np.asarray(t)
        for tr in reversed(self.transforms):
            cur = tr.inverse_transform(cur, t)
        return cur

    def reverter(self) -> Reverter:
        """Return a :class:`Reverter` bound to this transformer's fitted state."""
        return Reverter(self)


class Reverter:
    """Thin handle that inverts a (fitted) :class:`Transformer`."""

    def __init__(self, transformer: Transformer) -> None:
        self.transformer = transformer

    def inverse_transform(self, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        return self.transformer.inverse_transform(y, t)


def default_finance_transformer(
    *,
    seasonal_period: int | None = 5,
    poly_order: int = 1,
    robust_scale: bool = True,
) -> tuple[Transformer, Reverter]:
    """Build the recommended finance transform stack and its reverter.

    Detrend → (optional) deseason → robust-scale. Robust scaling is preferred for
    financial series because they contain outliers (gaps, spikes). Returns
    ``(transformer, reverter)`` mirroring the article's ``find_optimal_transformation``
    output shape.
    """
    ops: list[SeriesTransform] = [DetrendTransform(poly_order=poly_order)]
    if seasonal_period:
        ops.append(DeseasonTransform(period=seasonal_period))
    if robust_scale:
        ops.append(RobustScaleTransform())
    transformer = Transformer(ops)
    return transformer, transformer.reverter()
