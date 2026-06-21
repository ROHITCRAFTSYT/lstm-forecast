"""Individual reversible series transforms.

Every transform implements ``fit(y, t)`` / ``transform(y, t)`` / ``inverse_transform(y, t)``
where ``t`` is the integer position of each value relative to the start of the *observed*
series. Passing ``t`` explicitly is what lets trend and seasonal components be evaluated
correctly at future (forecast) positions, so a forecast can be reverted to the original
scale even though it lies beyond the training data.

All transforms are fit on training data only — the caller is responsible for not leaking
test/future values into ``fit``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SeriesTransform(ABC):
    """Base class for reversible 1-D series transforms."""

    fitted_: bool = False

    @abstractmethod
    def fit(self, y: np.ndarray, t: np.ndarray) -> SeriesTransform:
        """Estimate parameters from training values ``y`` at positions ``t``."""

    @abstractmethod
    def transform(self, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Apply the forward transform."""

    @abstractmethod
    def inverse_transform(self, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Invert the transform, valid at arbitrary positions ``t``."""

    def fit_transform(self, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        return self.fit(y, t).transform(y, t)

    def _check_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError(f"{type(self).__name__} must be fit before use.")


class LogTransform(SeriesTransform):
    """Natural-log transform (for strictly positive series like prices)."""

    def __init__(self, offset: float = 0.0) -> None:
        self.offset = offset

    def fit(self, y: np.ndarray, t: np.ndarray) -> LogTransform:
        if np.nanmin(y) + self.offset <= 0:
            raise ValueError("LogTransform requires y + offset > 0 for all values.")
        self.fitted_ = True
        return self

    def transform(self, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return np.log(y + self.offset)

    def inverse_transform(self, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return np.exp(y) - self.offset


class DetrendTransform(SeriesTransform):
    """Remove a polynomial trend fit against the positional index.

    With ``poly_order=1`` this removes a linear trend; ``poly_order=2`` a quadratic one
    (matching the air-passengers example in the design doc's source article).
    """

    def __init__(self, poly_order: int = 1) -> None:
        if poly_order < 1:
            raise ValueError("poly_order must be >= 1")
        self.poly_order = poly_order
        self.coeffs_: np.ndarray | None = None

    def fit(self, y: np.ndarray, t: np.ndarray) -> DetrendTransform:
        self.coeffs_ = np.polyfit(t.astype(float), y, deg=self.poly_order)
        self.fitted_ = True
        return self

    def _trend(self, t: np.ndarray) -> np.ndarray:
        assert self.coeffs_ is not None
        return np.polyval(self.coeffs_, t.astype(float))

    def transform(self, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return y - self._trend(t)

    def inverse_transform(self, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return y + self._trend(t)


class DeseasonTransform(SeriesTransform):
    """Subtract an additive seasonal component estimated per position-in-cycle.

    The seasonal profile is the mean of the (detrended) series within each phase of the
    cycle of length ``period``; it is evaluated by ``t % period`` so it extends naturally
    to forecast positions.
    """

    def __init__(self, period: int) -> None:
        if period < 2:
            raise ValueError("period must be >= 2")
        self.period = period
        self.profile_: np.ndarray | None = None

    def fit(self, y: np.ndarray, t: np.ndarray) -> DeseasonTransform:
        phase = t % self.period
        profile = np.zeros(self.period)
        for p in range(self.period):
            vals = y[phase == p]
            profile[p] = vals.mean() if vals.size else 0.0
        # Center so the seasonal component has zero mean (keeps level with detrend).
        profile -= profile.mean()
        self.profile_ = profile
        self.fitted_ = True
        return self

    def _seasonal(self, t: np.ndarray) -> np.ndarray:
        assert self.profile_ is not None
        return self.profile_[t % self.period]

    def transform(self, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return y - self._seasonal(t)

    def inverse_transform(self, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return y + self._seasonal(t)


class StandardScaleTransform(SeriesTransform):
    """Standardise to zero mean / unit variance."""

    def __init__(self) -> None:
        self.mean_ = 0.0
        self.std_ = 1.0

    def fit(self, y: np.ndarray, t: np.ndarray) -> StandardScaleTransform:
        self.mean_ = float(np.mean(y))
        self.std_ = float(np.std(y)) or 1.0
        self.fitted_ = True
        return self

    def transform(self, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return (y - self.mean_) / self.std_

    def inverse_transform(self, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return y * self.std_ + self.mean_


class RobustScaleTransform(SeriesTransform):
    """Scale by the median and IQR (robust to outliers — good for financial data)."""

    def __init__(self) -> None:
        self.center_ = 0.0
        self.scale_ = 1.0

    def fit(self, y: np.ndarray, t: np.ndarray) -> RobustScaleTransform:
        self.center_ = float(np.median(y))
        q75, q25 = np.percentile(y, [75, 25])
        self.scale_ = float(q75 - q25) or 1.0
        self.fitted_ = True
        return self

    def transform(self, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return (y - self.center_) / self.scale_

    def inverse_transform(self, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return y * self.scale_ + self.center_


class DifferenceTransform(SeriesTransform):
    """First-difference transform with an anchored inverse.

    ``inverse_transform`` assumes the values passed in are a *contiguous* block starting
    at the position immediately after the last training observation (i.e. a forecast
    horizon). This is the common case; it is therefore best used as the outermost
    transform when forecasting.
    """

    def __init__(self) -> None:
        self.anchor_value_: float = 0.0
        self.anchor_pos_: int = -1

    def fit(self, y: np.ndarray, t: np.ndarray) -> DifferenceTransform:
        self.anchor_value_ = float(y[-1])
        self.anchor_pos_ = int(t[-1])
        self.fitted_ = True
        return self

    def transform(self, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        self._check_fitted()
        # Anchored first differences: d[0] = y[0] - anchor, d[k] = y[k] - y[k-1].
        return np.diff(y, prepend=self.anchor_value_)

    def inverse_transform(self, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        self._check_fitted()
        # Cumulatively integrate differences from the stored anchor (exact inverse).
        return self.anchor_value_ + np.cumsum(y)
