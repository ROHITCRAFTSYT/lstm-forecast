"""Financial feature engineering: returns, technical indicators, calendar/Fourier terms.

All features are computed causally (no look-ahead): every value at time ``t`` uses only
information available up to and including ``t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(price: pd.Series) -> pd.Series:
    """Log returns ``ln(P_t / P_{t-1})``."""
    return pd.Series(np.log(price / price.shift(1)), index=price.index, name="log_return")


def rsi(price: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing)."""
    delta = price.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(
    price: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line and histogram."""
    ema_fast = price.ewm(span=fast, adjust=False).mean()
    ema_slow = price.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": macd_line - signal_line,
        }
    )


def bollinger(price: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    """Bollinger bands and the %B position within the bands."""
    mid = price.rolling(window).mean()
    std = price.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    width = (upper - lower) / mid
    pct_b = (price - lower) / (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame({"bb_width": width, "bb_pct_b": pct_b})


def rolling_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
    """Rolling standard deviation of returns (realised volatility)."""
    return returns.rolling(window).std()


def fourier_terms(
    index: pd.DatetimeIndex,
    period: float,
    n_harmonics: int = 2,
    prefix: str = "fourier",
) -> pd.DataFrame:
    """Sine/cosine Fourier terms encoding seasonality of a given ``period``.

    Useful for weekly (period≈5 trading days) or annual (period≈252) seasonality.
    """
    # Ordinal position keeps phase continuity across the series.
    t = np.arange(len(index), dtype=float)
    out: dict[str, np.ndarray] = {}
    for k in range(1, n_harmonics + 1):
        out[f"{prefix}_sin{k}_{int(period)}"] = np.sin(2 * np.pi * k * t / period)
        out[f"{prefix}_cos{k}_{int(period)}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(out, index=index)


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Cyclically-encoded calendar features (day of week, month)."""
    dow = index.dayofweek.to_numpy(dtype=float)
    month = index.month.to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "dow_sin": np.sin(2 * np.pi * dow / 5.0),
            "dow_cos": np.cos(2 * np.pi * dow / 5.0),
            "month_sin": np.sin(2 * np.pi * (month - 1) / 12.0),
            "month_cos": np.cos(2 * np.pi * (month - 1) / 12.0),
        },
        index=index,
    )


def add_finance_features(
    df: pd.DataFrame,
    *,
    price_col: str = "close",
    include: tuple[str, ...] = (
        "log_return",
        "volatility",
        "rsi",
        "macd",
        "bollinger",
        "calendar",
    ),
    fourier_periods: tuple[float, ...] = (),
    fourier_harmonics: int = 2,
    dropna: bool = True,
) -> pd.DataFrame:
    """Augment an OHLCV frame with the requested causal features.

    Parameters mirror the design doc's finance feature set. Returns a new frame; the
    original ``price_col`` is preserved as the forecasting target.
    """
    if price_col not in df.columns:
        raise KeyError(f"price_col {price_col!r} not in dataframe columns {list(df.columns)}")

    price = df[price_col]
    parts: list[pd.DataFrame | pd.Series] = [df.copy()]
    rets = log_returns(price)

    if "log_return" in include:
        parts.append(rets.rename("log_return"))
    if "volatility" in include:
        parts.append(rolling_volatility(rets).rename("volatility"))
    if "rsi" in include:
        parts.append(rsi(price).rename("rsi"))
    if "macd" in include:
        parts.append(macd(price))
    if "bollinger" in include:
        parts.append(bollinger(price))
    if "calendar" in include and isinstance(df.index, pd.DatetimeIndex):
        parts.append(calendar_features(df.index))
    for period in fourier_periods:
        if isinstance(df.index, pd.DatetimeIndex):
            parts.append(fourier_terms(df.index, period, n_harmonics=fourier_harmonics))

    out = pd.concat(parts, axis=1)
    if dropna:
        out = out.dropna()
    return out
