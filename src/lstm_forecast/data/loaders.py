"""Load financial time series from providers, CSV, or a synthetic generator.

The synthetic generator means the whole library — including the smoke run and CI — works
offline with no network and no API keys. ``load_prices`` will use yfinance when the
``data`` extra is installed and the network is reachable, and falls back clearly otherwise.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import pandas as pd

# Canonical OHLCV column names produced by all loaders.
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class DataLoadError(RuntimeError):
    """Raised when a data source cannot be loaded."""


def _ensure_datetime_index(df: pd.DataFrame, date_col: str | None) -> pd.DataFrame:
    if date_col is not None and date_col in df.columns:
        df = df.set_index(date_col)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.index.name = "date"
    return df


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case columns and keep a stable OHLCV subset where present."""
    df = df.rename(columns={c: str(c).strip().lower().replace(" ", "_") for c in df.columns})
    # yfinance sometimes returns 'adj_close'; prefer it as the close if present.
    if "adj_close" in df.columns and "close" in df.columns:
        df["close"] = df["adj_close"]
    keep = [c for c in OHLCV_COLUMNS if c in df.columns]
    if "close" not in keep:
        # Fall back to the first numeric column as 'close' so downstream code always has it.
        numeric = df.select_dtypes("number").columns.tolist()
        if not numeric:
            raise DataLoadError("No numeric columns found to use as 'close'.")
        df = df.rename(columns={numeric[0]: "close"})
        keep = ["close"] + [c for c in keep if c != "close"]
    return df[keep].astype(float)


def load_csv(
    path: str | Path,
    *,
    date_col: str | None = "date",
) -> pd.DataFrame:
    """Load OHLCV (or at least a price) series from a CSV file."""
    path = Path(path)
    if not path.exists():
        raise DataLoadError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    df = _ensure_datetime_index(df, date_col if date_col in df.columns else None)
    return _normalise_columns(df)


def load_synthetic_prices(
    n: int = 750,
    *,
    start: str = "2019-01-01",
    seed: int = 20,
    freq: str = "B",
    trend: float = 0.0004,
    volatility: float = 0.012,
    seasonal_amplitude: float = 0.05,
    seasonal_period: int = 252,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Generate a realistic synthetic OHLCV series (geometric random walk + seasonality).

    Used for offline tests, the smoke run, and as a deterministic fallback when no data
    provider is available. The output schema matches :func:`load_prices`.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=n, freq=freq)
    t = np.arange(n)
    seasonal = seasonal_amplitude * np.sin(2 * np.pi * t / seasonal_period)
    daily_ret = trend + seasonal / seasonal_period + rng.normal(0, volatility, size=n)
    close = start_price * np.exp(np.cumsum(daily_ret))
    intraday = np.abs(rng.normal(0, volatility, size=n)) * close
    open_ = close - rng.normal(0, volatility / 2, size=n) * close
    high = np.maximum(open_, close) + intraday / 2
    low = np.minimum(open_, close) - intraday / 2
    volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    df.index.name = "date"
    return df


def load_prices(
    ticker: str,
    *,
    start: str | None = None,
    end: str | None = None,
    cache_dir: str | Path | None = ".cache",
    use_cache: bool = True,
    allow_synthetic_fallback: bool = False,
) -> pd.DataFrame:
    """Load daily OHLCV price history for ``ticker``.

    Tries an on-disk parquet cache first, then yfinance (requires the ``data`` extra and
    network). Set ``allow_synthetic_fallback=True`` to return a deterministic synthetic
    series instead of raising when the provider is unavailable (useful for demos/tests).
    """
    cache_path: Path | None = None
    if cache_dir is not None and use_cache:
        cache_root = Path(cache_dir) / "prices"
        cache_root.mkdir(parents=True, exist_ok=True)
        key = f"{ticker.upper()}_{start or 'min'}_{end or 'max'}.parquet"
        cache_path = cache_root / key
        if cache_path.exists():
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                cache_path.unlink(missing_ok=True)

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        if allow_synthetic_fallback:
            return load_synthetic_prices()
        raise DataLoadError(
            "yfinance is not installed. Install the data extra "
            "(`pip install lstm-forecast[data]`) or pass allow_synthetic_fallback=True."
        ) from exc

    try:
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )
    except Exception as exc:
        if allow_synthetic_fallback:
            return load_synthetic_prices()
        raise DataLoadError(f"Failed to download {ticker}: {exc}") from exc

    if raw is None or raw.empty:
        if allow_synthetic_fallback:
            return load_synthetic_prices()
        raise DataLoadError(f"No data returned for ticker {ticker!r}.")

    df = _normalise_columns(_ensure_datetime_index(raw, None))
    if cache_path is not None:
        with contextlib.suppress(Exception):  # caching is best-effort (e.g. no pyarrow)
            df.to_parquet(cache_path)
    return df
