"""Natural-language forecast insights via Claude, with a deterministic fallback.

When the AI client is available, Claude turns the numbers into a concise analyst-style
narrative. When it isn't, a template summary is returned so the feature still produces
something useful (and the API/dashboard never break for lack of a key).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from lstm_forecast.ai.client import AIClient
from lstm_forecast.forecasting.forecaster import ForecastResult

_SYSTEM = (
    "You are a quantitative forecasting analyst. Given a model's numeric forecast, its "
    "uncertainty intervals, and how it compared to simple baselines on a held-out test "
    "set, write a concise, sober explanation for a technical user. Explain the projected "
    "direction and magnitude, the uncertainty, and whether the model meaningfully beat the "
    "baselines. Never give financial advice or guarantees; note that forecasts are "
    "uncertain. Keep it under 200 words."
)


def describe_forecast(result: ForecastResult, label: str = "the series") -> str:
    """Build a compact textual description of a forecast (used in prompts and fallbacks)."""
    point = np.asarray(result.point, dtype=float)
    last_hist = float(result.history_values[-1])
    end = float(point[-1])
    pct = (end / last_hist - 1) * 100 if last_hist else float("nan")
    direction = "up" if end > last_hist else "down" if end < last_hist else "flat"
    width = float(np.mean(result.upper - result.lower))

    lines = [
        f"Target: {label}",
        f"Last observed value: {last_hist:.4f}",
        f"Horizon: {len(point)} steps",
        f"Forecast end value: {end:.4f} ({pct:+.2f}% vs last), direction: {direction}",
        f"Mean {int((1 - result.alpha) * 100)}% interval width: {width:.4f}",
    ]
    if result.metrics:
        frame = result.metrics_frame()
        best = frame.index[0]
        lines.append("Test-set RMSE (lower is better):")
        for name, row in frame.iterrows():
            lines.append(f"  - {name}: rmse={row.get('rmse', float('nan')):.4f}")
        lines.append(f"Best model on test set: {best}")
    if result.interval:
        lines.append(
            f"Interval coverage on test set: {result.interval.get('coverage', float('nan')):.2f} "
            f"(nominal {result.interval.get('nominal', 1 - result.alpha):.2f})"
        )
    dm = result.significance.get("vs_naive") if result.significance else None
    if isinstance(dm, dict):
        lines.append(
            f"Diebold-Mariano vs naive: winner={dm.get('winner')}, "
            f"p={dm.get('p_value', float('nan')):.3f} "
            f"({'significant' if dm.get('significant') else 'not significant'} at 5%)"
        )
    return "\n".join(lines)


def _fallback_text(description: str) -> str:
    return (
        "AI insights are running in offline mode (no ANTHROPIC_API_KEY configured). "
        "Here is a deterministic summary of the forecast:\n\n"
        f"{description}\n\n"
        "Note: forecasts are uncertain and not financial advice. Set ANTHROPIC_API_KEY to "
        "enable Claude-generated narrative analysis."
    )


def generate_insights(
    result: ForecastResult,
    *,
    label: str = "the series",
    client: AIClient | None = None,
) -> str:
    """Return an NL explanation of ``result`` (Claude if available, else a template)."""
    description = describe_forecast(result, label)
    client = client or AIClient()
    if not client.available:
        return _fallback_text(description)
    user = (
        "Explain this forecast for a technical user:\n\n"
        f"{description}\n\n"
        "Focus on direction, magnitude, uncertainty, and baseline comparison."
    )
    try:
        return client.complete(system=_SYSTEM, messages=[{"role": "user", "content": user}])
    except Exception:
        return _fallback_text(description)


def stream_insights(
    result: ForecastResult,
    *,
    label: str = "the series",
    client: AIClient | None = None,
) -> Iterator[str]:
    """Stream the NL explanation. Falls back to yielding the template in one chunk."""
    description = describe_forecast(result, label)
    client = client or AIClient()
    if not client.available:
        yield _fallback_text(description)
        return
    user = f"Explain this forecast for a technical user:\n\n{description}"
    try:
        yield from client.stream(system=_SYSTEM, messages=[{"role": "user", "content": user}])
    except Exception:
        yield _fallback_text(description)
