"""Command-line interface: run a forecast or launch the API from the terminal."""

from __future__ import annotations

import argparse
import sys

from lstm_forecast.config import get_settings


def _cmd_forecast(args: argparse.Namespace) -> int:
    from lstm_forecast import Forecaster, Pipeline
    from lstm_forecast.ai import generate_insights
    from lstm_forecast.data import add_finance_features, load_prices
    from lstm_forecast.transforms import default_finance_transformer

    df = load_prices(args.ticker, allow_synthetic_fallback=args.allow_synthetic)
    feat = add_finance_features(df, fourier_periods=(5.0,)) if args.features else df
    exog = feat.drop(columns=["close"]) if args.features else None

    f = Forecaster(
        y=feat["close"],
        current_dates=feat.index,
        future_dates=args.horizon,
        test_length=args.test_length,
        exog=exog,
        name="lstm",
    )
    transformer, reverter = default_finance_transformer(seasonal_period=args.seasonal_period)
    pipe = Pipeline(transformer=transformer, reverter=reverter)
    result = pipe.fit_predict(f, lags=args.lags, epochs=args.epochs, alpha=args.alpha)

    print(f"\n=== {args.ticker} — test-set benchmark (RMSE-sorted) ===")
    print(result.metrics_frame().to_string())
    print(f"\n=== {args.horizon}-step forecast ===")
    for d, p, lo, hi in zip(result.future_dates, result.point, result.lower, result.upper,
                            strict=False):
        print(f"  {str(d)[:10]}  {p:10.4f}  [{lo:10.4f}, {hi:10.4f}]")

    if args.insights:
        print("\n=== AI insights ===")
        print(generate_insights(result, label=args.ticker))

    if args.plot:
        import matplotlib.pyplot as plt

        f.plot(ci=True)
        out = args.plot if isinstance(args.plot, str) else f"{args.ticker}_forecast.png"
        plt.tight_layout()
        plt.savefig(out, dpi=120)
        print(f"\nSaved plot to {out}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "lstm_forecast.api.main:app",
        host=args.host or settings.api.host,
        port=args.port or settings.api.port,
        reload=args.reload,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lstm-forecast", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    fc = sub.add_parser("forecast", help="Run a forecast for a ticker.")
    fc.add_argument("ticker")
    fc.add_argument("--horizon", type=int, default=21)
    fc.add_argument("--test-length", type=int, default=42)
    fc.add_argument("--lags", type=int, default=21)
    fc.add_argument("--epochs", type=int, default=60)
    fc.add_argument("--alpha", type=float, default=0.1)
    fc.add_argument("--seasonal-period", type=int, default=5)
    fc.add_argument("--features", action="store_true", help="Add finance features (multivariate).")
    fc.add_argument("--insights", action="store_true", help="Generate AI insights.")
    fc.add_argument("--plot", nargs="?", const=True, default=False, help="Save a forecast plot.")
    fc.add_argument("--allow-synthetic", action="store_true",
                    help="Fall back to synthetic data if the provider is unavailable.")
    fc.set_defaults(func=_cmd_forecast)

    sv = sub.add_parser("serve", help="Launch the FastAPI service.")
    sv.add_argument("--host", default=None)
    sv.add_argument("--port", type=int, default=None)
    sv.add_argument("--reload", action="store_true")
    sv.set_defaults(func=_cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
