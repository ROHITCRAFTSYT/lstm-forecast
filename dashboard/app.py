"""Streamlit dashboard for lstm-forecast.

Run with:  streamlit run dashboard/app.py

Consumes the core library in-process — no business logic here, only UI wiring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from lstm_forecast import Forecaster, Pipeline
from lstm_forecast.ai import generate_insights
from lstm_forecast.ai.assistant import ChatAssistant
from lstm_forecast.ai.client import AIClient
from lstm_forecast.ai.doc_index import DocIndex
from lstm_forecast.data import add_finance_features, load_prices
from lstm_forecast.forecasting.forecaster import ModelSpec
from lstm_forecast.transforms import default_finance_transformer

st.set_page_config(page_title="lstm-forecast", page_icon="📈", layout="wide")
st.title("📈 lstm-forecast — LSTM + RAG + Claude")
st.caption("Forecasts are uncertain and **not financial advice**.")


@st.cache_data(show_spinner=False)
def _load(ticker: str) -> pd.DataFrame:
    return load_prices(ticker, allow_synthetic_fallback=True)


with st.sidebar:
    st.header("Configuration")
    ticker = st.text_input("Ticker", value="AAPL")
    horizon = st.slider("Forecast horizon", 5, 60, 21)
    test_length = st.slider("Test length", 10, 120, 42)
    lags = st.slider("Lags (lookback)", 5, 90, 21)
    epochs = st.slider("Epochs", 10, 200, 60, step=10)
    alpha = st.select_slider("Interval level", options=[0.8, 0.9, 0.95], value=0.9)
    use_features = st.checkbox("Multivariate finance features", value=True)
    use_rag = st.checkbox("Retrieval-augmented (RAG)", value=False)
    run_backtest = st.checkbox("Dynamic (backtested) intervals", value=False)
    run = st.button("Run forecast", type="primary")

ai_enabled = AIClient().available
st.sidebar.markdown(f"**Claude AI:** {'🟢 enabled' if ai_enabled else '⚪ offline (no API key)'}")


def _run_forecast() -> tuple[Forecaster, object]:
    df = _load(ticker)
    if use_features:
        feat = add_finance_features(df, fourier_periods=(5.0,))
        target, dates, exog = feat["close"], feat.index, feat.drop(columns=["close"])
    else:
        target, dates, exog = df["close"], df.index, None

    f = Forecaster(
        y=target, current_dates=dates, future_dates=horizon, test_length=test_length, exog=exog
    )
    transformer, reverter = default_finance_transformer(seasonal_period=5)
    if use_rag:
        split = f.y.size - f.test_length
        transformer.fit(f.y[:split], np.arange(split))
        ref = transformer.transform(f.y[:split], np.arange(split))
        from lstm_forecast.rag import build_analog_retriever

        f.attach_retriever(build_analog_retriever(ref, window_len=lags))
    pipe = Pipeline(transformer=transformer, reverter=reverter)
    result = pipe.fit_predict(
        f,
        spec=ModelSpec(lags=lags, epochs=epochs),
        alpha=round(1 - alpha, 3),
        run_backtest=run_backtest,
    )
    return f, result


if run:
    with st.spinner("Training and forecasting…"):
        f, result = _run_forecast()
    st.session_state["result"] = result
    st.session_state["label"] = ticker

result = st.session_state.get("result")

if result is not None:
    label = st.session_state.get("label", "series")
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Forecast")
        try:
            import plotly.graph_objects as go

            hist = pd.Series(result.history_values, index=result.history_dates)
            fig = go.Figure()
            fig.add_scatter(x=hist.index[-200:], y=hist.values[-200:], name="history",
                            line=dict(color="#1f4e79"))
            fig.add_scatter(x=result.future_dates, y=result.point, name="forecast",
                            line=dict(color="#c0392b"))
            fig.add_scatter(x=result.future_dates, y=result.upper, name="upper",
                            line=dict(width=0), showlegend=False)
            fig.add_scatter(x=result.future_dates, y=result.lower, name="interval",
                            fill="tonexty", line=dict(width=0),
                            fillcolor="rgba(192,57,43,0.2)")
            if result.test_dates is not None and result.test_pred is not None:
                fig.add_scatter(x=result.test_dates, y=result.test_pred, name="test forecast",
                                line=dict(color="#e67e22", dash="dash"))
            fig.update_layout(height=440, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.line_chart(pd.Series(result.point, index=result.future_dates))

    with col2:
        st.subheader("Benchmark (test set)")
        frame = result.metrics_frame()
        if not frame.empty:
            st.dataframe(frame.round(4), use_container_width=True)
            st.caption(f"Best model: **{frame.index[0]}** (lowest RMSE)")
        if result.interval:
            st.metric("Interval coverage (test)",
                      f"{result.interval.get('coverage', float('nan')):.2f}",
                      f"nominal {result.interval.get('nominal', 0.9):.2f}")

    st.subheader("🤖 AI insights")
    if st.button("Generate insights"):
        with st.spinner("Asking Claude…" if ai_enabled else "Building summary…"):
            st.write(generate_insights(result, label=label))

    st.subheader("💬 Ask the assistant")
    question = st.text_input("Question about this run or the library")
    if question:
        idx = DocIndex()
        for cand in ("README.md", "docs"):
            from pathlib import Path

            if Path(cand).exists():
                idx.add_paths([cand])
        idx.build()
        assistant = ChatAssistant(idx, result=result)
        with st.spinner("Thinking…"):
            st.write(assistant.ask(question))
else:
    st.info("Configure options in the sidebar and click **Run forecast**.")
