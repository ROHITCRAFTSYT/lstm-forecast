"""A RAG chat assistant grounded in project docs and the current forecast run.

Uses Claude tool-use to let the model pull in documentation chunks and run metrics on
demand, rather than stuffing everything into the prompt. Without an API key it falls back
to returning the most relevant retrieved documentation for the question.
"""

from __future__ import annotations

import json
from typing import Any

from lstm_forecast.ai.client import AIClient
from lstm_forecast.ai.doc_index import DocIndex
from lstm_forecast.forecasting.forecaster import ForecastResult

_SYSTEM = (
    "You are an assistant for the lstm-forecast library. Answer the user's question using "
    "the tools to retrieve documentation and the current forecast run's results. Ground "
    "every factual claim in retrieved content; if the docs don't cover something, say so. "
    "Be concise and never give financial advice."
)

_TOOLS: list[dict[str, Any]] = [
    {
        "name": "retrieve_docs",
        "description": "Search the project documentation and run notes for relevant passages.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "get_run_metrics",
        "description": "Get the current forecast run's test-set metrics (model vs baselines).",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_forecast",
        "description": "Get the current forecast point path and interval summary.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


class ChatAssistant:
    """Grounded chat over docs + the active :class:`ForecastResult`."""

    def __init__(
        self,
        doc_index: DocIndex,
        *,
        result: ForecastResult | None = None,
        client: AIClient | None = None,
    ) -> None:
        self.doc_index = doc_index
        self.result = result
        self.client = client or AIClient()

    def _dispatch(self, name: str, tool_input: dict[str, Any]) -> str:
        if name == "retrieve_docs":
            chunks = self.doc_index.search(tool_input.get("query", ""), k=4)
            return "\n\n---\n\n".join(f"[{c.source}] {c.text}" for c in chunks) or "No matches."
        if name == "get_run_metrics":
            if self.result is None:
                return "No active forecast run."
            return self.result.metrics_frame().to_string()
        if name == "get_forecast":
            if self.result is None:
                return "No active forecast run."
            return json.dumps(
                {
                    "point": [round(v, 4) for v in self.result.point.tolist()],
                    "lower": [round(v, 4) for v in self.result.lower.tolist()],
                    "upper": [round(v, 4) for v in self.result.upper.tolist()],
                    "alpha": self.result.alpha,
                }
            )
        return f"Unknown tool: {name}"

    def ask(self, question: str) -> str:
        """Answer ``question``. Uses Claude tool-use if available, else doc retrieval."""
        if not self.client.available:
            return self._fallback(question)
        try:
            return self.client.tool_loop(
                system=_SYSTEM,
                messages=[{"role": "user", "content": question}],
                tools=_TOOLS,
                dispatch=self._dispatch,
            )
        except Exception:
            return self._fallback(question)

    def _fallback(self, question: str) -> str:
        chunks = self.doc_index.search(question, k=3)
        if not chunks:
            return (
                "AI chat is offline (no ANTHROPIC_API_KEY) and no relevant documentation was "
                "found. Set ANTHROPIC_API_KEY to enable grounded answers."
            )
        body = "\n\n".join(f"- {c.text}" for c in chunks)
        return (
            "AI chat is offline (no ANTHROPIC_API_KEY); returning the most relevant "
            f"documentation for your question:\n\n{body}"
        )
