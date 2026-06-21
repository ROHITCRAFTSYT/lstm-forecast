"""Claude-powered AI layer: insights, structured tuning, and a RAG chat assistant.

Everything here degrades gracefully: with no ``ANTHROPIC_API_KEY`` the functions return
useful deterministic fallbacks instead of raising, so the rest of the system is unaffected.
"""

from __future__ import annotations

from lstm_forecast.ai.assistant import ChatAssistant
from lstm_forecast.ai.client import AIClient, AIUnavailableError
from lstm_forecast.ai.doc_index import DocIndex
from lstm_forecast.ai.insights import generate_insights
from lstm_forecast.ai.tuner import TuningSuggestion, suggest_tuning

__all__ = [
    "AIClient",
    "AIUnavailableError",
    "ChatAssistant",
    "DocIndex",
    "TuningSuggestion",
    "generate_insights",
    "suggest_tuning",
]
