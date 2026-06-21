"""Thin wrapper around the Anthropic SDK — the single chokepoint for all Claude calls.

Centralises model id, adaptive thinking, effort, streaming and structured parsing, and
exposes a single ``available`` flag so callers can branch to deterministic fallbacks when
no API key is configured. Per the project's claude-api guidance: default model
``claude-opus-4-8``, ``thinking={"type": "adaptive"}``, ``output_config.effort``, and
``messages.parse`` for structured output.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, TypeVar

from pydantic import BaseModel

from lstm_forecast.config import AISettings, get_settings

T = TypeVar("T", bound=BaseModel)


class AIUnavailableError(RuntimeError):
    """Raised when an AI call is attempted but the client is not available."""


class AIClient:
    """Lazily-constructed Anthropic client with graceful no-key handling."""

    def __init__(self, settings: AISettings | None = None) -> None:
        self.settings = settings or get_settings().ai
        self._client: Any = None
        self._import_ok = True
        if self.settings.enabled:
            try:
                import anthropic

                self._client = anthropic.Anthropic(
                    api_key=self.settings.api_key,
                    timeout=self.settings.request_timeout,
                )
            except ImportError:
                self._import_ok = False

    @property
    def available(self) -> bool:
        """True when a key is set and the SDK is importable."""
        return self.settings.enabled and self._import_ok and self._client is not None

    def _require(self) -> Any:
        if not self.available:
            raise AIUnavailableError(
                "Claude AI is unavailable. Set ANTHROPIC_API_KEY and install the 'ai' extra "
                "(`pip install lstm-forecast[ai]`)."
            )
        return self._client

    # ------------------------------------------------------------------- calls
    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        thinking: bool = True,
    ) -> str:
        """Non-streaming completion; returns the concatenated text blocks."""
        client = self._require()
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": max_tokens or self.settings.max_tokens,
            "system": system,
            "messages": messages,
            "output_config": {"effort": self.settings.effort},
        }
        if thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        resp = client.messages.create(**kwargs)
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

    def stream(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Stream text deltas. Recommended for chat/insight responses."""
        client = self._require()
        with client.messages.stream(
            model=self.settings.model,
            max_tokens=max_tokens or self.settings.max_tokens,
            system=system,
            messages=messages,
            output_config={"effort": self.settings.effort},
        ) as stream:
            yield from stream.text_stream

    def parse(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        max_tokens: int | None = None,
    ) -> T:
        """Structured output validated against a Pydantic ``schema`` via ``messages.parse``."""
        client = self._require()
        resp = client.messages.parse(
            model=self.settings.model,
            max_tokens=max_tokens or self.settings.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
        )
        if resp.parsed_output is None:  # pragma: no cover - defensive
            raise AIUnavailableError("Model did not return a parseable structured output.")
        return resp.parsed_output

    def tool_loop(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        dispatch: Any,
        max_turns: int = 6,
        max_tokens: int | None = None,
    ) -> str:
        """Run a manual tool-use loop. ``dispatch(name, input) -> str`` executes tools."""
        client = self._require()
        convo = list(messages)
        final_text = ""
        for _ in range(max_turns):
            resp = client.messages.create(
                model=self.settings.model,
                max_tokens=max_tokens or self.settings.max_tokens,
                system=system,
                messages=convo,
                tools=tools,
                output_config={"effort": self.settings.effort},
            )
            text = "".join(
                b.text for b in resp.content if getattr(b, "type", None) == "text"
            )
            if text:
                final_text = text
            if resp.stop_reason != "tool_use":
                break
            convo.append({"role": "assistant", "content": resp.content})
            tool_results = []
            for block in resp.content:
                if getattr(block, "type", None) == "tool_use":
                    result = dispatch(block.name, block.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result),
                        }
                    )
            convo.append({"role": "user", "content": tool_results})
        return final_text
