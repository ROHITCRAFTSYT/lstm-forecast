"""Centralised configuration via pydantic-settings.

Settings are read from environment variables (and an optional ``.env`` file) using the
``LSTM_FORECAST_`` prefix. Nested settings use a ``__`` delimiter, e.g.
``LSTM_FORECAST_AI__MODEL=claude-opus-4-8``.

Nothing here requires any secret to be present — the AI sub-config simply has an empty
API key by default, and the rest of the system runs fully without it.
"""

from __future__ import annotations

import functools
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AISettings(BaseSettings):
    """Configuration for the Claude-powered AI layer."""

    model_config = SettingsConfigDict(
        env_prefix="LSTM_FORECAST_AI__", extra="ignore", populate_by_name=True
    )

    api_key: str = Field(
        default="",
        validation_alias="ANTHROPIC_API_KEY",
        description="Anthropic API key. Empty disables all AI features (graceful fallback).",
    )
    model: str = Field(default="claude-opus-4-8", description="Claude model id.")
    effort: str = Field(default="high", description="Reasoning effort: low|medium|high|max.")
    max_tokens: int = Field(default=4096, ge=1)
    request_timeout: float = Field(default=60.0, gt=0)

    @property
    def enabled(self) -> bool:
        """True when an API key is configured and AI features can be used."""
        return bool(self.api_key.strip())


class APISettings(BaseSettings):
    """Configuration for the FastAPI service."""

    model_config = SettingsConfigDict(env_prefix="LSTM_FORECAST_API__", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "*"

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


class Settings(BaseSettings):
    """Top-level application settings."""

    model_config = SettingsConfigDict(
        env_prefix="LSTM_FORECAST_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    cache_dir: Path = Field(default=Path(".cache"))
    device: str = Field(default="auto", description="torch device: auto|cpu|cuda|mps")
    seed: int = Field(default=20)

    ai: AISettings = Field(default_factory=AISettings)
    api: APISettings = Field(default_factory=APISettings)

    def ensure_cache_dir(self) -> Path:
        """Create the cache directory if necessary and return it."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance.

    Cached so repeated calls are cheap and consistent within a process. Tests that need
    to vary the environment can call ``get_settings.cache_clear()``.
    """
    return Settings()
