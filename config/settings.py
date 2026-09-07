"""Application settings and configuration management for GroundLens."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """GroundLens settings backed by environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 1. Open-Weights LLM Configuration (Groq)
    # GroundLens strictly relies on open-weights models served via Groq.
    # Disbarred from primary reasoning: GPT-4, GPT-4o, Claude, Gemini.
    GROQ_API_KEY: Optional[str] = Field(
        default=None,
        description="Groq API key for running open-weights reasoning models.",
    )
    GROQ_MODEL: str = Field(
        default="llama-3.3-70b-versatile",
        description="Open-weights model on Groq (e.g. llama-3.3-70b-versatile, llama-3.1-8b-instant).",
    )

    # 2. Community Discussion APIs (Reddit via PRAW & Stack Exchange)
    REDDIT_CLIENT_ID: Optional[str] = Field(
        default=None,
        description="Reddit API Client ID from reddit.com/prefs/apps.",
    )
    REDDIT_CLIENT_SECRET: Optional[str] = Field(
        default=None,
        description="Reddit API Client Secret.",
    )
    REDDIT_USER_AGENT: str = Field(
        default="GroundLens/1.0 (Research Agent)",
        description="User agent string sent with Reddit API requests.",
    )
    STACKEXCHANGE_KEY: Optional[str] = Field(
        default=None,
        description="Optional Stack Exchange application key to increase rate limits.",
    )

    # 3. Observability (LangSmith)
    LANGCHAIN_API_KEY: Optional[str] = Field(
        default=None,
        description="LangSmith API key for execution tracing.",
    )
    LANGCHAIN_PROJECT: str = Field(
        default="groundlens",
        description="LangSmith project name for traces.",
    )
    LANGCHAIN_TRACING_V2: bool = Field(
        default=False,
        description="Enable LangSmith v2 tracing.",
    )

    # 4. Caching
    CACHE_TTL_SECONDS: int = Field(
        default=600,
        description="Default TTL in seconds for cached external responses (10 mins).",
    )
    CACHE_DB_PATH: str = Field(
        default=".cache.db",
        description="Path to local SQLite cache file.",
    )

    @property
    def has_groq_key(self) -> bool:
        """Check if Groq API key is configured."""
        return bool(self.GROQ_API_KEY and self.GROQ_API_KEY.strip() and not self.GROQ_API_KEY.startswith("your_"))

    @property
    def has_reddit_creds(self) -> bool:
        """Check if complete Reddit API credentials are present."""
        return bool(
            self.REDDIT_CLIENT_ID
            and self.REDDIT_CLIENT_SECRET
            and not self.REDDIT_CLIENT_ID.startswith("your_")
            and not self.REDDIT_CLIENT_SECRET.startswith("your_")
        )

    @property
    def has_langsmith_key(self) -> bool:
        """Check if LangSmith tracing credentials are provided."""
        return bool(
            self.LANGCHAIN_API_KEY
            and self.LANGCHAIN_API_KEY.strip()
            and not self.LANGCHAIN_API_KEY.startswith("your_")
        )


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Retrieve singleton Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
