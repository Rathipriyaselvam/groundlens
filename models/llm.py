"""Open-weights LLM factory leveraging Groq.

Enforces strict open-weights model usage. Disallows closed-source models (GPT, Claude, Gemini)
and fails fast if required configurations are absent.
"""

from __future__ import annotations

import os
from typing import Optional
from langchain_groq import ChatGroq
from config.logging_config import logger
from config.settings import get_settings


DISALLOWED_MODELS = {
    "gpt-4",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "claude-3-opus",
    "claude-3-sonnet",
    "claude-3-5-sonnet",
    "claude-2",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-pro",
}


def get_groq_llm(temperature: float = 0.0, max_tokens: int = 1024) -> ChatGroq:
    """Instantiate a ChatGroq client enforcing open-weights model constraints.

    Raises:
        RuntimeError: If GROQ_API_KEY is not set or placeholder is detected.
        ValueError: If a prohibited closed model is specified in GROQ_MODEL.
    """
    settings = get_settings()

    if not settings.has_groq_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing or invalid. GroundLens requires an open-weights reasoning model "
            "hosted via Groq (e.g., Llama 3.3 70B, Llama 3.1 8B, Qwen 2.5). "
            "Please set GROQ_API_KEY in your .env file. "
            "Silent fallback to closed models (GPT-4/Claude/Gemini) is strictly disallowed."
        )

    model_name = settings.GROQ_MODEL.strip()

    # Safety check: Verify the configured model is not a closed-source model
    model_lower = model_name.lower()
    for disallowed in DISALLOWED_MODELS:
        if disallowed in model_lower:
            raise ValueError(
                f"Prohibited model detected: '{model_name}'. GroundLens strictly requires an open-weights model "
                f"(such as 'llama-3.3-70b-versatile', 'llama-3.1-8b-instant', or 'mixtral-8x7b-32768'). "
                f"Closed proprietary models ({disallowed}) are disallowed."
            )

    logger.debug("Initializing ChatGroq with open-weights model: %s (temp=%.2f)", model_name, temperature)

    return ChatGroq(
        groq_api_key=settings.GROQ_API_KEY,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
