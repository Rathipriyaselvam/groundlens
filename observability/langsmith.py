"""LangSmith tracing configuration, custom metadata enrichment, and execution logging."""

from __future__ import annotations

import functools
import os
import time
from typing import Any, Callable, Dict, List, Optional
from config.logging_config import logger
from config.settings import get_settings


def configure_langsmith() -> bool:
    """Configure LangSmith environment variables if credentials are provided."""
    settings = get_settings()
    if settings.has_langsmith_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true" if settings.LANGCHAIN_TRACING_V2 else "false"
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY or ""
        os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
        logger.info("LangSmith tracing enabled for project '%s'.", settings.LANGCHAIN_PROJECT)
        return True
    else:
        logger.debug("LangSmith credentials not configured; external tracing is inactive.")
        return False


def build_trace_metadata(state: Dict[str, Any]) -> Dict[str, Any]:
    """Construct structured observability metadata for trace recording."""
    return {
        "intent": state.get("intent", "unknown"),
        "sources": state.get("required_sources", []),
        "grounding_status": state.get("grounding_status", "pending"),
        "confidence": state.get("confidence", 0.0),
        "citation_count": len(state.get("citations", [])),
        "errors_count": len(state.get("errors", [])),
        "documents_retrieved": len(state.get("retrieved_documents", [])),
        "documents_filtered": len(state.get("filtered_documents", [])),
    }


def record_node_event(
    state: Dict[str, Any],
    node_name: str,
    status: str,
    details: Optional[Dict[str, Any]] = None,
    duration_ms: Optional[float] = None,
) -> None:
    """Log an execution event into the state's tool_events audit trail."""
    event: Dict[str, Any] = {
        "node": node_name,
        "status": status,
        "timestamp": time.time(),
        "duration_ms": duration_ms if duration_ms is not None else 0.0,
        "details": details or {},
    }
    if "tool_events" not in state or state["tool_events"] is None:
        state["tool_events"] = []
    state["tool_events"].append(event)
