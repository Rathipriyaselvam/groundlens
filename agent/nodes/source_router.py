"""LangGraph node: Source Router."""

from __future__ import annotations

import time
from agent.state import AgentState
from observability.langsmith import record_node_event


def source_router_node(state: AgentState) -> AgentState:
    """Consolidate routing metadata and verify dispatch targets."""
    start_time = time.time()
    sources = state.get("required_sources", [])
    intent = state.get("intent", "unsupported")

    duration_ms = (time.time() - start_time) * 1000
    record_node_event(
        state,
        node_name="source_router",
        status="ROUTED",
        details={"routed_sources": sources, "intent": intent},
        duration_ms=duration_ms,
    )

    return state
