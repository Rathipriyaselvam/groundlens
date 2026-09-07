"""LangGraph node: Content Safety Filter."""

from __future__ import annotations

import time
from agent.state import AgentState
from observability.langsmith import record_node_event
from safety.content_filter import content_filter


def content_safety_filter_node(state: AgentState) -> AgentState:
    """Evaluate retrieved documents for hate speech, harassment, explicit, or hazardous content."""
    start_time = time.time()
    raw_docs = state.get("retrieved_documents", [])

    safe_docs, filtered_docs = content_filter.filter_documents(raw_docs)
    duration_ms = (time.time() - start_time) * 1000

    status = "WARNING_FILTERED" if filtered_docs else "PASSED"
    record_node_event(
        state,
        node_name="content_safety_filter",
        status=status,
        details={
            "initial_count": len(raw_docs),
            "safe_count": len(safe_docs),
            "filtered_count": len(filtered_docs),
            "filtered_ids": [d.get("source_id") for d in filtered_docs],
        },
        duration_ms=duration_ms,
    )

    errors = list(state.get("errors", []))
    if filtered_docs:
        errors.append(f"Content safety filter removed {len(filtered_docs)} unsafe source(s).")

    return {
        **state,
        "retrieved_documents": safe_docs,
        "errors": errors,
    }
