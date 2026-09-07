"""LangGraph node: Stack Exchange Retriever."""

from __future__ import annotations

import time
from agent.state import AgentState
from observability.langsmith import record_node_event
from tools.stackexchange_tool import stackexchange_service


def stackexchange_retriever_node(state: AgentState) -> AgentState:
    """Execute live Stack Exchange search and append normalized evidence."""
    if "stackexchange" not in state.get("required_sources", []):
        return state

    start_time = time.time()
    question = state.get("question", "")
    existing_docs = list(state.get("retrieved_documents", []))
    errors = list(state.get("errors", []))

    results = []
    try:
        results = stackexchange_service.search_discussions(question, limit=3)
    except Exception as e:
        errors.append(f"Stack Exchange retrieval failed: {e}")

    offset = len([d for d in existing_docs if d.get("source_type") == "stackexchange"])
    for i, r in enumerate(results, start=offset + 1):
        r["source_id"] = f"stackexchange_{i:02d}"

    updated_docs = existing_docs + results
    duration_ms = (time.time() - start_time) * 1000

    record_node_event(
        state,
        node_name="stackexchange_retriever",
        status="SUCCESS" if results else "EMPTY",
        details={"retrieved_count": len(results)},
        duration_ms=duration_ms,
    )

    return {
        **state,
        "retrieved_documents": updated_docs,
        "errors": errors,
    }
