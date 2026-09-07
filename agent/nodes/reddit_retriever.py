"""LangGraph node: Reddit Retriever."""

from __future__ import annotations

import time
from agent.state import AgentState
from observability.langsmith import record_node_event
from tools.reddit_tool import reddit_service


def reddit_retriever_node(state: AgentState) -> AgentState:
    """Execute live Reddit search using PRAW and append normalized evidence."""
    if "reddit" not in state.get("required_sources", []):
        return state

    start_time = time.time()
    question = state.get("question", "")
    existing_docs = list(state.get("retrieved_documents", []))
    errors = list(state.get("errors", []))

    results = []
    try:
        results = reddit_service.search_discussions(question, limit=3, max_comments=2)
    except Exception as e:
        errors.append(f"Reddit retrieval failed: {e}")

    # Re-index source IDs if multiple docs exist
    offset = len([d for d in existing_docs if d.get("source_type") == "reddit"])
    for i, r in enumerate(results, start=offset + 1):
        r["source_id"] = f"reddit_{i:02d}"

    updated_docs = existing_docs + results
    duration_ms = (time.time() - start_time) * 1000

    record_node_event(
        state,
        node_name="reddit_retriever",
        status="SUCCESS" if results else "EMPTY",
        details={"retrieved_count": len(results)},
        duration_ms=duration_ms,
    )

    return {
        **state,
        "retrieved_documents": updated_docs,
        "errors": errors,
    }
