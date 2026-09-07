"""LangGraph node: Prompt Injection Detector."""

from __future__ import annotations

import time
from agent.state import AgentState
from observability.langsmith import record_node_event
from safety.injection_detector import injection_detector


def prompt_injection_detector_node(state: AgentState) -> AgentState:
    """Detect and quarantine adversarial prompt injection patterns within retrieved documents."""
    start_time = time.time()
    docs_to_check = state.get("retrieved_documents", [])

    safe_docs, blocked_docs = injection_detector.filter_documents(docs_to_check)
    duration_ms = (time.time() - start_time) * 1000

    status = "BLOCKED_INJECTION" if blocked_docs else "PASSED"
    record_node_event(
        state,
        node_name="prompt_injection_detector",
        status=status,
        details={
            "checked_count": len(docs_to_check),
            "safe_count": len(safe_docs),
            "blocked_count": len(blocked_docs),
            "blocked_ids": [d.get("source_id") for d in blocked_docs],
        },
        duration_ms=duration_ms,
    )

    errors = list(state.get("errors", []))
    if blocked_docs:
        errors.append(f"Prompt injection detector quarantined {len(blocked_docs)} untrusted document(s).")

    return {
        **state,
        "filtered_documents": safe_docs,
        "errors": errors,
    }
