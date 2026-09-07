"""LangGraph node: Scope Checker."""

from __future__ import annotations

import time
from agent.router import query_router
from agent.state import AgentState
from observability.langsmith import record_node_event


def scope_checker_node(state: AgentState) -> AgentState:
    """Evaluate whether question falls within the supported domain scope."""
    start_time = time.time()
    question = state.get("question", "")

    in_scope, reason = query_router.check_scope(question)
    duration_ms = (time.time() - start_time) * 1000

    if not in_scope:
        refusal_msg = (
            "I’m designed for grounded research using supported community discussions and live structured APIs. "
            "I don't have a grounded source for this request."
        )
        record_node_event(
            state,
            node_name="scope_checker",
            status="OUT_OF_SCOPE",
            details={"reason": reason},
            duration_ms=duration_ms,
        )
        return {
            **state,
            "scope_status": "out_of_scope",
            "grounding_status": "refused",
            "confidence": 0.0,
            "answer": refusal_msg,
            "errors": [f"Scope checker: {reason}"],
        }

    record_node_event(
        state,
        node_name="scope_checker",
        status="PASSED",
        details={"reason": "in_scope"},
        duration_ms=duration_ms,
    )
    return {
        **state,
        "scope_status": "in_scope",
    }
