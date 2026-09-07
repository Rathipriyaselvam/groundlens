"""LangGraph node: Final Response compilation."""

from __future__ import annotations

import time
from agent.state import AgentState
from observability.langsmith import record_node_event


def final_response_node(state: AgentState) -> AgentState:
    """Format and finalize the agent response and execution trace summary."""
    start_time = time.time()
    grounding_status = state.get("grounding_status", "pending")
    scope_status = state.get("scope_status", "in_scope")
    answer = state.get("answer", "")
    citations = state.get("citations", [])

    # Final safety sanity check
    if not answer:
        if scope_status == "out_of_scope":
            answer = "I’m designed for grounded research using supported community discussions and live structured APIs. I don't have a grounded source for this request."
        elif grounding_status == "insufficient":
            answer = "I don't have sufficient grounded information to answer that reliably."
        else:
            answer = "No grounded information could be verified for this request."

    duration_ms = (time.time() - start_time) * 1000
    record_node_event(
        state,
        node_name="final_response",
        status="COMPLETED",
        details={
            "grounding_status": grounding_status,
            "citations_count": len(citations),
            "answer_length": len(answer),
        },
        duration_ms=duration_ms,
    )

    return {
        **state,
        "answer": answer,
    }
