"""LangGraph node: Input Guardrail."""

from __future__ import annotations

import time
from typing import Any, Dict
from agent.state import AgentState
from observability.langsmith import record_node_event
from safety.input_guardrail import input_guardrail


def input_guardrail_node(state: AgentState) -> AgentState:
    """Validate incoming question before processing."""
    start_time = time.time()
    question = state.get("question", "")

    validation = input_guardrail.validate_input(question)
    duration_ms = (time.time() - start_time) * 1000

    if not validation["allowed"]:
        reason = validation["reason"]
        msg = validation["message"]
        record_node_event(
            state,
            node_name="input_guardrail",
            status="FAILED",
            details={"reason": reason, "message": msg},
            duration_ms=duration_ms,
        )
        return {
            **state,
            "scope_status": "rejected",
            "grounding_status": "refused",
            "confidence": 0.0,
            "answer": msg,
            "errors": [f"Input guardrail rejected query: {reason}"],
        }

    record_node_event(
        state,
        node_name="input_guardrail",
        status="PASSED",
        details={"reason": "valid"},
        duration_ms=duration_ms,
    )
    return {
        **state,
        "scope_status": "in_scope",
    }
