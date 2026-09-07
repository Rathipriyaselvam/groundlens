"""LangGraph node: Evidence Validator."""

from __future__ import annotations

import time
from agent.state import AgentState
from grounding.validator import evidence_validator
from observability.langsmith import record_node_event


def evidence_validator_node(state: AgentState) -> AgentState:
    """Validate completeness, sufficiency, and confidence of filtered evidence."""
    start_time = time.time()
    docs = state.get("filtered_documents", [])
    intent = state.get("intent", "unsupported")
    req_sources = state.get("required_sources", [])

    status, conf, notes = evidence_validator.evaluate_evidence(docs, intent, req_sources)
    duration_ms = (time.time() - start_time) * 1000

    record_node_event(
        state,
        node_name="evidence_validator",
        status="PASSED" if status == "grounded" else ("WARNING" if status == "partial" else "INSUFFICIENT"),
        details={"grounding_status": status, "confidence": conf, "notes": notes},
        duration_ms=duration_ms,
    )

    errors = list(state.get("errors", []))
    if status == "insufficient":
        errors.append("Evidence validator: Insufficient grounded evidence.")

    return {
        **state,
        "grounding_status": status,
        "confidence": conf,
        "errors": errors,
    }
