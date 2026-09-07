"""LangGraph node: Citation Validator."""

from __future__ import annotations

import time
from agent.state import AgentState
from grounding.citation import citation_validator
from observability.langsmith import record_node_event


def citation_validator_node(state: AgentState) -> AgentState:
    """Validate all citations in the generated answer against verified execution documents."""
    start_time = time.time()
    answer = state.get("answer", "")
    docs = state.get("filtered_documents", [])
    errors = list(state.get("errors", []))

    is_valid, sanitized_answer, verified_citations, violations = citation_validator.validate_citations(
        answer, docs
    )
    duration_ms = (time.time() - start_time) * 1000

    if violations:
        errors.append(f"Citation validator stripped {len(violations)} hallucinated citation(s): {violations}")

    record_node_event(
        state,
        node_name="citation_validator",
        status="PASSED" if is_valid else "WARNING_STRIPPED_HALLUCINATIONS",
        details={
            "is_valid": is_valid,
            "verified_count": len(verified_citations),
            "violation_count": len(violations),
            "violations": violations,
        },
        duration_ms=duration_ms,
    )

    return {
        **state,
        "answer": sanitized_answer,
        "citations": verified_citations,
        "errors": errors,
    }
