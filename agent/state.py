"""LangGraph agent state schema definition."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """Execution state tracking the complete lifecycle of a GroundLens query."""

    # Input & Normalization
    question: str
    normalized_question: str

    # Guardrails & Routing
    scope_status: str  # "in_scope", "out_of_scope", "rejected"
    intent: str  # "social", "rest_api", "both", "unsupported"
    required_sources: List[str]  # e.g., ["reddit"], ["stackexchange"], ["open_meteo"], ["rest_countries"]

    # Retrieval & Pipeline Storage
    retrieved_documents: List[Dict[str, Any]]  # Raw normalized retrieved sources
    api_results: List[Dict[str, Any]]  # Structured data from REST APIs
    filtered_documents: List[Dict[str, Any]]  # Post safety and injection checks

    # Grounding & Citations
    grounding_status: str  # "grounded", "partial", "insufficient", "refused"
    confidence: float  # Confidence score between 0.0 and 1.0
    citations: List[Dict[str, Any]]  # Verified citations attached to answer
    answer: str  # Grounded synthesized answer

    # Observability & Audit Trail
    errors: List[str]  # Non-fatal error logs & warnings
    tool_events: List[Dict[str, Any]]  # Audit log of tool execution & guardrail steps


def create_initial_state(question: str) -> AgentState:
    """Initialize a fresh AgentState dictionary for a query."""
    return {
        "question": question,
        "normalized_question": question.strip(),
        "scope_status": "pending",
        "intent": "pending",
        "required_sources": [],
        "retrieved_documents": [],
        "api_results": [],
        "filtered_documents": [],
        "grounding_status": "pending",
        "confidence": 0.0,
        "citations": [],
        "answer": "",
        "errors": [],
        "tool_events": [],
    }
