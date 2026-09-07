"""LangGraph node: Intent Classifier."""

from __future__ import annotations

import time
from agent.router import query_router
from agent.state import AgentState
from observability.langsmith import record_node_event


def intent_classifier_node(state: AgentState) -> AgentState:
    """Classify the user intent into 'social', 'rest_api', 'both', or 'unsupported'."""
    start_time = time.time()
    question = state.get("question", "")

    classification = query_router.classify_intent_and_sources(question)
    intent = classification["intent"]
    sources = classification["required_sources"]

    duration_ms = (time.time() - start_time) * 1000

    record_node_event(
        state,
        node_name="intent_classifier",
        status="PASSED",
        details={"intent": intent, "required_sources": sources},
        duration_ms=duration_ms,
    )

    return {
        **state,
        "intent": intent,
        "required_sources": sources,
    }
