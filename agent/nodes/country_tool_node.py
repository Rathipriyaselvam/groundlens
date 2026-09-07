"""LangGraph node: REST Countries Tool."""

from __future__ import annotations

import time
from agent.state import AgentState
from observability.langsmith import record_node_event
from tools.countries_tool import countries_service


def country_api_tool_node(state: AgentState) -> AgentState:
    """Execute live demographic/geographic data retrieval from REST Countries."""
    if "rest_countries" not in state.get("required_sources", []):
        return state

    start_time = time.time()
    question = state.get("question", "")
    existing_docs = list(state.get("retrieved_documents", []))
    api_results = list(state.get("api_results", []))
    errors = list(state.get("errors", []))

    country_data = None
    try:
        country_data = countries_service.get_country_info(question)
    except Exception as e:
        errors.append(f"REST Countries lookup failed: {e}")

    updated_docs = list(existing_docs)
    if country_data:
        country_data["source_id"] = "restcountries_01"
        updated_docs.append(country_data)
        api_results.append(country_data)

    duration_ms = (time.time() - start_time) * 1000

    record_node_event(
        state,
        node_name="country_api_tool",
        status="SUCCESS" if country_data else "EMPTY",
        details={
            "country": country_data.get("country_name") if country_data else "not_found",
            "population": country_data.get("population") if country_data else None,
        },
        duration_ms=duration_ms,
    )

    return {
        **state,
        "retrieved_documents": updated_docs,
        "api_results": api_results,
        "errors": errors,
    }
