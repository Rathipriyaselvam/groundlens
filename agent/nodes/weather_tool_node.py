"""LangGraph node: Open-Meteo Weather Tool."""

from __future__ import annotations

import time
from agent.state import AgentState
from observability.langsmith import record_node_event
from tools.weather_tool import weather_service


def weather_api_tool_node(state: AgentState) -> AgentState:
    """Execute live meteorological retrieval from Open-Meteo."""
    if "open_meteo" not in state.get("required_sources", []):
        return state

    start_time = time.time()
    question = state.get("question", "")
    existing_docs = list(state.get("retrieved_documents", []))
    api_results = list(state.get("api_results", []))
    errors = list(state.get("errors", []))

    weather_data = None
    try:
        weather_data = weather_service.get_weather(question)
    except Exception as e:
        errors.append(f"Open-Meteo weather fetch failed: {e}")

    updated_docs = list(existing_docs)
    if weather_data:
        weather_data["source_id"] = "openmeteo_01"
        updated_docs.append(weather_data)
        api_results.append(weather_data)

    duration_ms = (time.time() - start_time) * 1000

    record_node_event(
        state,
        node_name="weather_api_tool",
        status="SUCCESS" if weather_data else "EMPTY",
        details={
            "location": weather_data.get("location") if weather_data else "not_found",
            "temperature": weather_data.get("temperature") if weather_data else None,
        },
        duration_ms=duration_ms,
    )

    return {
        **state,
        "retrieved_documents": updated_docs,
        "api_results": api_results,
        "errors": errors,
    }
