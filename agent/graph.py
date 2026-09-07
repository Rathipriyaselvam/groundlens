"""LangGraph StateGraph compilation linking all 14 nodes with conditional edges."""

from __future__ import annotations

from typing import Dict, Literal
from langgraph.graph import END, START, StateGraph

from agent.nodes.answer_generator import grounded_answer_generator_node
from agent.nodes.citation_validator import citation_validator_node
from agent.nodes.content_safety_filter import content_safety_filter_node
from agent.nodes.country_tool_node import country_api_tool_node
from agent.nodes.evidence_validator import evidence_validator_node
from agent.nodes.final_response import final_response_node
from agent.nodes.input_guardrail import input_guardrail_node
from agent.nodes.intent_classifier import intent_classifier_node
from agent.nodes.prompt_injection_detector import prompt_injection_detector_node
from agent.nodes.reddit_retriever import reddit_retriever_node
from agent.nodes.scope_checker import scope_checker_node
from agent.nodes.source_router import source_router_node
from agent.nodes.stackexchange_retriever import stackexchange_retriever_node
from agent.nodes.weather_tool_node import weather_api_tool_node
from agent.state import AgentState, create_initial_state
from observability.langsmith import configure_langsmith


def route_after_input_guardrail(state: AgentState) -> Literal["final_response", "scope_checker"]:
    """Branch to final response if input guardrail rejects the query."""
    if state.get("scope_status") == "rejected":
        return "final_response"
    return "scope_checker"


def route_after_scope_checker(state: AgentState) -> Literal["final_response", "intent_classifier"]:
    """Branch to final response if query is out of domain scope."""
    if state.get("scope_status") == "out_of_scope":
        return "final_response"
    return "intent_classifier"


def route_after_source_router(state: AgentState) -> Literal["final_response", "weather_api_tool"]:
    """Branch directly to final response if intent is unsupported or no sources required."""
    if state.get("intent") == "unsupported" or not state.get("required_sources"):
        return "final_response"
    return "weather_api_tool"


def route_after_evidence_validator(state: AgentState) -> Literal["final_response", "grounded_answer_generator"]:
    """Branch to final response if evidence is insufficient."""
    if state.get("grounding_status") == "insufficient":
        return "final_response"
    return "grounded_answer_generator"


def build_graph():
    """Compile the 14-node GroundLens LangGraph execution pipeline."""
    configure_langsmith()

    workflow = StateGraph(AgentState)

    # 1. Register all 14 nodes
    workflow.add_node("input_guardrail", input_guardrail_node)
    workflow.add_node("scope_checker", scope_checker_node)
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("source_router", source_router_node)
    workflow.add_node("weather_api_tool", weather_api_tool_node)
    workflow.add_node("country_api_tool", country_api_tool_node)
    workflow.add_node("reddit_retriever", reddit_retriever_node)
    workflow.add_node("stackexchange_retriever", stackexchange_retriever_node)
    workflow.add_node("content_safety_filter", content_safety_filter_node)
    workflow.add_node("prompt_injection_detector", prompt_injection_detector_node)
    workflow.add_node("evidence_validator", evidence_validator_node)
    workflow.add_node("grounded_answer_generator", grounded_answer_generator_node)
    workflow.add_node("citation_validator", citation_validator_node)
    workflow.add_node("final_response", final_response_node)

    # 2. Add edges and conditional routing
    workflow.add_edge(START, "input_guardrail")

    workflow.add_conditional_edges(
        "input_guardrail",
        route_after_input_guardrail,
        {
            "final_response": "final_response",
            "scope_checker": "scope_checker",
        },
    )

    workflow.add_conditional_edges(
        "scope_checker",
        route_after_scope_checker,
        {
            "final_response": "final_response",
            "intent_classifier": "intent_classifier",
        },
    )

    workflow.add_edge("intent_classifier", "source_router")

    workflow.add_conditional_edges(
        "source_router",
        route_after_source_router,
        {
            "final_response": "final_response",
            "weather_api_tool": "weather_api_tool",
        },
    )

    # Sequential pipeline across active tool nodes (each checks state["required_sources"])
    workflow.add_edge("weather_api_tool", "country_api_tool")
    workflow.add_edge("country_api_tool", "reddit_retriever")
    workflow.add_edge("reddit_retriever", "stackexchange_retriever")
    workflow.add_edge("stackexchange_retriever", "content_safety_filter")

    # Safety and Validation pipeline
    workflow.add_edge("content_safety_filter", "prompt_injection_detector")
    workflow.add_edge("prompt_injection_detector", "evidence_validator")

    workflow.add_conditional_edges(
        "evidence_validator",
        route_after_evidence_validator,
        {
            "final_response": "final_response",
            "grounded_answer_generator": "grounded_answer_generator",
        },
    )

    workflow.add_edge("grounded_answer_generator", "citation_validator")
    workflow.add_edge("citation_validator", "final_response")
    workflow.add_edge("final_response", END)

    return workflow.compile()


# Compiled agent runnable
app_graph = build_graph()


def run_agent(question: str) -> AgentState:
    """Convenience helper to run a query from input to final state."""
    initial_state = create_initial_state(question)
    final_state = app_graph.invoke(initial_state)
    return final_state
