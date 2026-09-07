"""Automated tests for query routing, scope checking, and intent classification."""

import pytest
from agent.router import query_router


def test_weather_routes_to_open_meteo():
    """TEST 1: Weather question routes to Open-Meteo."""
    query = "What's the weather in Chennai right now?"
    in_scope, _ = query_router.check_scope(query)
    assert in_scope is True

    classification = query_router.classify_intent_and_sources(query)
    assert classification["intent"] == "rest_api"
    assert "open_meteo" in classification["required_sources"]


def test_country_population_routes_to_rest_countries():
    """TEST 2: Country population question routes to REST Countries."""
    query = "What is the population of India?"
    in_scope, _ = query_router.check_scope(query)
    assert in_scope is True

    classification = query_router.classify_intent_and_sources(query)
    assert classification["intent"] == "rest_api"
    assert "rest_countries" in classification["required_sources"]


def test_opinion_question_routes_to_reddit():
    """TEST 3: Opinion question routes to Reddit."""
    query = "What are developers saying about remote software jobs?"
    in_scope, _ = query_router.check_scope(query)
    assert in_scope is True

    classification = query_router.classify_intent_and_sources(query)
    assert classification["intent"] == "social"
    assert "reddit" in classification["required_sources"]


def test_combined_question_calls_both_categories():
    """TEST 4: Combined question calls both source categories (REST + Social)."""
    query = "What is India's population and what do people on Reddit say about living in India?"
    in_scope, _ = query_router.check_scope(query)
    assert in_scope is True

    classification = query_router.classify_intent_and_sources(query)
    assert classification["intent"] == "both"
    assert "rest_countries" in classification["required_sources"]
    assert "reddit" in classification["required_sources"]


def test_unsupported_question_is_refused():
    """TEST 5: Unsupported question receives out-of-scope refusal."""
    query = "Write me a poem about my girlfriend."
    in_scope, reason = query_router.check_scope(query)
    assert in_scope is False
    assert "unsupported" in reason.lower()
