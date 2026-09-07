"""Automated tests for citation extraction, validation, and hallucination defense."""

import pytest
from grounding.citation import citation_validator


def test_fake_citation_rejected():
    """TEST 7: Fake / nonexistent citation is rejected and stripped."""
    retrieved_docs = [
        {
            "source_id": "openmeteo_01",
            "source_type": "open_meteo",
            "title": "Weather Chennai",
            "url": "https://open-meteo.com",
        }
    ]

    # Generated answer containing valid citation [openmeteo_01] and fake citation [reddit_99]
    hallucinated_answer = (
        "Chennai temperature is 31°C [openmeteo_01]. People on Reddit also like the breeze [reddit_99]."
    )

    is_valid, sanitized, verified, violations = citation_validator.validate_citations(
        hallucinated_answer, retrieved_docs
    )

    assert is_valid is False
    assert "reddit_99" in violations
    assert "[reddit_99]" not in sanitized
    assert "[openmeteo_01]" in sanitized
    assert len(verified) == 1
    assert verified[0]["source_id"] == "openmeteo_01"


def test_valid_citations_accepted():
    """Verify that legitimate citations matching retrieved evidence pass cleanly."""
    retrieved_docs = [
        {
            "source_id": "restcountries_01",
            "source_type": "rest_countries",
            "title": "India",
            "url": "https://restcountries.com/v3.1/name/india",
        },
        {
            "source_id": "reddit_01",
            "source_type": "reddit",
            "title": "Living in India",
            "url": "https://reddit.com/r/india",
        },
    ]

    answer = "India has a population of 1.4B [restcountries_01], and discussions mention vibrant food [reddit_01]."

    is_valid, sanitized, verified, violations = citation_validator.validate_citations(answer, retrieved_docs)

    assert is_valid is True
    assert len(violations) == 0
    assert len(verified) == 2
    assert {v["source_id"] for v in verified} == {"restcountries_01", "reddit_01"}
