"""Automated tests for external API tools and graceful degradation."""

from unittest.mock import patch
import pytest
import requests
from tools.countries_tool import countries_service
from tools.reddit_tool import reddit_service
from tools.stackexchange_tool import stackexchange_service
from tools.weather_tool import weather_service


def test_weather_service_parsing():
    """Verify Open-Meteo location extraction and weather fetch."""
    loc = weather_service.extract_location("What's the weather in Chennai right now?")
    assert loc is not None
    assert "chennai" in loc.lower()


def test_countries_service_parsing():
    """Verify country data retrieval for India."""
    res = countries_service.get_country_info("What is the population of India?")
    assert res is not None
    assert res["country_name"] == "India"
    assert res["capital"] == "New Delhi"
    assert res["source_id"] == "restcountries_01"


def test_stackexchange_html_stripping():
    """Verify that Stack Exchange service cleans HTML tags and code blocks properly."""
    raw_html = "<p>Use <code>asyncio.run()</code> to execute coroutines.</p>"
    clean = stackexchange_service._strip_html(raw_html)
    assert "<p>" not in clean
    assert "asyncio.run()" in clean


def test_api_failure_degrades_gracefully():
    """TEST 10: Simulated network failure degrades gracefully without crashing."""
    with patch("requests.get", side_effect=requests.RequestException("Connection timed out")):
        # Weather service failure
        weather_res = weather_service.get_weather("Weather in FakeCity123")
        assert weather_res is None

        # Stack Exchange failure
        se_res = stackexchange_service.search_discussions("Python asyncio")
        assert se_res == []


def test_reddit_service_graceful_on_missing_creds():
    """Verify Reddit service returns empty list gracefully when credentials are not set."""
    # Temporarily unset client
    with patch.object(reddit_service.settings, "REDDIT_CLIENT_ID", None):
        reddit_service._reddit_client = None
        res = reddit_service.search_discussions("Python")
        assert res == []
