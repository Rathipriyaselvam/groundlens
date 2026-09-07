"""Intent classification, scope checking, and source routing engine."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple
from config.logging_config import logger


# Domain keywords matching supported scope
SUPPORTED_SCOPE_KEYWORDS = {
    # Tech & Software
    "python", "javascript", "react", "golang", "java", "rust", "c++", "code", "coding",
    "programming", "software", "developer", "engineer", "framework", "library", "api",
    "database", "sql", "git", "github", "docker", "kubernetes", "bug", "error", "job", "career",
    "remote", "salary", "interview", "hiring", "tech", "ai", "machine learning",
    # Weather & Atmosphere
    "weather", "temperature", "forecast", "climate", "rain", "humidity", "wind", "storm",
    "sunny", "cloudy", "celsius", "fahrenheit", "snow", "degree",
    # Countries & Geography
    "country", "countries", "population", "capital", "currency", "currencies", "language",
    "languages", "border", "borders", "region", "geography", "demographics",
    # Community discussions & Consumer experiences
    "think", "saying", "opinion", "review", "experience", "complaint", "recommend",
    "recommendation", "reddit", "community", "sentiment", "discuss", "discussion", "people say",
    "users say", "travel", "visiting", "living in", "cost of living",
}

UNSUPPORTED_PATTERNS = [
    r"\b(?:write|compose)\s+(?:me\s+)?(?:a\s+)?(?:poem|poetry|song|rhyme|haiku|story|novel)\b",
    r"\b(?:medical|clinical|diagnosis|prescribe|prescription)\b",
    r"\b(?:legal\s+advice|sue|lawsuit|attorney)\b",
    r"\b(?:who\s+will\s+win\s+(?:the\s+)?(?:next\s+)?election)\b",
    r"\b(?:lottery|astrology|horoscope|fortune|predict\s+the\s+future)\b",
    r"\b(?:girlfriend|boyfriend|love\s+letter|flirt)\b",
]

WEATHER_KEYWORDS = [
    "weather", "temperature", "forecast", "humidity", "wind speed", "rain in",
    "hot in", "cold in", "celsius in", "fahrenheit in", "degree in",
]

COUNTRY_KEYWORDS = [
    "population", "capital of", "capital city", "currency of", "currencies of",
    "official language", "languages spoken", "borders of", "neighboring countries",
    "subregion", "population of",
]

SOCIAL_KEYWORDS = [
    "what are people saying", "what do people think", "what are developers saying",
    "what are users saying", "what do users say", "experiences", "complaints",
    "recommendations", "community sentiment", "on reddit", "reddit", "stack overflow",
    "stackexchange", "pros and cons", "is it worth", "opinions on", "review of",
    "living in", "visiting", "travelers say",
]


class QueryRouter:
    """Intelligently classifies user queries into scope, intent, and target sources."""

    def check_scope(self, query: str) -> Tuple[bool, str]:
        """Check if query belongs to supported domains."""
        q_lower = query.lower().strip()

        # Check explicit unsupported patterns first
        for pat in UNSUPPORTED_PATTERNS:
            if re.search(pat, q_lower):
                return False, "Query matches unsupported topic (creative writing, personal/medical advice, or speculative future events)."

        # Tokenize and check overlap with supported scope keywords
        words = set(re.findall(r"\b[a-zA-Z0-9_\-\+\#\.]+\b", q_lower))
        has_supported_topic = any(k in q_lower for k in SUPPORTED_SCOPE_KEYWORDS) or bool(words.intersection(SUPPORTED_SCOPE_KEYWORDS))

        if not has_supported_topic:
            # Questions with general conversational fluff or out of scope
            return False, "Query is outside the supported domains (tech/software, weather, country data, or community experiences)."

        return True, "in_scope"

    def classify_intent_and_sources(self, query: str) -> Dict[str, Any]:
        """Classify question into intent ('social', 'rest_api', 'both', 'unsupported') and sources."""
        q_lower = query.lower()

        needs_weather = any(wk in q_lower for wk in WEATHER_KEYWORDS)
        needs_country = any(ck in q_lower for ck in COUNTRY_KEYWORDS)

        # Detect social intent
        needs_social = any(sk in q_lower for sk in SOCIAL_KEYWORDS)

        # Questions asking how-to code or technical issues without live REST data
        tech_indicators = ["error", "exception", "bug", "how to", "why does", "difference between", "traceback"]
        needs_tech_qa = any(ti in q_lower for ti in tech_indicators)

        sources: List[str] = []

        if needs_weather:
            sources.append("open_meteo")
        if needs_country:
            sources.append("rest_countries")

        if needs_social:
            sources.append("reddit")
        elif needs_tech_qa and not needs_weather and not needs_country:
            sources.append("stackexchange")

        # Determine overall intent category
        has_rest = "open_meteo" in sources or "rest_countries" in sources
        has_social = "reddit" in sources or "stackexchange" in sources

        if has_rest and has_social:
            intent = "both"
        elif has_rest:
            intent = "rest_api"
        elif has_social:
            intent = "social"
        else:
            # If query mentions a country and asks general questions
            if any(w in q_lower for w in ["japan", "india", "germany", "france", "usa", "canada", "brazil"]):
                sources.append("rest_countries")
                intent = "rest_api"
            elif any(w in q_lower for w in ["python", "javascript", "react", "programming", "software"]):
                sources.append("reddit")
                intent = "social"
            else:
                intent = "unsupported"

        return {
            "intent": intent,
            "required_sources": sources,
        }


# Global singleton router
query_router = QueryRouter()
