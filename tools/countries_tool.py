"""REST Countries service providing structured live & reference data with caching and fallback."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
import requests

from cache.ttl_cache import cache
from config.logging_config import logger


class CountriesService:
    """Service to retrieve verified geographic, demographic, and political data for countries."""

    LOCAL_DATA_PATH = Path(__file__).parent / "data" / "countries.json"
    WORLD_BANK_POPULATION_URL = "https://api.worldbank.org/v2/country/{code}/indicator/SP.POP.TOTL?format=json"

    def __init__(self):
        self._local_countries: Optional[List[Dict[str, Any]]] = None

    def _load_local_data(self) -> List[Dict[str, Any]]:
        """Load local fallback dataset of countries."""
        if self._local_countries is not None:
            return self._local_countries

        if self.LOCAL_DATA_PATH.exists():
            try:
                with open(self.LOCAL_DATA_PATH, "r", encoding="utf-8") as f:
                    self._local_countries = json.load(f)
                    return self._local_countries
            except Exception as e:
                logger.error("Failed to load local countries dataset: %s", e)

        self._local_countries = []
        return self._local_countries

    def extract_country_name(self, query: str) -> Optional[str]:
        """Extract country name from natural language question."""
        # 1. Possessive pattern: "India's population", "Japan's capital"
        pos_match = re.search(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)?)'s\s+(?:population|capital|currency|languages|borders|weather)\b", query)
        if pos_match:
            candidate = pos_match.group(1).strip()
            if candidate.lower() not in {"what", "the", "how", "who"}:
                return candidate

        # 2. "of [Country]" patterns
        patterns = [
            r"population\s+of\s+([a-zA-Z\s]+?)(?:\s+and|\s+today|\s+currently|\?|$)",
            r"capital\s+of\s+([a-zA-Z\s]+?)(?:\s+and|\s+today|\s+currently|\?|$)",
            r"currency\s+of\s+([a-zA-Z\s]+?)(?:\s+and|\s+today|\s+currently|\?|$)",
            r"languages\s+(?:of|spoken\s+in)\s+([a-zA-Z\s]+?)(?:\s+and|\s+today|\s+currently|\?|$)",
            r"borders\s+of\s+([a-zA-Z\s]+?)(?:\s+and|\s+today|\s+currently|\?|$)",
            r"(?:about|visiting|living\s+in)\s+([a-zA-Z\s]+?)(?:\s+and|\?|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                if candidate.lower() not in {"the", "what", "which", "how", "tell", "people", "living", "living in"}:
                    return candidate

        # 3. Direct dictionary scan for known country names in query
        countries = self._load_local_data()
        q_lower = query.lower()
        # Sort countries by length descending to match multi-word countries like "United States" first
        sorted_countries = sorted(countries, key=lambda x: len(x.get("name", "")), reverse=True)
        for c in sorted_countries:
            c_name = c.get("name", "").lower()
            if len(c_name) > 3 and re.search(rf"\b{re.escape(c_name)}\b", q_lower):
                return c.get("name")

        return None

    def _fetch_live_worldbank_population(self, alpha2: str) -> Optional[int]:
        """Optionally fetch latest census/projection population from World Bank."""
        if not alpha2 or len(alpha2) != 2:
            return None
        cache_key = f"worldbank_pop:{alpha2.upper()}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            url = self.WORLD_BANK_POPULATION_URL.format(code=alpha2.upper())
            resp = requests.get(url, timeout=4.0, headers={"User-Agent": "GroundLens/1.0"})
            if resp.status_code == 200:
                data = resp.json()
                if len(data) > 1 and isinstance(data[1], list) and len(data[1]) > 0:
                    for entry in data[1]:
                        val = entry.get("value")
                        if val is not None:
                            pop_val = int(val)
                            cache.set(cache_key, pop_val, ttl_seconds=86400 * 7)
                            return pop_val
        except Exception as e:
            logger.debug("World Bank population fetch skipped for %s: %s", alpha2, e)
        return None

    def get_country_info(self, query: str) -> Optional[Dict[str, Any]]:
        """Retrieve verified country facts matching the query.

        Returns normalized dictionary with source_id 'restcountries_01'.
        Never invents missing values.
        """
        extracted = self.extract_country_name(query)
        target = (extracted or query).strip().lower()

        cache_key = f"country:{target}"
        cached = cache.get(cache_key)
        if cached:
            return cached

        countries = self._load_local_data()
        matched: Optional[Dict[str, Any]] = None

        # 1. First pass: exact name match
        for c in countries:
            name = c.get("name", "").lower()
            if target == name:
                matched = c
                break

        # 2. Second pass: exact altSpellings match
        if not matched:
            for c in countries:
                alt = [str(a).lower() for a in c.get("altSpellings", [])]
                if target in alt:
                    matched = c
                    break

        # 3. Third pass: word boundary match
        if not matched:
            for c in countries:
                name = c.get("name", "").lower()
                if re.search(rf"\b{re.escape(target)}\b", name):
                    matched = c
                    break

        if not matched:
            logger.info("Country not found in dataset for target '%s'", target)
            return None

        country_name = matched.get("name", "")
        capital = matched.get("capital", "N/A")
        if isinstance(capital, list):
            capital = ", ".join(capital)

        population = int(matched.get("population", 0))
        alpha2 = matched.get("alpha2Code", "")

        # Try live World Bank update for latest population figures
        wb_pop = self._fetch_live_worldbank_population(alpha2)
        if wb_pop:
            population = wb_pop

        region = matched.get("region", "")
        subregion = matched.get("subregion", "")

        # Currencies formatting
        raw_curr = matched.get("currencies", [])
        curr_items = []
        if isinstance(raw_curr, list):
            for cur in raw_curr:
                c_name = cur.get("name", "")
                c_code = cur.get("code", "")
                c_sym = cur.get("symbol", "")
                sym_str = f" ({c_sym})" if c_sym else ""
                curr_items.append(f"{c_name} [{c_code}]{sym_str}")
        elif isinstance(raw_curr, dict):
            for code, details in raw_curr.items():
                c_name = details.get("name", code)
                c_sym = details.get("symbol", "")
                sym_str = f" ({c_sym})" if c_sym else ""
                curr_items.append(f"{c_name} [{code}]{sym_str}")
        currencies_str = ", ".join(curr_items) if curr_items else "N/A"

        # Languages formatting
        raw_lang = matched.get("languages", [])
        lang_items = []
        if isinstance(raw_lang, list):
            for l in raw_lang:
                lang_items.append(l.get("name", ""))
        elif isinstance(raw_lang, dict):
            lang_items = list(raw_lang.values())
        languages_str = ", ".join(filter(None, lang_items)) if lang_items else "N/A"

        # Borders
        borders = matched.get("borders", [])
        borders_str = ", ".join(borders) if borders else "None (Island nation / territory)"

        pop_formatted = f"{population:,}"
        now_iso = datetime.now(timezone.utc).isoformat()

        content = (
            f"Country: {country_name}. "
            f"Capital: {capital}. "
            f"Population: {pop_formatted}. "
            f"Region: {region} ({subregion}). "
            f"Currencies: {currencies_str}. "
            f"Languages: {languages_str}. "
            f"Borders: {borders_str}."
        )

        result: Dict[str, Any] = {
            "source_id": "restcountries_01",
            "source_type": "rest_countries",
            "title": f"REST Countries Profile for {country_name}",
            "content": content,
            "url": f"https://restcountries.com/v3.1/name/{country_name.lower().replace(' ', '%20')}",
            "country_name": country_name,
            "capital": capital,
            "population": population,
            "population_formatted": pop_formatted,
            "region": region,
            "subregion": subregion,
            "currencies": currencies_str,
            "languages": languages_str,
            "borders": borders,
            "retrieved_at": now_iso,
            "metadata": {
                "alpha2Code": alpha2,
                "alpha3Code": matched.get("alpha3Code", ""),
                "demonym": matched.get("demonym", ""),
                "area": matched.get("area"),
                "timezones": matched.get("timezones", []),
            },
        }

        cache.set(cache_key, result, ttl_seconds=86400)
        return result


# Module singleton
countries_service = CountriesService()
