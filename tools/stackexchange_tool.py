"""Stack Exchange API integration service.

Provides community-grounded technical Q&A retrieval as a compliant alternative
to unauthenticated scraping of services like Quora.
"""

from __future__ import annotations

import html
import re
from typing import Any, Dict, List, Optional
import requests

from cache.ttl_cache import cache
from config.logging_config import logger
from config.settings import get_settings


class StackExchangeService:
    """Service to search Stack Exchange questions and answers via public REST API."""

    BASE_URL = "https://api.stackexchange.com/2.3"

    def __init__(self):
        self.settings = get_settings()

    def _strip_html(self, raw_html: str) -> str:
        """Clean HTML tags and decode HTML entities."""
        clean = re.sub(r"<pre><code>.*?</code></pre>", "[code snippet]", raw_html, flags=re.DOTALL)
        clean = re.sub(r"<.*?>", " ", clean)
        clean = html.unescape(clean)
        return re.sub(r"\s+", " ", clean).strip()

    def search_discussions(
        self,
        query: str,
        site: str = "stackoverflow",
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """Search Stack Exchange for relevant questions and accepted answers.

        Args:
            query: Question keywords.
            site: Stack Exchange site (default: 'stackoverflow').
            limit: Maximum questions to retrieve.

        Returns:
            List of normalized source documents.
        """
        cache_key = f"stackexchange:{site}:{query}:{limit}"
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            logger.debug("Cache hit for StackExchange search: '%s'", query)
            return cached_result

        params: Dict[str, Any] = {
            "order": "desc",
            "sort": "relevance",
            "q": query,
            "site": site,
            "pagesize": limit,
            "filter": "!9_bDDxJY5",  # Filter returning body, answers, score, tags
        }

        if self.settings.STACKEXCHANGE_KEY:
            params["key"] = self.settings.STACKEXCHANGE_KEY

        results: List[Dict[str, Any]] = []
        try:
            resp = requests.get(
                f"{self.BASE_URL}/search/advanced",
                params=params,
                timeout=8.0,
                headers={"User-Agent": "GroundLens/1.0"},
            )

            if resp.status_code != 200:
                logger.warning("Stack Exchange API error %s: %s", resp.status_code, resp.text[:200])
                return []

            data = resp.json()
            items = data.get("items", [])

            idx = 1
            for item in items:
                title = html.unescape(item.get("title", ""))
                body_raw = item.get("body", "")
                body_clean = self._strip_html(body_raw)[:600]

                # Check for accepted answer or answers in payload
                answers_text = ""
                answers = item.get("answers", [])
                if answers:
                    top_ans = max(answers, key=lambda a: a.get("score", 0))
                    ans_body = self._strip_html(top_ans.get("body", ""))[:600]
                    is_acc = top_ans.get("is_accepted", False)
                    acc_label = " (Accepted Answer)" if is_acc else ""
                    answers_text = f"\nTop Answer{acc_label} [Score {top_ans.get('score', 0)}]: {ans_body}"

                content = f"Question: {body_clean}{answers_text}" if body_clean else title

                doc: Dict[str, Any] = {
                    "source_id": f"stackexchange_{idx:02d}",
                    "source_type": "stackexchange",
                    "title": title,
                    "content": content,
                    "url": item.get("link", f"https://stackoverflow.com/q/{item.get('question_id')}"),
                    "metadata": {
                        "site": site,
                        "score": int(item.get("score", 0)),
                        "tags": item.get("tags", []),
                        "is_answered": bool(item.get("is_answered", False)),
                        "answer_count": int(item.get("answer_count", 0)),
                    },
                }
                results.append(doc)
                idx += 1

            if results:
                cache.set(cache_key, results, ttl_seconds=900)

            return results

        except requests.RequestException as e:
            logger.error("Network error during Stack Exchange request: %s", e)
            return []
        except Exception as e:
            logger.error("Unexpected error querying Stack Exchange: %s", e)
            return []


# Module singleton
stackexchange_service = StackExchangeService()
