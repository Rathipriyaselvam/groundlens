"""Reddit integration service using PRAW with rate-limiting, TTL caching, and graceful fallback."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import praw
from prawcore.exceptions import PrawcoreException, ResponseException

from cache.ttl_cache import cache
from config.logging_config import logger
from config.settings import get_settings


class RedditService:
    """Service to query Reddit discussions via PRAW with structured normalization and caching."""

    def __init__(self):
        self.settings = get_settings()
        self._reddit_client: Optional[praw.Reddit] = None

    def _get_client(self) -> Optional[praw.Reddit]:
        """Lazy initialization of PRAW Reddit client."""
        if self._reddit_client is not None:
            return self._reddit_client

        if not self.settings.has_reddit_creds:
            logger.info("Reddit API credentials not configured. Reddit retrieval will be disabled or skipped.")
            return None

        try:
            self._reddit_client = praw.Reddit(
                client_id=self.settings.REDDIT_CLIENT_ID,
                client_secret=self.settings.REDDIT_CLIENT_SECRET,
                user_agent=self.settings.REDDIT_USER_AGENT,
                request_timeout=8.0,
            )
            # Read-only mode
            self._reddit_client.read_only = True
            return self._reddit_client
        except Exception as e:
            logger.error("Failed to initialize PRAW Reddit client: %s", e)
            return None

    def search_discussions(
        self,
        query: str,
        subreddits: Optional[List[str]] = None,
        limit: int = 3,
        max_comments: int = 2,
    ) -> List[Dict[str, Any]]:
        """Search Reddit for posts and top comments matching query.

        Args:
            query: Search query terms.
            subreddits: Optional list of subreddits to restrict search (e.g. ['programming', 'cscareerquestions']).
            limit: Maximum number of posts to retrieve.
            max_comments: Top comments to fetch per post.

        Returns:
            List of normalized source documents.
        """
        cache_key = f"reddit:{query}:{','.join(subreddits or [])}:{limit}:{max_comments}"
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            logger.debug("Cache hit for Reddit search: '%s'", query)
            return cached_result

        client = self._get_client()
        if client is None:
            logger.warning("Reddit retrieval unavailable: Missing credentials.")
            return []

        results: List[Dict[str, Any]] = []
        try:
            target_sub = "+".join(subreddits) if subreddits else "all"
            subreddit = client.subreddit(target_sub)
            submissions = subreddit.search(query, sort="relevance", limit=limit, time_filter="year")

            idx = 1
            for sub in submissions:
                try:
                    # Fetch top comments
                    sub.comment_sort = "top"
                    sub.comments.replace_more(limit=0)
                    top_comments: List[str] = []
                    for c in sub.comments[:max_comments]:
                        if hasattr(c, "body") and c.body:
                            body_snippet = c.body.strip()[:400]
                            top_comments.append(f"[Score {getattr(c, 'score', 0)}] {body_snippet}")

                    content_parts = []
                    if sub.selftext:
                        content_parts.append(sub.selftext.strip()[:800])
                    if top_comments:
                        content_parts.append("Top comments: " + " | ".join(top_comments))

                    full_content = "\n".join(content_parts) if content_parts else sub.title

                    doc: Dict[str, Any] = {
                        "source_id": f"reddit_{idx:02d}",
                        "source_type": "reddit",
                        "title": sub.title,
                        "content": full_content,
                        "url": f"https://reddit.com{sub.permalink}",
                        "metadata": {
                            "subreddit": sub.subreddit.display_name,
                            "score": int(sub.score),
                            "num_comments": int(sub.num_comments),
                            "created_utc": float(sub.created_utc),
                        },
                    }
                    results.append(doc)
                    idx += 1
                except Exception as post_err:
                    logger.warning("Error parsing Reddit submission %s: %s", getattr(sub, "id", "unknown"), post_err)
                    continue

            # Cache results for 10 minutes
            if results:
                cache.set(cache_key, results, ttl_seconds=600)

            return results

        except (PrawcoreException, ResponseException) as e:
            logger.error("Reddit API error during search '%s': %s", query, e)
            return []
        except Exception as e:
            logger.error("Unexpected error during Reddit search '%s': %s", query, e)
            return []


# Module singleton
reddit_service = RedditService()
