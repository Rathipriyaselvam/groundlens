"""Citation management and strict post-generation verification."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple
from config.logging_config import logger


class CitationValidator:
    """Validates citations in generated answers against actual execution sources."""

    CITATION_PATTERN = re.compile(r"\[([a-zA-Z0-9_\-]+)\]")

    def extract_cited_ids(self, text: str) -> List[str]:
        """Extract all bracketed citation IDs from text (e.g., [reddit_01] -> 'reddit_01')."""
        matches = self.CITATION_PATTERN.findall(text)
        # Filter out common numeric footnote patterns that aren't source IDs
        valid_ids = []
        for m in matches:
            # Source IDs follow pattern: reddit_XX, openmeteo_XX, restcountries_XX, stackexchange_XX
            if "_" in m or m.startswith(("reddit", "openmeteo", "restcountries", "stackexchange")):
                valid_ids.append(m)
        return list(dict.fromkeys(valid_ids))  # preserve order, remove duplicates

    def validate_citations(
        self,
        answer: str,
        retrieved_documents: List[Dict[str, Any]],
    ) -> Tuple[bool, str, List[Dict[str, Any]], List[str]]:
        """Verify that all citations in the generated answer exist in current execution sources.

        Returns:
            Tuple of:
            - is_valid (bool): True if zero hallucinated or nonexistent citations exist.
            - sanitized_answer (str): Answer with any hallucinated citations stripped.
            - verified_citations (list): Full source objects for all verified citations.
            - violations (list): Log of invalid/hallucinated citation IDs detected.
        """
        cited_ids = self.extract_cited_ids(answer)
        valid_source_map = {doc.get("source_id"): doc for doc in retrieved_documents if doc.get("source_id")}

        violations: List[str] = []
        verified_citations: List[Dict[str, Any]] = []
        sanitized_answer = answer

        for cid in cited_ids:
            if cid in valid_source_map:
                doc = valid_source_map[cid]
                verified_citations.append({
                    "source_id": cid,
                    "source_type": doc.get("source_type"),
                    "title": doc.get("title"),
                    "url": doc.get("url"),
                })
            else:
                violations.append(cid)
                logger.warning("Hallucinated or nonexistent citation detected: '[%s]'. Stripping from answer.", cid)
                # Strip the fabricated citation tag
                sanitized_answer = re.sub(rf"\[{re.escape(cid)}\]", "", sanitized_answer)

        # Clean up any leftover empty brackets or double spaces
        sanitized_answer = re.sub(r"\[\s*\]", "", sanitized_answer)
        sanitized_answer = re.sub(r"\s{2,}", " ", sanitized_answer).strip()

        is_valid = (len(violations) == 0)
        return is_valid, sanitized_answer, verified_citations, violations


citation_validator = CitationValidator()
