"""Content safety filter evaluating hate speech, harassment, explicit, and dangerous material."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple
from config.logging_config import logger


# Content safety patterns
UNSAFE_CATEGORIES = {
    "hate_speech": [
        r"\b(?:kill\s+all|exterminate|subhuman|hate\s+all)\b",
        r"\b(?:slurs?|white\s+supremacist|neo-nazi)\b",
    ],
    "harassment_abuse": [
        r"\b(?:doxx|doxxing|harass\s+them|threaten\s+to\s+kill|kys|die\s+in\s+a\s+fire)\b",
    ],
    "explicit_sexual": [
        r"\b(?:nsfw|pornography|hardcore\s+porn|explicit\s+sex|erotic\s+roleplay)\b",
    ],
    "dangerous_content": [
        r"\b(?:how\s+to\s+build\s+a\s+bomb|make\s+napalm|synthesize\s+sarin|poison\s+water\s+supply)\b",
        r"\b(?:malware\s+code|ransomware\s+payload|keylogger\s+script|ddos\s+attack\s+script)\b",
        r"\b(?:suicide\s+method|how\s+to\s+hang\s+yourself|cut\s+wrists)\b",
    ],
}


class ContentSafetyFilter:
    """Evaluates external retrieved documents for unsafe, toxic, or hazardous content."""

    def __init__(self):
        self._compiled_patterns = {
            category: [re.compile(p, re.IGNORECASE) for p in patterns]
            for category, patterns in UNSAFE_CATEGORIES.items()
        }

    def inspect_content(self, text: str) -> Tuple[bool, List[str]]:
        """Check text against unsafe content categories.

        Returns:
            Tuple of (is_unsafe: bool, flagged_categories: list[str])
        """
        if not text:
            return False, []

        flagged: List[str] = []
        for cat, compiled_list in self._compiled_patterns.items():
            for pat in compiled_list:
                if pat.search(text):
                    flagged.append(cat)
                    break

        return (len(flagged) > 0), flagged

    def filter_documents(
        self,
        documents: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Filter out unsafe documents while preserving safe sources.

        Returns:
            Tuple of (safe_documents, filtered_documents)
        """
        safe_docs: List[Dict[str, Any]] = []
        filtered_docs: List[Dict[str, Any]] = []

        for doc in documents:
            combined_text = f"{doc.get('title', '')} {doc.get('content', '')}"
            is_unsafe, categories = self.inspect_content(combined_text)

            if is_unsafe:
                logger.warning(
                    "Content safety violation in document '%s' (categories: %s). Excluding from evidence.",
                    doc.get("source_id"),
                    categories,
                )
                flagged_doc = dict(doc)
                flagged_doc["safety_violations"] = categories
                filtered_docs.append(flagged_doc)
            else:
                safe_docs.append(doc)

        return safe_docs, filtered_docs


content_filter = ContentSafetyFilter()
