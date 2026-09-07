"""Multi-layered prompt injection detection and isolation for untrusted retrieved content."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple
from config.logging_config import logger


INJECTION_PATTERNS = [
    # Direct instruction overrides
    r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|prompts|commands|rules)",
    r"disregard\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|prompts|guidelines)",
    r"forget\s+(?:all\s+)?(?:previous|prior|above)\s+instructions",
    r"override\s+(?:system|instructions|prompt|rules)",
    
    # System prompt extraction
    r"reveal\s+(?:your\s+)?(?:system\s+prompt|instructions|initial\s+prompt|system\s+message)",
    r"(?:what\s+are\s+your|show\s+me\s+your|print\s+your)\s+instructions",
    r"repeat\s+(?:the\s+)?words\s+above",
    
    # Role hijacking & Jailbreaks
    r"you\s+are\s+now\s+(?:in\s+)?(?:developer|dan|god|unrestricted|jailbreak)\s+mode",
    r"act\s+as\s+(?:an?\s+)?unrestricted\s+ai",
    r"bypass\s+(?:all\s+)?(?:safety|content|policy)\s+filters",
    r"jailbreak",
    
    # Tool / Command execution triggers
    r"execute\s+(?:this\s+)?(?:command|code|script|sql)",
    r"call\s+(?:this\s+)?(?:tool|function|api)\s+with",
    r"<script>.*?</script>",
    r"\[system\]\s*:",
    r"system\s*:\s*you\s+must",
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


class PromptInjectionDetector:
    """Detects and isolates prompt injection attempts within untrusted external text."""

    def inspect_text(self, text: str) -> Tuple[bool, float, List[str]]:
        """Inspect a block of text for prompt injection patterns.

        Returns:
            Tuple of:
            - is_injected (bool): True if injection risk exceeds threshold.
            - risk_score (float): Calculated risk between 0.0 and 1.0.
            - detected_patterns (list): Patterns matched.
        """
        if not text:
            return False, 0.0, []

        matches = []
        for pattern in COMPILED_PATTERNS:
            found = pattern.findall(text)
            if found:
                matches.extend(found if isinstance(found[0], str) else [m[0] for m in found])

        if matches:
            risk = min(1.0, 0.5 + (len(matches) * 0.25))
            return True, risk, matches

        # Heuristic: Check for instruction-like directive density in retrieved social post
        directive_words = ["must", "shall", "always", "never", "instruction", "system", "command", "assistant"]
        q_lower = text.lower()
        score = sum(1 for w in directive_words if f" {w} " in q_lower)
        if score >= 5 and ("prompt" in q_lower or "system" in q_lower):
            return True, 0.7, ["heuristic_directive_density"]

        return False, 0.0, []

    def filter_documents(
        self,
        documents: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Filter a list of retrieved documents, removing or neutralizing injected sources.

        Returns:
            Tuple of:
            - safe_documents (list): Documents passing injection verification.
            - blocked_documents (list): Documents identified as containing injection attacks.
        """
        safe_docs: List[Dict[str, Any]] = []
        blocked_docs: List[Dict[str, Any]] = []

        for doc in documents:
            content = f"{doc.get('title', '')} {doc.get('content', '')}"
            is_inj, risk, matches = self.inspect_text(content)

            if is_inj:
                logger.warning(
                    "Prompt injection detected in source '%s' (risk=%.2f, matches=%s). Quarantining document.",
                    doc.get("source_id"),
                    risk,
                    matches,
                )
                blocked = dict(doc)
                blocked["injection_risk"] = risk
                blocked["injection_matches"] = matches
                blocked_docs.append(blocked)
            else:
                safe_docs.append(doc)

        return safe_docs, blocked_docs


injection_detector = PromptInjectionDetector()
