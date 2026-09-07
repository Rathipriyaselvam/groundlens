"""Evidence validator evaluating sufficiency, coverage, and confidence levels."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from config.logging_config import logger


class EvidenceValidator:
    """Validates retrieved evidence against required sources and intent."""

    def evaluate_evidence(
        self,
        documents: List[Dict[str, Any]],
        intent: str,
        required_sources: List[str],
    ) -> Tuple[str, float, Optional[str]]:
        """Determine grounding status and confidence based on retrieved documents.

        Returns:
            Tuple of (grounding_status, confidence, reason_note)
            where grounding_status is one of:
            - 'insufficient' (0 documents or critical failure)
            - 'partial' (subset of required sources found)
            - 'grounded' (sufficient evidence for complete response)
        """
        if not documents:
            logger.info("Zero evidence documents available. Setting grounding_status to 'insufficient'.")
            return "insufficient", 0.0, "No grounded evidence could be retrieved from live sources."

        # Check coverage across required sources
        found_types = {doc.get("source_type") for doc in documents if doc.get("source_type")}
        req_types = set(required_sources)

        missing = req_types - found_types

        if missing and found_types:
            # Partial grounding
            logger.info("Partial evidence: retrieved %s, missing required sources: %s", found_types, missing)
            missing_str = ", ".join(missing)
            return (
                "partial",
                0.55,
                f"Partial evidence available from {', '.join(found_types)}. Missing sources: {missing_str}.",
            )

        if not found_types and documents:
            return "insufficient", 0.1, "Retrieved documents contained no identifiable source types."

        # Complete coverage
        logger.debug("Sufficient evidence retrieved for sources: %s", found_types)
        return "grounded", 0.95, f"Complete evidence retrieved across all target sources ({', '.join(found_types)})."


evidence_validator = EvidenceValidator()
