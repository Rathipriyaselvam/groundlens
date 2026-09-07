"""User input guardrail validator executed before agent routing."""

from __future__ import annotations

import re
from typing import Dict
from config.logging_config import logger
from safety.injection_detector import injection_detector


MAX_INPUT_LENGTH = 1500


class InputGuardrail:
    """Pre-execution validation of user questions."""

    def validate_input(self, question: str) -> Dict[str, Any]:
        """Verify that user question is well-formed, safe, and within length limits.

        Returns:
            Dict containing:
            - allowed (bool): Whether input passes validation.
            - reason (str): Explanation if rejected, 'valid' otherwise.
        """
        if not question or not question.strip():
            return {
                "allowed": False,
                "reason": "empty_input",
                "message": "The input question cannot be empty.",
            }

        cleaned = question.strip()

        if len(cleaned) > MAX_INPUT_LENGTH:
            return {
                "allowed": False,
                "reason": "input_too_long",
                "message": f"Input exceeds maximum allowed length ({MAX_INPUT_LENGTH} characters). Please shorten your query.",
            }

        # Check prompt injection patterns in user query
        is_inj, risk, patterns = injection_detector.inspect_text(cleaned)
        if is_inj and risk >= 0.7:
            logger.warning("Adversarial prompt injection attempt in user query: %s (patterns=%s)", cleaned[:80], patterns)
            return {
                "allowed": False,
                "reason": "prompt_injection_detected",
                "message": "The query contains prohibited instructional override or prompt extraction patterns.",
            }

        return {
            "allowed": True,
            "reason": "valid",
            "message": "Input passed guardrail verification.",
        }


input_guardrail = InputGuardrail()
