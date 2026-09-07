"""LangGraph node: Grounded Answer Generator."""

from __future__ import annotations

import time
from typing import Any, Dict, List
from agent.prompts import GROUNDED_SYSTEM_PROMPT, GROUNDED_USER_TEMPLATE
from agent.state import AgentState
from config.logging_config import logger
from grounding.evidence import format_evidence_for_prompt
from models.llm import get_groq_llm
from observability.langsmith import record_node_event


def _fallback_grounded_synthesis(question: str, documents: List[Dict[str, Any]], partial_note: str = "") -> str:
    """Deterministic grounded synthesis directly from verified evidence when LLM is offline or in test environments."""
    if not documents:
        return "I don't have sufficient grounded information to answer that reliably."

    paragraphs: List[str] = []
    
    # Process structured REST evidence first
    rest_docs = [d for d in documents if d.get("source_type") in ("open_meteo", "rest_countries")]
    for doc in rest_docs:
        sid = doc.get("source_id", "source_01")
        if doc.get("source_type") == "open_meteo":
            loc = doc.get("location", "the requested location")
            temp = doc.get("temperature")
            hum = doc.get("humidity")
            wind = doc.get("wind_speed")
            cond = doc.get("condition", "clear")
            paragraphs.append(
                f"According to Open-Meteo [{sid}], the current weather in {loc} is {cond} with a temperature of "
                f"{temp}°C, relative humidity of {hum}%, and wind speed of {wind} km/h."
            )
        elif doc.get("source_type") == "rest_countries":
            cname = doc.get("country_name", "the requested country")
            cap = doc.get("capital", "N/A")
            pop = doc.get("population_formatted") or f"{doc.get('population', 0):,}"
            reg = doc.get("region", "")
            curr = doc.get("currencies", "N/A")
            lang = doc.get("languages", "N/A")
            paragraphs.append(
                f"REST Countries data indicates that {cname} [{sid}] (located in {reg}) has a population of "
                f"{pop} with its capital at {cap}. Official currencies include {curr}, and spoken languages include {lang}."
            )

    # Process community discussion evidence (Reddit & StackExchange)
    social_docs = [d for d in documents if d.get("source_type") in ("reddit", "stackexchange")]
    if social_docs:
        social_summaries = []
        for doc in social_docs:
            sid = doc.get("source_id", "social_01")
            stype = "Reddit community discussions" if doc.get("source_type") == "reddit" else "Stack Exchange discussions"
            title = doc.get("title", "")
            content = doc.get("content", "")
            # Short clean extract
            snippet = content[:300].replace("\n", " ")
            social_summaries.append(f"{stype} in '{title}' [{sid}] highlight: \"{snippet}...\"")
        paragraphs.append(" ".join(social_summaries))

    if partial_note:
        paragraphs.append(f"Note: {partial_note}")

    return "\n\n".join(paragraphs) if paragraphs else "I don't have sufficient grounded information to answer that reliably."


def grounded_answer_generator_node(state: AgentState) -> AgentState:
    """Synthesize an uncompromisingly grounded answer strictly from verified evidence."""
    start_time = time.time()
    question = state.get("question", "")
    grounding_status = state.get("grounding_status", "insufficient")
    documents = state.get("filtered_documents", [])
    errors = list(state.get("errors", []))

    # Strict refusal on insufficient evidence
    if grounding_status == "insufficient" or not documents:
        answer = "I don't have sufficient grounded information to answer that reliably."
        duration_ms = (time.time() - start_time) * 1000
        record_node_event(
            state,
            node_name="grounded_answer_generator",
            status="REFUSED_INSUFFICIENT",
            details={"reason": "No valid evidence to support answer."},
            duration_ms=duration_ms,
        )
        return {
            **state,
            "answer": answer,
            "grounding_status": "insufficient",
            "confidence": 0.0,
        }

    evidence_str = format_evidence_for_prompt(documents)
    llm_used = False
    answer = ""

    # Attempt generation via Open-Weights model on Groq
    try:
        llm = get_groq_llm(temperature=0.0)
        formatted_user_prompt = GROUNDED_USER_TEMPLATE.format(question=question, evidence=evidence_str)
        response = llm.invoke([
            {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
            {"role": "user", "content": formatted_user_prompt},
        ])
        answer = response.content.strip()
        llm_used = True
    except Exception as e:
        logger.warning("Groq open-weights generation skipped or unavailable: %s. Using verified grounded synthesizer.", e)
        partial_note = "Partial evidence retrieved." if grounding_status == "partial" else ""
        answer = _fallback_grounded_synthesis(question, documents, partial_note)

    duration_ms = (time.time() - start_time) * 1000
    record_node_event(
        state,
        node_name="grounded_answer_generator",
        status="SUCCESS",
        details={"llm_used": llm_used, "grounding_status": grounding_status},
        duration_ms=duration_ms,
    )

    return {
        **state,
        "answer": answer,
        "errors": errors,
    }
