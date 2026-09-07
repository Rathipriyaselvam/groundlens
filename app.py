"""GroundLens — Grounded Research & Reality Check Agent
Streamlit web application interface.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List
import streamlit as st

from agent.graph import run_agent
from cache.ttl_cache import cache
from config.settings import get_settings


# Page configuration
st.set_page_config(
    page_title="GroundLens — Grounded Research Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .status-badge-grounded {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-badge-partial {
        background-color: #FEF08A;
        color: #854D0E;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-badge-insufficient {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .source-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .metric-box {
        background-color: #F1F5F9;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def render_sidebar():
    """Render application metadata, status indicators, and supported sources."""
    settings = get_settings()

    with st.sidebar:
        st.title("🔍 GroundLens")
        st.caption("Grounded Reality-Check & Research Agent")
        st.markdown("---")

        st.subheader("ℹ️ About")
        st.markdown(
            "**GroundLens** is an evidence-first research agent. It intelligently retrieves live structured "
            "data and community discussions, strictly refusing to answer factual claims from LLM memory "
            "when evidence is missing."
        )

        st.markdown("---")
        st.subheader("🌐 Supported Sources")
        st.markdown("""
        - **Reddit** (PRAW): Community sentiment, discussions, complaints, recommendations.
        - **Stack Exchange API**: Technical discussions, coding solutions (compliant Quora alternative).
        - **Open-Meteo**: Live weather, temperature, humidity, wind, and conditions.
        - **REST Countries**: Official capitals, population, currencies, borders, languages.
        """)

        st.markdown("---")
        st.subheader("🎯 Supported Topics")
        st.markdown("""
        - Technology, software, coding, products
        - Geography, countries, world demographics
        - Weather, climate, meteorological data
        - Public community sentiment & consumer experiences
        """)

        st.markdown("---")
        st.subheader("⚙️ System Status")

        # Open-weights model status
        groq_status = "🟢 Active" if settings.has_groq_key else "🟡 Offline / Synthesizer Fallback"
        st.markdown(f"**Reasoning Model:** `{settings.GROQ_MODEL}`")
        st.caption(f"Groq API: {groq_status}")

        # Reddit status
        reddit_status = "🟢 Connected" if settings.has_reddit_creds else "⚪ Skipped (No credentials)"
        st.caption(f"Reddit API (PRAW): {reddit_status}")

        # Stack Exchange status
        st.caption("Stack Exchange API: 🟢 Public REST Active")

        # REST APIs
        st.caption("Open-Meteo Weather: 🟢 Live Public API")
        st.caption("REST Countries: 🟢 Live & Local Dataset")

        # LangSmith
        langsmith_status = "🟢 Tracing Active" if settings.has_langsmith_key else "⚪ Inactive"
        st.caption(f"LangSmith Tracing: {langsmith_status}")

        # Cache stats
        cache_stats = cache.stats()
        active_keys = cache_stats.get("active_keys", 0)
        st.caption(f"SQLite TTL Cache: 🟢 {active_keys} cached items")


def display_results(result: Dict[str, Any]):
    """Render grounded answer, badges, sources, and agent trace."""
    answer = result.get("answer", "")
    grounding_status = result.get("grounding_status", "pending")
    confidence = result.get("confidence", 0.0)
    citations = result.get("citations", [])
    errors = result.get("errors", [])
    tool_events = result.get("tool_events", [])
    filtered_docs = result.get("filtered_documents", [])
    intent = result.get("intent", "unsupported")

    st.markdown("### 💡 Grounded Answer")
    st.markdown(answer)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Grounding Status**")
        if grounding_status == "grounded":
            st.markdown('<span class="status-badge-grounded">✓ Fully Grounded</span>', unsafe_allow_html=True)
        elif grounding_status == "partial":
            st.markdown('<span class="status-badge-partial">⚠️ Partial Grounding</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge-insufficient">⛔ Insufficient Grounding</span>', unsafe_allow_html=True)

    with col2:
        st.markdown("**Confidence Score**")
        st.progress(min(1.0, max(0.0, float(confidence))))
        st.caption(f"Confidence: **{int(confidence * 100)}%**")

    with col3:
        st.markdown("**Intent Category**")
        st.markdown(f"🎯 **`{intent.upper()}`**")

    # Sources used
    st.markdown("#### 📚 Sources Used")
    if citations:
        for c in citations:
            sid = c.get("source_id", "source")
            stype = c.get("source_type", "unknown").replace("_", " ").title()
            title = c.get("title", "Reference Source")
            url = c.get("url", "#")
            st.markdown(
                f'<div class="source-card">'
                f'<strong>[{sid}] {stype}:</strong> <a href="{url}" target="_blank">{title}</a>'
                f'</div>',
                unsafe_allow_html=True,
            )
    elif filtered_docs:
        for doc in filtered_docs:
            sid = doc.get("source_id", "source")
            stype = doc.get("source_type", "unknown").replace("_", " ").title()
            title = doc.get("title", "Reference Source")
            url = doc.get("url", "#")
            st.markdown(
                f'<div class="source-card">'
                f'<strong>[{sid}] {stype}:</strong> <a href="{url}" target="_blank">{title}</a>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("No sources cited or required for this response.")

    # Warnings if any
    if errors:
        st.markdown("#### ⚠️ System Warnings & Security Logs")
        for err in errors:
            st.warning(err)

    # Agent Trace Expandable Panel
    with st.expander("🔬 Agent Trace (Observability Audit Trail)", expanded=False):
        st.caption("Step-by-step trace of agent execution nodes. (LangSmith captures comprehensive traces when configured).")
        if tool_events:
            for idx, event in enumerate(tool_events, start=1):
                node = event.get("node", "unknown")
                status = event.get("status", "DONE")
                dur = event.get("duration_ms", 0.0)
                details = event.get("details", {})
                st.markdown(f"**{idx}. `{node}`** — Status: `{status}` ({dur:.1f} ms)")
                if details:
                    st.json(details, expanded=False)
        else:
            st.write("No trace events recorded.")


def main():
    """Main Streamlit execution function."""
    render_sidebar()

    st.markdown('<div class="main-header">GroundLens</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Grounded answers from live community discussions and public data.</div>',
        unsafe_allow_html=True,
    )

    # Starter question buttons
    st.markdown("**Suggested Questions:**")
    starters = [
        "What's the weather in Chennai right now?",
        "What is the population of India?",
        "What are developers saying about remote software jobs?",
        "What is the population of Japan and what do travelers say about visiting Japan?",
        "Write me a poem about my girlfriend.",
    ]

    cols = st.columns(len(starters))
    clicked_query = None
    for idx, starter in enumerate(starters):
        with cols[idx]:
            # Clean short label
            label = starter if len(starter) <= 30 else starter[:27] + "..."
            if st.button(label, key=f"starter_{idx}", help=starter):
                clicked_query = starter

    # Input box
    prompt = st.chat_input("Ask a grounded question (e.g. weather, countries, community sentiment, programming)...")

    query_to_run = clicked_query or prompt

    if query_to_run:
        st.markdown(f"**Query:** *{query_to_run}*")
        with st.spinner("Executing GroundLens 14-node LangGraph pipeline..."):
            result = run_agent(query_to_run)
            display_results(result)


if __name__ == "__main__":
    main()
