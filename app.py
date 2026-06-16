"""Streamlit UI for Atlas — the only frontend.

Takes a research question, runs the LangGraph pipeline, and renders the cited
report: each claim with a colour-coded confidence bar, the source list, and the
overall confidence score. Everything it depends on is a free service.
"""

import asyncio
import logging
import os

import streamlit as st

import config
from graph import run
from llm import DailyQuotaExceededError
from models import Claim, Report

logger = logging.getLogger(__name__)

GREEN, AMBER, RED = "#1a9850", "#f0ad4e", "#d73027"


def _confidence_colour(confidence: float) -> str:
    """Return a colour for a confidence score: green >0.7, amber 0.4-0.7, red <0.4."""
    if confidence > 0.7:
        return GREEN
    if confidence >= 0.4:
        return AMBER
    return RED


def _render_claim(claim: Claim) -> None:
    """Render a single claim with a colour-coded confidence bar and citations."""
    colour = _confidence_colour(claim.confidence)
    st.markdown(f"**{claim.text}**")
    st.markdown(
        f"<div style='background:{colour};width:{claim.confidence*100:.0f}%;"
        f"padding:2px 6px;border-radius:4px;color:white;font-size:0.8em'>"
        f"confidence {claim.confidence:.2f}</div>",
        unsafe_allow_html=True,
    )
    if claim.conflicting_evidence:
        st.warning(f"⚠️ Sources conflict on this claim ({len(claim.conflicting_evidence)} conflicting).")
    with st.expander(f"Supporting sources ({len(claim.supporting_evidence)})"):
        for ev in claim.supporting_evidence:
            st.markdown(f"- [{ev.source_name}]({ev.source_url})")
    st.divider()


def _render_report(report: Report) -> None:
    """Render the full report: overall score, claims, and a sources list."""
    colour = _confidence_colour(report.overall_confidence)
    st.subheader("Report")
    st.markdown(
        f"Overall confidence: "
        f"<span style='color:{colour};font-weight:bold'>"
        f"{report.overall_confidence:.2f}</span>",
        unsafe_allow_html=True,
    )
    if not report.claims:
        st.info("No evidence-backed claims were found for this question.")
        return
    for claim in report.claims:
        _render_claim(claim)
    with st.expander(f"All sources ({len(report.sources)})"):
        for ev in report.sources:
            st.markdown(f"- [{ev.source_name}]({ev.source_url})")


def _run_research(question: str) -> Report:
    """Execute the async graph run from the synchronous Streamlit context."""
    return asyncio.run(run(question))


def main() -> None:
    """Render the Atlas Streamlit application."""
    st.set_page_config(page_title="Atlas Research Engine", page_icon="🧭")
    st.title("🧭 Atlas — Multi-Agent Research Engine")
    st.caption("Planner → parallel retrieval (web · Wikipedia · arXiv) → synthesis.")

    question = st.text_input("Research question", placeholder="e.g. How does CRISPR gene editing work?")
    if st.button("Research", type="primary") and question.strip():
        _run_with_progress(question.strip())

    st.markdown("---")
    if config.tracing_enabled():
        st.caption(f"🔎 This run is traced in LangSmith (project: {os.getenv('LANGCHAIN_PROJECT', 'atlas')}).")
    st.caption("Runs entirely on free services: Google Gemini free tier, DuckDuckGo, Wikipedia, arXiv.")


def _run_with_progress(question: str) -> None:
    """Run the pipeline showing node progress, then render the report or error."""
    status = st.status("Researching…", expanded=True)
    try:
        status.write("🧠 Planning sub-questions…")
        status.write("🌐 Retrieving evidence in parallel (web · Wikipedia · arXiv)…")
        status.write("🧩 Synthesising claims and resolving conflicts…")
        report = _run_research(question)
        status.update(label="Done", state="complete")
        _render_report(report)
    except EnvironmentError as error:
        status.update(label="Configuration error", state="error")
        st.error(str(error))
    except DailyQuotaExceededError as error:
        status.update(label="Free-tier quota reached", state="error")
        st.warning(str(error))
    except (RuntimeError, ValueError) as error:
        status.update(label="Research failed", state="error")
        logger.exception("Research run failed")
        st.error(f"Research failed: {error}")


if __name__ == "__main__":
    main()
