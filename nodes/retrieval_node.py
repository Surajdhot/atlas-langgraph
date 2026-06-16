"""Retrieval node: research one sub-question, dispatched in parallel.

``dispatch_retrieval`` fans out one Send per sub-question so every branch runs
concurrently in a single LangGraph superstep. Each branch (``retrieval_node``)
lets the LLM pick which free tools fit the sub-question, calls them
concurrently, and returns evidence that the state reducer merges across all
branches.
"""

import asyncio
import logging

from langchain_core.messages import HumanMessage
from langgraph.types import Send

import config
from llm import ainvoke_llm
from models import Evidence, SubQuestion, SubQuestionStatus
from parsing import parse_json_array
from prompt_loader import load_prompt
from state import AtlasState
from tools import TOOL_REGISTRY

logger = logging.getLogger(__name__)

DEFAULT_SOURCES: tuple[str, ...] = ("web", "wikipedia")


def dispatch_retrieval(state: AtlasState) -> list[Send]:
    """Fan out one parallel retrieval branch per sub-question via the Send API.

    Args:
        state: Graph state holding the planned ``sub_questions``.

    Returns:
        A list of Send commands, one per sub-question, all targeting the
        ``retrieval`` node so they execute concurrently.
    """
    sub_questions = state.get("sub_questions", [])
    logger.info("Dispatching %d parallel retrieval branches", len(sub_questions))
    return [
        Send("retrieval", {"question": state["question"], "sub_question": sq})
        for sq in sub_questions
    ]


async def _select_sources(sub_question_text: str) -> list[str]:
    """Ask the LLM which free sources fit this sub-question, with a safe default."""
    prompt = load_prompt("retrieval", sub_question=sub_question_text)
    response = await ainvoke_llm([HumanMessage(content=prompt)])
    chosen = [s for s in parse_json_array(response) if s in TOOL_REGISTRY]
    return chosen or list(DEFAULT_SOURCES)


async def _call_tool(source: str, sub_question: SubQuestion) -> list[Evidence]:
    """Invoke one tool off the event loop, returning its evidence."""
    tool = TOOL_REGISTRY[source]
    payload = {"query": sub_question.text, "sub_question_id": sub_question.id}
    return await asyncio.to_thread(tool.invoke, payload)


async def retrieval_node(state: AtlasState) -> dict:
    """Research a single sub-question and return merged evidence for it.

    Staggers by the sub-question index so concurrent branches do not all hit
    Gemini's free-tier rate limit at the same instant; the LLM backoff handles
    any 429s that still slip through.

    Args:
        state: Send payload carrying ``question`` and one ``sub_question``.

    Returns:
        A state update with this branch's ``evidence`` (merged by the reducer).
    """
    sub_question: SubQuestion = state["sub_question"]
    index = _branch_index(sub_question.id)
    await asyncio.sleep(index * config.PARALLEL_STAGGER_SECONDS)

    sources = await _select_sources(sub_question.text)
    logger.info("Sub-question %s -> sources %s", sub_question.id, sources)
    results = await asyncio.gather(
        *(_call_tool(source, sub_question) for source in sources)
    )

    evidence: list[Evidence] = []
    for batch in results:
        evidence.extend(batch)
    sub_question.status = (
        SubQuestionStatus.RESEARCHED if evidence else SubQuestionStatus.FAILED
    )
    return {"evidence": evidence[: config.MAX_EVIDENCE_PER_QUESTION]}


def _branch_index(sub_question_id: str) -> int:
    """Derive a stagger index from a sub-question id like ``sq3`` (0 on failure)."""
    digits = "".join(ch for ch in sub_question_id if ch.isdigit())
    return int(digits) if digits else 0
