"""Planner node: decompose a research question into focused sub-questions."""

import logging

from langchain_core.messages import HumanMessage

import config
from llm import invoke_llm
from models import SubQuestion
from parsing import parse_json_array
from prompt_loader import load_prompt
from state import AtlasState

logger = logging.getLogger(__name__)


def _build_sub_questions(texts: list[str]) -> list[SubQuestion]:
    """Turn raw sub-question strings into clamped, id'd SubQuestion objects."""
    cleaned = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
    clamped = cleaned[: config.MAX_SUB_QUESTIONS]
    return [SubQuestion(id=f"sq{i}", text=text) for i, text in enumerate(clamped)]


def planner_node(state: AtlasState) -> dict:
    """Decompose ``state['question']`` into 2-5 sub-questions via the LLM.

    Args:
        state: The graph state carrying the research question.

    Returns:
        A state update with the ``sub_questions`` list.
    """
    question = state["question"]
    prompt = load_prompt(
        "planner",
        question=question,
        min_sub_questions=config.MIN_SUB_QUESTIONS,
        max_sub_questions=config.MAX_SUB_QUESTIONS,
    )
    logger.info("Planning sub-questions for: %s", question)
    response = invoke_llm([HumanMessage(content=prompt)])
    sub_questions = _build_sub_questions(parse_json_array(response))

    if not sub_questions:
        logger.warning("Planner produced no sub-questions; using the question itself")
        sub_questions = [SubQuestion(id="sq0", text=question)]

    logger.info("Planner produced %d sub-questions", len(sub_questions))
    return {"sub_questions": sub_questions}
