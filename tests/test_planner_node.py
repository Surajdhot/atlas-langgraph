"""Tests for the planner node — LLM is mocked, no network calls."""

from importlib import import_module
from unittest.mock import patch

from nodes.planner_node import planner_node

# Fetch the real submodule (the package __init__ shadows the name with the function).
planner_mod = import_module("nodes.planner_node")


def test_planner_returns_two_to_five_subquestions() -> None:
    """A normal LLM response yields between 2 and 5 non-empty sub-questions."""
    fake = '["What is X?", "How does X work?", "What are the risks of X?"]'
    with patch.object(planner_mod, "invoke_llm", return_value=fake):
        result = planner_node({"question": "Tell me about X"})

    subs = result["sub_questions"]
    assert 2 <= len(subs) <= 5
    assert all(sq.text.strip() for sq in subs)
    assert len({sq.id for sq in subs}) == len(subs)


def test_planner_clamps_to_max() -> None:
    """More than MAX_SUB_QUESTIONS proposals are clamped to the maximum."""
    many = "[" + ", ".join(f'"q{i}"' for i in range(8)) + "]"
    with patch.object(planner_mod, "invoke_llm", return_value=many):
        result = planner_node({"question": "Q"})

    assert len(result["sub_questions"]) == planner_mod.config.MAX_SUB_QUESTIONS


def test_planner_falls_back_when_no_json() -> None:
    """Unparseable output falls back to researching the original question."""
    with patch.object(planner_mod, "invoke_llm", return_value="not json at all"):
        result = planner_node({"question": "The original question"})

    subs = result["sub_questions"]
    assert len(subs) == 1
    assert subs[0].text == "The original question"
