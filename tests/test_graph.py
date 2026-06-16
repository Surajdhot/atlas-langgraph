"""End-to-end graph test with mocked nodes, asserting parallel retrieval.

Everything external is mocked. The key assertion is that retrieval branches
execute *concurrently* (peak concurrency > 1), proving the Send-API fan-out is
genuinely parallel rather than sequential.
"""

import asyncio
import json
from importlib import import_module
from unittest.mock import patch

import pytest

import config
from models import Evidence

# Fetch the real submodules (the package __init__ shadows these names with functions).
planner_mod = import_module("nodes.planner_node")
retrieval_mod = import_module("nodes.retrieval_node")
syn_mod = import_module("nodes.synthesis_node")

_PLANNER_RESPONSE = '["sub one", "sub two", "sub three"]'
_SYNTH_RESPONSE = json.dumps(
    {
        "claims": [
            {
                "text": "a grounded claim",
                "confidence": 0.7,
                "supporting_evidence_ids": [0],
                "conflicting_evidence_ids": [],
            }
        ]
    }
)


class _ConcurrencyTracker:
    """Tracks how many retrieval branches run at the same instant."""

    def __init__(self) -> None:
        """Initialise the live and peak concurrency counters."""
        self.current = 0
        self.peak = 0

    async def enter(self) -> None:
        """Mark a branch as entering its concurrent section."""
        self.current += 1
        self.peak = max(self.peak, self.current)
        await asyncio.sleep(0.05)
        self.current -= 1


@pytest.mark.asyncio
async def test_graph_runs_end_to_end_in_parallel(monkeypatch) -> None:
    """The graph produces a report and retrieval branches overlap in time."""
    tracker = _ConcurrencyTracker()
    monkeypatch.setattr(config, "PARALLEL_STAGGER_SECONDS", 0)

    async def fake_select(text: str) -> list[str]:
        await tracker.enter()
        return ["web"]

    async def fake_call_tool(source, sub_question) -> list[Evidence]:
        return [Evidence("web", "https://x", f"evidence for {sub_question.id}", sub_question.id)]

    with patch.object(planner_mod, "invoke_llm", return_value=_PLANNER_RESPONSE), patch.object(
        retrieval_mod, "_select_sources", fake_select
    ), patch.object(retrieval_mod, "_call_tool", fake_call_tool), patch.object(
        syn_mod, "invoke_llm", return_value=_SYNTH_RESPONSE
    ):
        from graph import build_graph

        app = build_graph()
        result = await app.ainvoke({"question": "A research question", "evidence": []})

    assert result["report"] is not None
    assert len(result["report"].claims) == 1
    # Three branches were dispatched; at least two ran concurrently.
    assert tracker.peak > 1
    assert len(result["evidence"]) == 3
