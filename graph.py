"""LangGraph wiring for Atlas — the only place the graph is defined.

Flow: START -> planner -> (Send fan-out) -> retrieval (parallel) -> synthesis
-> END. The conditional fan-out from the planner uses the Send API so every
sub-question is researched concurrently, and the evidence reducer in
``AtlasState`` merges results from all branches. LangSmith tracing is active
whenever its environment variables are set (see ``config``).
"""

import logging

from langgraph.graph import END, START, StateGraph

import config
from models import Report
from nodes import dispatch_retrieval, planner_node, retrieval_node, synthesis_node
from state import AtlasState

logger = logging.getLogger(__name__)


def build_graph() -> "CompiledStateGraph":
    """Construct and compile the Atlas state graph.

    Returns:
        The compiled LangGraph application ready to invoke.
    """
    builder = StateGraph(AtlasState)
    builder.add_node("planner", planner_node)
    builder.add_node("retrieval", retrieval_node)
    builder.add_node("synthesis", synthesis_node)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", dispatch_retrieval, ["retrieval"])
    builder.add_edge("retrieval", "synthesis")
    builder.add_edge("synthesis", END)
    return builder.compile()


# Compiled once at import; LangGraph apps are reusable across runs.
_APP = build_graph()


async def run(question: str) -> Report:
    """Run the full research pipeline for a question and return the report.

    Args:
        question: The research question to investigate.

    Returns:
        The synthesised :class:`models.Report`.
    """
    config.validate_environment()
    logger.info("Atlas run started (tracing=%s)", config.tracing_enabled())
    final_state = await _APP.ainvoke({"question": question, "evidence": []})
    report = final_state.get("report")
    if report is None:
        raise RuntimeError("Graph finished without producing a report")
    return report
