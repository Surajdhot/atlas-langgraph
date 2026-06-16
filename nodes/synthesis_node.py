"""Synthesis node: combine evidence into cited, confidence-scored claims."""

import logging

from langchain_core.messages import HumanMessage

import config
from llm import invoke_llm
from models import Claim, Evidence, Report
from parsing import parse_json_object
from prompt_loader import load_prompt
from state import AtlasState

logger = logging.getLogger(__name__)


def _format_evidence(evidence: list[Evidence]) -> str:
    """Render evidence as numbered lines the synthesis prompt can reference."""
    return "\n".join(
        f"[{i}] ({ev.source_name}) {ev.content}" for i, ev in enumerate(evidence)
    )


def _resolve(ids: list, evidence: list[Evidence]) -> list[Evidence]:
    """Map evidence indices from the LLM back to Evidence objects, ignoring bad ids.

    Accepts ints or numeric strings (the LLM sometimes quotes ids).
    """
    resolved = []
    for raw in ids:
        try:
            index = int(raw)
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(evidence):
            resolved.append(evidence[index])
    return resolved


def _score_claim(llm_confidence: object, support_count: int, has_conflict: bool) -> float:
    """Compute a claim's confidence from its evidential support and conflicts.

    More independent supporting sources raise confidence; any conflict applies
    a penalty so contradictions are reflected, not hidden.
    """
    if isinstance(llm_confidence, (int, float)):
        base = float(llm_confidence)
    else:
        base = min(1.0, 0.3 + 0.2 * support_count)
    if has_conflict:
        base *= config.CONFLICT_PENALTY
    return round(max(0.0, min(1.0, base)), 3)


def _build_claim(raw: dict, evidence: list[Evidence]) -> Claim | None:
    """Build a Claim from one raw LLM entry, dropping any with no support."""
    supporting = _resolve(raw.get("supporting_evidence_ids", []), evidence)
    if not supporting:
        return None
    conflicting = _resolve(raw.get("conflicting_evidence_ids", []), evidence)
    confidence = _score_claim(raw.get("confidence"), len(supporting), bool(conflicting))
    return Claim(
        text=str(raw.get("text", "")).strip(),
        confidence=confidence,
        supporting_evidence=supporting,
        conflicting_evidence=conflicting,
    )


def _build_report(question: str, claims: list[Claim], evidence: list[Evidence]) -> Report:
    """Assemble the final report and its overall confidence (mean of claims)."""
    overall = round(sum(c.confidence for c in claims) / len(claims), 3) if claims else 0.0
    return Report(
        question=question, claims=claims, sources=evidence, overall_confidence=overall
    )


def synthesis_node(state: AtlasState) -> dict:
    """Synthesise gathered evidence into a cited report with confidence scores.

    Args:
        state: Graph state with the original ``question`` and merged ``evidence``.

    Returns:
        A state update containing the final ``report``.
    """
    question = state["question"]
    evidence = state.get("evidence", [])
    logger.info("Synthesising %d evidence items", len(evidence))
    prompt = load_prompt("synthesis", question=question, evidence=_format_evidence(evidence))
    parsed = parse_json_object(invoke_llm([HumanMessage(content=prompt)]))

    claims = [c for raw in parsed.get("claims", []) if (c := _build_claim(raw, evidence))]
    logger.info("Synthesis produced %d grounded claims", len(claims))
    return {"report": _build_report(question, claims, evidence)}
