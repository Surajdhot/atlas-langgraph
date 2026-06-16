"""Tests for the synthesis node — LLM is mocked, no network calls."""

import json
from importlib import import_module
from unittest.mock import patch

from models import Evidence
from nodes.synthesis_node import synthesis_node

# Fetch the real submodule (the package __init__ shadows the name with the function).
syn_mod = import_module("nodes.synthesis_node")


def _state() -> dict:
    """Build a state with two evidence items for a single question."""
    evidence = [
        Evidence("Source A", "https://a", "content a", "sq0"),
        Evidence("Source B", "https://b", "content b", "sq0"),
    ]
    return {"question": "Q", "evidence": evidence}


def _llm_payload() -> str:
    """LLM response: a supported claim, a conflicting claim, an unsupported one."""
    return json.dumps(
        {
            "claims": [
                {
                    "text": "supported no conflict",
                    "confidence": 0.8,
                    "supporting_evidence_ids": [0, 1],
                    "conflicting_evidence_ids": [],
                },
                {
                    "text": "supported with conflict",
                    "confidence": 0.8,
                    "supporting_evidence_ids": [0],
                    "conflicting_evidence_ids": [1],
                },
                {
                    "text": "unsupported",
                    "confidence": 0.9,
                    "supporting_evidence_ids": [],
                    "conflicting_evidence_ids": [],
                },
            ]
        }
    )


def test_no_claim_without_supporting_evidence() -> None:
    """Claims with no supporting evidence are never included."""
    with patch.object(syn_mod, "invoke_llm", return_value=_llm_payload()):
        report = synthesis_node(_state())["report"]

    texts = {c.text for c in report.claims}
    assert "unsupported" not in texts
    assert len(report.claims) == 2
    assert all(c.supporting_evidence for c in report.claims)


def test_conflicting_evidence_lowers_confidence() -> None:
    """A claim with conflicting evidence scores lower than one without."""
    with patch.object(syn_mod, "invoke_llm", return_value=_llm_payload()):
        report = synthesis_node(_state())["report"]

    by_text = {c.text: c for c in report.claims}
    assert (
        by_text["supported with conflict"].confidence
        < by_text["supported no conflict"].confidence
    )


def test_overall_confidence_is_average() -> None:
    """Overall confidence is the mean of the kept claims' confidences."""
    with patch.object(syn_mod, "invoke_llm", return_value=_llm_payload()):
        report = synthesis_node(_state())["report"]

    expected = round(sum(c.confidence for c in report.claims) / len(report.claims), 3)
    assert report.overall_confidence == expected
