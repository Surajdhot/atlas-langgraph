"""Domain dataclasses shared across the Atlas graph.

These are plain dataclasses (not graph state). They model the research
artefacts that flow between nodes: sub-questions, evidence, claims, and the
final report.
"""

from dataclasses import dataclass, field
from enum import Enum


class SubQuestionStatus(str, Enum):
    """Lifecycle status of a sub-question as it moves through the graph."""

    PENDING = "pending"
    RESEARCHED = "researched"
    FAILED = "failed"


@dataclass
class SubQuestion:
    """A single focused, independently researchable sub-question."""

    id: str
    text: str
    status: SubQuestionStatus = SubQuestionStatus.PENDING


@dataclass
class Evidence:
    """A discrete piece of evidence gathered from one external source."""

    source_name: str
    source_url: str
    content: str
    sub_question_id: str


@dataclass
class Claim:
    """A synthesised claim with a confidence score and its evidential basis.

    ``confidence`` is in ``[0, 1]``. ``conflicting_evidence`` is non-empty when
    sources disagree, which lowers the confidence rather than hiding the clash.
    """

    text: str
    confidence: float
    supporting_evidence: list[Evidence] = field(default_factory=list)
    conflicting_evidence: list[Evidence] = field(default_factory=list)


@dataclass
class Report:
    """The final cited report combining all claims for a research question."""

    question: str
    claims: list[Claim] = field(default_factory=list)
    sources: list[Evidence] = field(default_factory=list)
    overall_confidence: float = 0.0
