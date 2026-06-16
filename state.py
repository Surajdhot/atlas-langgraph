"""The single typed graph state for Atlas.

``AtlasState`` is the one TypedDict that flows through the LangGraph graph.
The ``evidence`` field uses an ``Annotated`` reducer so that evidence produced
by parallel retrieval branches (fanned out with the Send API) is *merged*
rather than overwritten — this accumulation is essential for parallelism.
"""

import operator
from typing import Annotated, Optional, TypedDict

from models import Evidence, Report, SubQuestion


def merge_evidence(
    existing: list[Evidence], new: list[Evidence]
) -> list[Evidence]:
    """Reducer that concatenates evidence from concurrent retrieval branches.

    Without a reducer, the last parallel branch to write would overwrite the
    others. Concatenation lets every branch's evidence accumulate into one list.
    """
    return list(existing) + list(new)


class AtlasState(TypedDict, total=False):
    """State shared across all graph nodes.

    ``sub_question`` is a transient field populated only inside Send-dispatched
    retrieval branches; the main pipeline uses ``sub_questions`` (plural).
    """

    question: str
    sub_questions: list[SubQuestion]
    sub_question: SubQuestion
    evidence: Annotated[list[Evidence], merge_evidence]
    report: Optional[Report]


# Exported so tests can assert the reducer behaviour directly.
_EVIDENCE_REDUCER = operator.add
