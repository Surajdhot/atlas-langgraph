"""DuckDuckGo web-search tool (free, no API key)."""

import logging

from langchain_core.tools import tool

import config
from models import Evidence

logger = logging.getLogger(__name__)


def _run_web_search(query: str, sub_question_id: str) -> list[Evidence]:
    """Query DuckDuckGo and map results to Evidence, tolerating failures.

    Network errors and empty results are logged and yield an empty list so a
    single failing source never breaks a retrieval branch.
    """
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(
                ddgs.text(query, max_results=config.WEB_SEARCH_MAX_RESULTS)
            )
    except (ImportError, ValueError, RuntimeError, ConnectionError) as error:
        logger.warning("Web search failed for %r: %s", query, error)
        return []

    if not results:
        logger.info("Web search returned no results for %r", query)
        return []

    return [
        Evidence(
            source_name="DuckDuckGo",
            source_url=item.get("href", ""),
            content=f"{item.get('title', '')}: {item.get('body', '')}".strip(),
            sub_question_id=sub_question_id,
        )
        for item in results
    ]


@tool
def web_search(query: str, sub_question_id: str = "") -> list[Evidence]:
    """Search the web via DuckDuckGo for current events and general queries.

    Args:
        query: The search query text.
        sub_question_id: Id of the sub-question this evidence answers.

    Returns:
        A list of Evidence objects (possibly empty on failure).
    """
    return _run_web_search(query, sub_question_id)
