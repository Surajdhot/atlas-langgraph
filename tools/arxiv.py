"""arXiv API tool returning paper abstracts as evidence (free, no API key)."""

import logging

import feedparser
import httpx
from langchain_core.tools import tool

import config
from models import Evidence

logger = logging.getLogger(__name__)


def _query_arxiv(query: str) -> list[dict]:
    """Query the arXiv Atom API and return parsed feed entries."""
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": config.ARXIV_MAX_RESULTS,
        "sortBy": "relevance",
    }
    try:
        response = httpx.get(
            config.ARXIV_API_URL,
            params=params,
            timeout=config.HTTP_TIMEOUT_SECONDS,
            follow_redirects=True,
        )
        response.raise_for_status()
    except httpx.HTTPError as error:
        logger.warning("arXiv query failed for %r: %s", query, error)
        return []
    return feedparser.parse(response.text).entries


def _entry_to_evidence(entry: dict, sub_question_id: str) -> Evidence:
    """Convert a single parsed arXiv entry into an Evidence object."""
    summary = entry.get("summary", "").replace("\n", " ").strip()
    title = entry.get("title", "Untitled").replace("\n", " ").strip()
    return Evidence(
        source_name=f"arXiv: {title}",
        source_url=entry.get("link", ""),
        content=summary,
        sub_question_id=sub_question_id,
    )


@tool
def arxiv_search(query: str, sub_question_id: str = "") -> list[Evidence]:
    """Search arXiv for scientific paper abstracts on technical topics.

    Args:
        query: The scientific topic or question to search for.
        sub_question_id: Id of the sub-question this evidence answers.

    Returns:
        A list of Evidence objects built from paper abstracts (possibly empty).
    """
    entries = _query_arxiv(query)
    if not entries:
        logger.info("arXiv returned no results for %r", query)
        return []
    return [_entry_to_evidence(entry, sub_question_id) for entry in entries]
