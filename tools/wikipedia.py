"""Wikipedia REST summary tool (free, no API key)."""

import logging

import httpx
from langchain_core.tools import tool

import config
from models import Evidence

logger = logging.getLogger(__name__)


def _resolve_title(query: str) -> str | None:
    """Resolve a free-text query to the closest Wikipedia article title."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": 1,
    }
    try:
        response = httpx.get(
            config.WIKIPEDIA_SEARCH_URL,
            params=params,
            timeout=config.HTTP_TIMEOUT_SECONDS,
            headers={"User-Agent": "Atlas-Research-Engine/1.0"},
        )
        response.raise_for_status()
        hits = response.json().get("query", {}).get("search", [])
    except (httpx.HTTPError, ValueError, KeyError) as error:
        logger.warning("Wikipedia search failed for %r: %s", query, error)
        return None
    return hits[0]["title"] if hits else None


def _fetch_summary(title: str, sub_question_id: str) -> list[Evidence]:
    """Fetch the REST summary extract for a resolved article title."""
    url = config.WIKIPEDIA_SUMMARY_URL.format(title=title.replace(" ", "_"))
    try:
        response = httpx.get(
            url,
            timeout=config.HTTP_TIMEOUT_SECONDS,
            headers={"User-Agent": "Atlas-Research-Engine/1.0"},
        )
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, ValueError) as error:
        logger.warning("Wikipedia summary failed for %r: %s", title, error)
        return []

    extract = data.get("extract", "").strip()
    if not extract:
        return []
    page_url = data.get("content_urls", {}).get("desktop", {}).get("page", url)
    return [
        Evidence(
            source_name=f"Wikipedia: {data.get('title', title)}",
            source_url=page_url,
            content=extract,
            sub_question_id=sub_question_id,
        )
    ]


@tool
def wikipedia_lookup(query: str, sub_question_id: str = "") -> list[Evidence]:
    """Look up an encyclopaedic summary on Wikipedia for general knowledge.

    Args:
        query: The topic or question to look up.
        sub_question_id: Id of the sub-question this evidence answers.

    Returns:
        A list with at most one Evidence object (empty if nothing is found).
    """
    title = _resolve_title(query)
    if not title:
        logger.info("No Wikipedia article found for %r", query)
        return []
    return _fetch_summary(title, sub_question_id)
