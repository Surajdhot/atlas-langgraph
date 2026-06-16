"""Lenient JSON extraction for LLM responses.

Gemini sometimes wraps JSON in markdown code fences or adds stray prose. These
helpers pull the first valid JSON array or object out of a raw response.
"""

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _strip_fences(text: str) -> str:
    """Remove a surrounding markdown code fence if present."""
    match = _FENCE_RE.search(text)
    return match.group(1) if match else text


def parse_json_array(text: str) -> list[Any]:
    """Extract a JSON array from an LLM response, returning [] on failure."""
    candidate = _strip_fences(text).strip()
    start, end = candidate.find("["), candidate.rfind("]")
    if start == -1 or end == -1 or end < start:
        logger.warning("No JSON array found in response: %r", text[:200])
        return []
    try:
        result = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as error:
        logger.warning("Failed to parse JSON array: %s", error)
        return []
    return result if isinstance(result, list) else []


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object from an LLM response, returning {} on failure."""
    candidate = _strip_fences(text).strip()
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end == -1 or end < start:
        logger.warning("No JSON object found in response: %r", text[:200])
        return {}
    try:
        result = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as error:
        logger.warning("Failed to parse JSON object: %s", error)
        return {}
    return result if isinstance(result, dict) else {}
