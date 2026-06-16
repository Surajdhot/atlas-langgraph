"""The single point where the LLM provider is named.

Every LLM call in Atlas goes through ``get_llm()`` (and the retrying invoke
helpers here), so swapping providers later means editing only this file. The
helpers wrap calls with exponential backoff for Gemini free-tier 429 errors.
"""

import asyncio
import logging
import time

from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

import config

logger = logging.getLogger(__name__)


def _rate_limit_exceptions() -> tuple[type[Exception], ...]:
    """Return the exception types that signal a retryable rate-limit (429).

    google-api-core is imported lazily so the module still loads if the exact
    dependency layout differs; a generic fallback keeps behaviour sane.
    """
    try:
        from google.api_core.exceptions import ResourceExhausted, TooManyRequests

        return (ResourceExhausted, TooManyRequests)
    except ImportError:
        return ()


_RETRYABLE = _rate_limit_exceptions()


def _is_rate_limit_error(error: Exception) -> bool:
    """Return True when an exception looks like a 429 rate-limit error."""
    if _RETRYABLE and isinstance(error, _RETRYABLE):
        return True
    text = str(error).lower()
    return "429" in text or "resource" in text and "exhausted" in text


def get_llm(temperature: float = config.LLM_TEMPERATURE) -> ChatGoogleGenerativeAI:
    """Return a configured Gemini chat model — the only provider definition.

    A low default temperature keeps planning and synthesis deterministic.
    """
    return ChatGoogleGenerativeAI(
        model=config.MODEL,
        temperature=temperature,
        google_api_key=config.GOOGLE_API_KEY,
    )


def invoke_llm(messages: list[BaseMessage], temperature: float = config.LLM_TEMPERATURE) -> str:
    """Invoke the LLM synchronously, retrying 429s with exponential backoff."""
    llm = get_llm(temperature)
    for attempt in range(config.RETRY_ATTEMPTS):
        try:
            return llm.invoke(messages).content
        except Exception as error:  # noqa: BLE001 - narrowed immediately below
            if not _is_rate_limit_error(error) or attempt == config.RETRY_ATTEMPTS - 1:
                raise
            wait = config.RETRY_BACKOFF_SECONDS[attempt]
            logger.warning("Rate limited (429); retrying in %ss", wait)
            time.sleep(wait)
    raise RuntimeError("Unreachable: retry loop exited without returning")


async def ainvoke_llm(
    messages: list[BaseMessage], temperature: float = config.LLM_TEMPERATURE
) -> str:
    """Invoke the LLM asynchronously, retrying 429s with exponential backoff.

    Used by concurrent retrieval branches so a burst of parallel calls that hit
    Gemini's per-minute limit degrades gracefully instead of all failing.
    """
    llm = get_llm(temperature)
    for attempt in range(config.RETRY_ATTEMPTS):
        try:
            result = await llm.ainvoke(messages)
            return result.content
        except Exception as error:  # noqa: BLE001 - narrowed immediately below
            if not _is_rate_limit_error(error) or attempt == config.RETRY_ATTEMPTS - 1:
                raise
            wait = config.RETRY_BACKOFF_SECONDS[attempt]
            logger.warning("Rate limited (429); retrying in %ss", wait)
            await asyncio.sleep(wait)
    raise RuntimeError("Unreachable: retry loop exited without returning")
