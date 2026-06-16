"""The single point where the LLM provider is named.

Every LLM call in Atlas goes through ``get_llm()`` (and the retrying invoke
helpers here), so swapping providers later means editing only this file. The
helpers wrap calls with exponential backoff for Gemini free-tier 429 errors.
"""

import asyncio
import logging
import re
import time

from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

import config

logger = logging.getLogger(__name__)


def _retryable_exceptions() -> tuple[type[Exception], ...]:
    """Return exception types that signal a transient, retryable LLM error.

    Covers 429 rate limits and 5xx server overloads from the Gemini SDK. Imports
    are lazy so the module still loads if a dependency's layout differs.
    """
    excs: list[type[Exception]] = []
    try:
        from google.genai.errors import ServerError

        excs.append(ServerError)
    except ImportError:
        pass
    try:
        from google.api_core.exceptions import (
            ResourceExhausted,
            ServiceUnavailable,
            TooManyRequests,
        )

        excs.extend([ResourceExhausted, ServiceUnavailable, TooManyRequests])
    except ImportError:
        pass
    return tuple(excs)


_RETRYABLE = _retryable_exceptions()
_RETRYABLE_MARKERS = (
    "429",
    "resource_exhausted",
    "resource exhausted",
    "503",
    "unavailable",
    "overloaded",
    "too many requests",
)


class DailyQuotaExceededError(RuntimeError):
    """Raised when the Gemini free-tier *daily* request quota is exhausted."""


_DAILY_QUOTA_MARKERS = ("perday", "requests per day", "requestsperday")


def _is_daily_quota_error(error: Exception) -> bool:
    """Return True when the error is a per-day quota cap (not a per-minute one).

    Retrying a daily cap is futile (it resets only once a day), so callers should
    fail fast instead of waiting through the backoff.
    """
    text = str(error).lower()
    return "429" in text and any(m in text for m in _DAILY_QUOTA_MARKERS)


def _is_retryable_error(error: Exception) -> bool:
    """Return True when an exception is a transient 429/5xx worth retrying."""
    if _is_daily_quota_error(error):
        return False
    if _RETRYABLE and isinstance(error, _RETRYABLE):
        return True
    text = str(error).lower()
    return any(marker in text for marker in _RETRYABLE_MARKERS)


_DAILY_QUOTA_MESSAGE = (
    "Gemini free-tier daily quota exhausted (about 20 requests/day for "
    f"'{config.MODEL}'). It resets around midnight Pacific time. Try again later, "
    "or set MODEL in .env to another free model such as gemini-2.5-flash-lite."
)


_RETRY_DELAY_RE = re.compile(r"retry(?:\s+in|delay['\"]?:?\s*['\"]?)\s*([\d.]+)\s*s", re.I)


def _retry_delay(error: Exception, fallback: float) -> float:
    """Return the server-suggested retry delay, else the fallback (capped at 60s).

    Gemini 429s include the exact wait (e.g. "retry in 47s"); honouring it is far
    more efficient than a blind exponential backoff against a per-minute limit.
    """
    match = _RETRY_DELAY_RE.search(str(error))
    suggested = float(match.group(1)) if match else fallback
    return min(suggested, 60.0)


def get_llm(temperature: float = config.LLM_TEMPERATURE) -> ChatGoogleGenerativeAI:
    """Return a configured Gemini chat model — the only provider definition.

    A low default temperature keeps planning and synthesis deterministic;
    thinking is disabled and output capped so structured JSON is not truncated.
    """
    return ChatGoogleGenerativeAI(
        model=config.MODEL,
        temperature=temperature,
        google_api_key=config.GOOGLE_API_KEY,
        max_retries=config.LLM_MAX_RETRIES,
        max_tokens=config.LLM_MAX_OUTPUT_TOKENS,
        thinking_budget=config.LLM_THINKING_BUDGET,
    )


def invoke_llm(messages: list[BaseMessage], temperature: float = config.LLM_TEMPERATURE) -> str:
    """Invoke the LLM synchronously, retrying 429s with exponential backoff."""
    llm = get_llm(temperature)
    for attempt in range(config.RETRY_ATTEMPTS):
        try:
            return llm.invoke(messages).content
        except Exception as error:  # noqa: BLE001 - narrowed immediately below
            if _is_daily_quota_error(error):
                raise DailyQuotaExceededError(_DAILY_QUOTA_MESSAGE) from error
            if not _is_retryable_error(error) or attempt == config.RETRY_ATTEMPTS - 1:
                raise
            wait = _retry_delay(error, config.RETRY_BACKOFF_SECONDS[attempt])
            logger.warning("Transient LLM error (429/5xx); retrying in %.0fs", wait)
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
            if _is_daily_quota_error(error):
                raise DailyQuotaExceededError(_DAILY_QUOTA_MESSAGE) from error
            if not _is_retryable_error(error) or attempt == config.RETRY_ATTEMPTS - 1:
                raise
            wait = _retry_delay(error, config.RETRY_BACKOFF_SECONDS[attempt])
            logger.warning("Transient LLM error (429/5xx); retrying in %.0fs", wait)
            await asyncio.sleep(wait)
    raise RuntimeError("Unreachable: retry loop exited without returning")
