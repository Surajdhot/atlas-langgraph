"""Central configuration for Atlas.

All constants and environment-derived settings live here so that no logic
file hardcodes magic values. Importing this module loads the ``.env`` file,
wires LangSmith tracing environment variables, and configures logging.
"""

import logging
import os

from dotenv import load_dotenv

load_dotenv()

# --- Core research constants -------------------------------------------------
MAX_SUB_QUESTIONS: int = 5
MIN_SUB_QUESTIONS: int = 2
MAX_EVIDENCE_PER_QUESTION: int = 4
CONFIDENCE_THRESHOLD: float = 0.6
# Multiplier applied to a claim's confidence when sources conflict.
CONFLICT_PENALTY: float = 0.5

# --- LLM settings ------------------------------------------------------------
# A current free-tier Gemini flash model. Overridable via the MODEL env var.
MODEL: str = os.getenv("MODEL", "gemini-2.0-flash")
LLM_TEMPERATURE: float = 0.2

# Gemini free tier is strictly rate limited, so retries use long waits.
RETRY_ATTEMPTS: int = 3
RETRY_BACKOFF_SECONDS: tuple[int, ...] = (4, 8, 16)
# Stagger between concurrent retrieval branches to avoid a 429 thundering herd.
PARALLEL_STAGGER_SECONDS: float = 1.0

# --- External source settings ------------------------------------------------
HTTP_TIMEOUT_SECONDS: float = 15.0
WIKIPEDIA_SUMMARY_URL: str = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
WIKIPEDIA_SEARCH_URL: str = "https://en.wikipedia.org/w/api.php"
ARXIV_API_URL: str = "http://export.arxiv.org/api/query"
ARXIV_MAX_RESULTS: int = MAX_EVIDENCE_PER_QUESTION
WEB_SEARCH_MAX_RESULTS: int = MAX_EVIDENCE_PER_QUESTION

# --- Secrets -----------------------------------------------------------------
GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()


def _wire_langsmith_tracing() -> None:
    """Propagate LangSmith settings so tracing turns on when keys are present.

    LangSmith is optional: if no API key is configured, tracing is disabled
    and the rest of the system runs unchanged on free services only.
    """
    api_key = os.getenv("LANGCHAIN_API_KEY")
    if api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
        os.environ["LANGCHAIN_API_KEY"] = api_key
        os.environ["LANGCHAIN_PROJECT"] = os.getenv(
            "LANGCHAIN_PROJECT", "atlas-research-engine"
        )
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"


def configure_logging() -> None:
    """Configure root logging once, using the level from the environment."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def validate_environment() -> None:
    """Validate required environment variables, raising a clear error if missing.

    ``GOOGLE_API_KEY`` is mandatory. LangSmith keys are optional; their absence
    simply disables tracing.
    """
    if not GOOGLE_API_KEY:
        raise EnvironmentError(
            "GOOGLE_API_KEY is required. Get a free key at "
            "https://aistudio.google.com/app/apikey and set it in your .env file."
        )


def tracing_enabled() -> bool:
    """Return whether LangSmith tracing is currently enabled."""
    return os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() == "true"


_wire_langsmith_tracing()
configure_logging()
