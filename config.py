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
# Note: gemini-2.0-flash has no free-tier quota on many keys; 2.5-flash does.
MODEL: str = os.getenv("MODEL", "gemini-2.5-flash")
LLM_TEMPERATURE: float = 0.2
# Cap output and disable "thinking" tokens: our nodes emit structured JSON, and
# thinking on gemini-2.5 models can consume the output budget and truncate it.
LLM_MAX_OUTPUT_TOKENS: int = 4096
LLM_THINKING_BUDGET: int = 0

# Gemini free tier is strictly rate limited (~5 requests/minute), so retries use
# long waits. The actual wait usually comes from the 429's own suggested delay.
RETRY_ATTEMPTS: int = 4
RETRY_BACKOFF_SECONDS: tuple[int, ...] = (8, 16, 32, 48)
# Let our own backoff own retries; disable the SDK's internal retry so waits
# do not stack (a stacked retry could block a single call for ~2 minutes).
LLM_MAX_RETRIES: int = 0
# Stagger between concurrent retrieval branches to avoid a 429 thundering herd.
PARALLEL_STAGGER_SECONDS: float = 2.0

# --- External source settings ------------------------------------------------
HTTP_TIMEOUT_SECONDS: float = 15.0
WIKIPEDIA_SUMMARY_URL: str = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
WIKIPEDIA_SEARCH_URL: str = "https://en.wikipedia.org/w/api.php"
ARXIV_API_URL: str = "https://export.arxiv.org/api/query"
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
    api_key = os.getenv("LANGCHAIN_API_KEY", "")
    # Ignore an unset key or the placeholder shipped in .env.example, otherwise
    # every run spams 403 errors trying to send traces with a fake key.
    if api_key and not api_key.startswith("your_"):
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
