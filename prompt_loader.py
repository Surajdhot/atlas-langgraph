"""Helper for loading prompt templates from the prompts/ directory.

Prompts live as ``.txt`` files (never hardcoded in Python). This module reads
them and fills ``{placeholder}`` fields via ``str.format``.
"""

from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"


@lru_cache(maxsize=None)
def _read_prompt(name: str) -> str:
    """Read and cache the raw text of a prompt file by name (without suffix)."""
    path = PROMPTS_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8")


def load_prompt(name: str, **fields: object) -> str:
    """Load a prompt template and substitute the given placeholder fields.

    Args:
        name: Prompt file stem, e.g. ``"planner"`` for ``prompts/planner.txt``.
        **fields: Values substituted into ``{placeholder}`` markers.

    Returns:
        The rendered prompt string.
    """
    return _read_prompt(name).format(**fields)
