"""External data-source tools for Atlas.

Each tool is a LangChain ``@tool`` wrapping a free, keyless data source and
returning a list of :class:`models.Evidence`.
"""

from tools.arxiv import arxiv_search
from tools.web_search import web_search
from tools.wikipedia import wikipedia_lookup

# Registry used by the retrieval node to resolve tool names chosen by the LLM.
TOOL_REGISTRY = {
    "web": web_search,
    "wikipedia": wikipedia_lookup,
    "arxiv": arxiv_search,
}

__all__ = ["web_search", "wikipedia_lookup", "arxiv_search", "TOOL_REGISTRY"]
