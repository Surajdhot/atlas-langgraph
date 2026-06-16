# CLAUDE.md

## Project identity
Atlas is a multi-agent research engine built on LangGraph, using Google
Gemini (free tier) as the LLM. A planner node decomposes a research question
into sub-questions, retrieval nodes gather evidence from DuckDuckGo web
search, Wikipedia, and arXiv in parallel, and a synthesis node combines
evidence into a cited report with per-claim confidence scores. It flags
contradictions between sources rather than hiding them. LangSmith traces
every node execution.

## Cost constraint — strict
- Every external service must be free.
- LLM: Google Gemini via langchain-google-genai (free API tier).
- Web search: DuckDuckGo via the duckduckgo-search library (no key).
- Wikipedia and arXiv: free, keyless.
- LangSmith: free individual tier, optional.
- Never add a paid service or one requiring a credit card.

## Code style rules
- Python only.
- All functions must have docstrings.
- Type hints on every function signature.
- Max function length: 40 lines. Split if longer.
- No print statements — use the logging module everywhere.
- Never use bare except. Catch specific exceptions.
- Constants go in config.py, never hardcoded in logic files.

## Architecture decisions — do not change these
- Use LangGraph for all orchestration. The graph is defined in graph.py only.
- Graph state is a single typed TypedDict defined in state.py.
- Each node is a function in nodes/ — one file per node.
- Each external data source is a LangChain tool in tools/.
- All LLM access goes through a single get_llm() factory in llm.py, so the
  model provider can be swapped in one place.
- All prompts live in prompts/ as .txt files — never hardcoded in Python.
- LangSmith tracing enabled via environment variables.

## What NOT to do
- Do not use CrewAI or AutoGen — LangGraph only.
- Do not use any paid API. Gemini free tier only.
- Do not collapse the graph into one giant function — use proper nodes/edges.
- Do not create a heavy frontend — Streamlit is the only UI.
- Do not write placeholder, stub, or TODO code — everything must work.
- Do not commit .env — only .env.example.

## Gemini free-tier handling
- The free tier has rate limits (requests per minute). All Gemini calls must
  handle 429 rate-limit errors with exponential backoff and retry.
- Keep model calls efficient — do not make redundant calls.

## Testing
- Every node and tool needs a test file in tests/.
- Use pytest with pytest-asyncio.
- Mock all external API and LLM calls in tests.

## Git commit style
- Conventional commits: feat:, fix:, refactor:, test:, docs:, chore:
- One logical change per commit.
- Commit after each component works.
