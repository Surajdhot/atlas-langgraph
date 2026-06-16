# 🧭 Atlas — Multi-Agent Research Engine

Atlas answers a research question by decomposing it into sub-questions,
gathering evidence from multiple free sources **in parallel**, and synthesising
a cited report where every claim carries a confidence score and contradictions
between sources are flagged rather than hidden.

Built on **LangGraph** for orchestration and **LangSmith** for observability,
using **Google Gemini (free tier)** as the LLM. Every external service is free.

## Architecture

```
            ┌───────────┐
question →  │  planner  │  decompose into 2–5 sub-questions
            └─────┬─────┘
                  │  Send API fan-out (one branch per sub-question)
        ┌─────────┼─────────┐
        ▼         ▼         ▼
   ┌─────────┐         (parallel retrieval branches)
   │retrieval│  each branch: LLM picks sources → web / Wikipedia / arXiv
   └────┬────┘         evidence merged via a state reducer
        │
        ▼
   ┌──────────┐
   │synthesis │  grounded claims + per-claim confidence + conflict flags
   └────┬─────┘
        ▼
     Report  → Streamlit UI (confidence bars, sources, overall score)
```

Three technical centrepieces:
1. **Parallel retrieval** via LangGraph's `Send` API — sub-questions are
   researched concurrently, not sequentially.
2. **A state reducer** (`merge_evidence` in `state.py`) that accumulates
   evidence from every parallel branch instead of overwriting it.
3. **Graceful Gemini free-tier rate-limit handling** — 429s retry with
   exponential backoff, and branches stagger to avoid a thundering herd.

## Free services used

| Concern        | Service                          | Key needed            |
| -------------- | -------------------------------- | --------------------- |
| LLM            | Google Gemini (`gemini-2.5-flash`) | free AI Studio key  |
| Web search     | DuckDuckGo (`duckduckgo-search`) | none                  |
| Encyclopaedia  | Wikipedia REST API               | none                  |
| Papers         | arXiv API                        | none                  |
| Tracing        | LangSmith (optional)             | free key (optional)   |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # then add your free GOOGLE_API_KEY
streamlit run app.py
```

Get a free Google AI Studio key: https://aistudio.google.com/app/apikey
LangSmith tracing is optional — leave `LANGCHAIN_API_KEY` blank to disable it.

### Free-tier quota (important)

The Gemini free tier is small: `gemini-2.5-flash` allows roughly **5 requests
per minute and ~20 per day**. Each research run uses ~6–7 model calls, so you
get only a few full runs per day. Atlas handles this gracefully:

- 429 (rate-limit) and 503 (overload) errors retry with backoff, honouring the
  delay the API suggests.
- When the **daily** quota is exhausted, Atlas fails fast with a clear message
  instead of spinning — it resets around midnight Pacific time.
- To stretch the quota, set `MODEL=gemini-2.5-flash-lite` in `.env` (a separate,
  lighter free model). `gemini-2.0-flash` and `gemini-1.5-flash` are **not**
  available on the current free tier and will not work.

## Run with Docker

```bash
cp .env.example .env          # add your key
docker compose up --build
# open http://localhost:8501
```

## Tests

```bash
pytest tests/
```

All external API and LLM calls are mocked, so the suite runs offline and free.

## Project layout

- `config.py` — constants, env loading, LangSmith wiring, validation
- `llm.py` — the single `get_llm()` factory + 429 backoff (only provider swap point)
- `state.py` — the typed `AtlasState` with the evidence-merging reducer
- `models.py` — `SubQuestion`, `Evidence`, `Claim`, `Report`
- `graph.py` — the LangGraph definition (the only place the graph is wired)
- `nodes/` — planner, parallel retrieval, synthesis (one file each)
- `tools/` — DuckDuckGo, Wikipedia, arXiv (one LangChain tool each)
- `prompts/` — prompt templates as `.txt` files
- `app.py` — Streamlit UI
