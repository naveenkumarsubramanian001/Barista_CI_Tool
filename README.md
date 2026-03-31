# Barista CI — AI-Powered Competitive Intelligence

> A full-stack research platform that generates comprehensive, citation-grounded competitive intelligence reports using multi-agent AI workflows.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Frontend                        │
│  React + Vite + Tailwind CSS + Framer Motion    │
│  (pnpm monorepo)                                │
└──────────────────────┬──────────────────────────┘
                       │ REST API
┌──────────────────────▼──────────────────────────┐
│               Python Backend                     │
│           FastAPI + LangGraph                    │
│                                                  │
│  ┌─────────────┐  ┌──────────┐  ┌────────────┐ │
│  │   Query      │→│  Multi   │→│  Hybrid    │ │
│  │ Decomposer  │  │ Search   │  │ Fuzzy      │ │
│  └─────────────┘  │ Agent    │  │Discrimin.  │ │
│                    └──────────┘  └─────┬──────┘ │
│                                        ▼        │
│       ┌──────────────┐  ┌─────────────────────┐ │
│       │ Rank + Filter│→│  Summariser (Groq)  │ │
│       └──────────────┘  │  Anti-hallucination │ │
│                         │  Constrained Prompt │ │
│                         └─────────────────────┘ │
└─────────────────────────────────────────────────┘
```

## Key Features

- **Multi-source search** — Tavily, Serper, Bing, Google CSE
- **Hybrid Fuzzy Discriminator** — Mamdani fuzzy inference + weighted scoring for unbiased article evaluation
- **Anti-hallucination guardrails** — temperature=0, citation-only constraints, post-generation validation
- **Full report rendering** — Executive summary, key findings, cross-source analysis, official/trusted insights
- **Clickable inline citations** — `[N]` references scroll to the sources section
- **Company Tracker** — Monitor competitor news over time
- **PDF export** — Generate downloadable research reports

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- pnpm 9+

### Backend Setup

```bash
# 1. Copy environment file
cp .env.example .env
# Fill in your API keys (GROQ_API_KEY required at minimum)

# 2. Install dependencies
pip install -r requirements.txt
# or with uv:
uv sync

# 3. Start the server
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# 4. (Optional) Run pipeline test via CLI
# Run with the default query
python main.py

# Or test a custom query
python main.py --query "Latest advancements in Tesla's Optimus Robot"
```

### Frontend Setup

```bash
cd Frontend

# 1. Install dependencies
pnpm install

# 2. Start development server
pnpm dev
```

The frontend runs on `http://localhost:5173` by default and proxies API calls to the Python backend on port 8000.

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ | Groq API key for LLM access |
| `GROQ_MODEL` | | Model name (default: `llama-3.3-70b-versatile`) |
| `TAVILY_API_KEY` | ✅ | Tavily search API key |
| `SERPER_API_KEY` | | Serper Google search API |
| `BING_SEARCH_API_KEY` | | Bing Search API key |
| `GOOGLE_API_KEY` | | Google custom search API key |
| `GOOGLE_CSE_ID` | | Google custom search engine ID |

## Project Structure

```
├── api.py                    # FastAPI server entry point
├── config.py                 # LLM, embedding, and search config
├── database.py               # Database stub (in-memory)
├── scheduler.py              # Scheduler stub
├── agents/
│   ├── QueryDecomposer.py    # Query decomposition + embedding alignment
│   ├── multi_search_agent.py # Parallel multi-provider search
│   ├── discriminators.py     # Hybrid fuzzy scoring + LLM evaluation
│   ├── fuzzy_discriminator.py# Mamdani fuzzy inference system
│   ├── summariser.py         # Report generation with anti-hallucination
│   └── analyzer_agents.py    # Competitor analysis agents
├── graph/
│   ├── workflow.py           # Main LangGraph research workflow
│   └── analyzer_workflow.py  # Sub-workflow for document analysis
├── nodes/
│   └── rank_filter.py        # Article ranking and filtering
├── routers/
│   ├── analyze.py            # Document upload + analysis
│   └── companies.py          # Company tracker CRUD
├── utils/
│   └── json_utils.py         # Robust JSON extraction from LLM output
├── Frontend/
│   ├── artifacts/
│   │   ├── research-platform/ # React frontend app
│   │   └── api-server/        # Express mock API (dev only)
│   └── lib/
│       └── api-spec/          # OpenAPI specification
└── .env.example              # Environment variable template
```

## License

MIT
