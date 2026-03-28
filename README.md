# Barista CI Tool

Competitive intelligence backend with LangGraph workflow orchestration, human-in-the-loop article selection, analyzer upload flow, and PDF report generation.

## Source of Truth for Dependencies

`pyproject.toml` is the canonical dependency source.
`requirements.txt` is a compatibility mirror for environments that still use `pip -r`.

## Quick Setup (Recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
cp .env.example .env
```

## Alternative Setup (requirements mirror)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

## Environment Configuration

Key runtime options in `.env`:

- `SEARCH_STRATEGY=parallel|single|fallback`
- `SEARCH_PROVIDER=tavily|serper|google|bing`
- `ENABLE_FUZZY_SCORING=true|false`
- `STRICT_STARTUP_VALIDATION=true|false`
- `CHECKPOINTER_BACKEND=sqlite|memory`
- `CHECKPOINT_DB_PATH=checkpoints.sqlite`
- `CORS_ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173`

At least one search provider key must be configured.

## Run API

```bash
source .venv/bin/activate
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

## Tests

Contract tests:

```bash
source .venv/bin/activate
python -m pytest -q tests/test_api_contract_e2e.py
```

PDF regression tests:

```bash
source .venv/bin/activate
python -m pytest -q tests/test_comparative_pdf_report.py
```

Live integration tests (may skip when optional runtime deps are unavailable):

```bash
source .venv/bin/activate
python -m pytest -q tests/test_integration_live_paths.py
```

## CI Quality Gates

CI runs on every push/PR and executes:

- `ruff check .`
- Python compile check
- API contract tests
- PDF regression tests
- Integration tests (allowed to skip when optional deps are not available)
