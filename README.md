# City Trip Planner

A multi-agent travel planner built with Python, LangChain, LangGraph and Streamlit. Describe a city trip in plain language, review a day-by-day itinerary grounded in live web research, request changes, and approve the final version before exporting it.

Five agents cooperate over shared, checkpointed state: an Orchestrator, two parallel researchers for sights and food, a Planner and a Critic. The app uses Nebius Token Factory for language models, Tavily for web research, Open-Meteo for weather and SQLite for saved progress. LangSmith tracing is optional. There is no RAG, vector store or ingestion step.

## Quick start (Windows PowerShell)

Requires Python 3.11 or newer and Nebius and Tavily credentials. Run these commands from the Week3 project directory:

```powershell
cd C:\Sunita\GIT\GenAcademyProject\Week3
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Create .env only if it does not already exist.
if (!(Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

Fill in these values in your local `.env`:

```dotenv
NEBIUS_API_KEY=your_nebius_key
NEBIUS_BASE_URL=your_nebius_api_base_url
NEBIUS_MODEL=your_model_id
TAVILY_API_KEY=your_tavily_key
LANGSMITH_TRACING=false
```

Choose a Nebius model that supports tool calling and structured output. For optional tracing, set `LANGSMITH_TRACING=true` and supply `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT`. Existing shell environment variables take precedence over `.env`. Keep credentials local; do not commit them.

Start the Streamlit application:

```powershell
python -m streamlit run src/app.py
```

Open the URL printed in the terminal, normally **http://localhost:8501**. Stop the application with **Ctrl+C**.

For subsequent runs:

```powershell
cd C:\Sunita\GIT\GenAcademyProject\Week3
.\.venv\Scripts\Activate.ps1
python -m streamlit run src/app.py
```

If PowerShell blocks virtual environment activation, use its Python executable directly:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/app.py
```

To use a different port:

```powershell
python -m streamlit run src/app.py --server.port 8502
```

## Example user inputs

Include a city, trip length, interests, budget preference and pace. Add an ISO start date (`YYYY-MM-DD`) if you want the Planner to request weather; choose a date within the available forecast window.

| Example request | What it asks for |
| --- | --- |
| `1 day in Lisbon, relaxed pace, history, budget` | A small first trip to try |
| `4 days in Lisbon, normal pace, food and history, budget` | Food and historical sights with a budget preference considered |
| `3 days in Kyoto, relaxed pace, temples and gardens, mid-range budget` | A slower cultural trip |
| `2 days in Paris, packed pace, art and food, luxury budget` | A fuller schedule with food and art |

Pace controls daily stop counts: **relaxed: 3**, **normal: 4**, **packed: 5**. Research must find enough evidenced sights and food places before planning can begin.

Example feedback after selecting **Request changes**:

- `Keep Lisbon, but shorten the trip to 2 days and use a relaxed pace.`
- `Keep the city and dates, but focus more on history and reduce the budget.`
- `Change the destination to Porto for 3 days, with food and history.`

## Using the app

1. Enter a trip request and start planning. Follow the agent activity as sights and food research are collected, merged, drafted and checked.
2. Review the daily stops, sources and warnings. Weather may be unavailable; uncertain details remain visible as soft findings.
3. Select **Request changes** to revise the trip, **Approve & export** to export the displayed checked version, or **Cancel**.
4. Download the JSON and Markdown after a complete export. Files are also stored in `outputs/`; a manifest records their SHA-256 hashes.

Paused trips persist in `checkpoints.sqlite`. After restarting the app, select a trip under **Resume trip** in the sidebar and click **Open**. **New trip** starts a separate thread.

If an agent fails or research coverage is insufficient, recovery offers the allowed **Revise**, **Retry** or **Cancel** actions. At the trip step limit, only **New trip** or **Cancel** is available. Retry does not reset exhausted research attempts; use a revision or a new trip when needed. A failed file write offers **Retry export** without requiring a new plan.

## Workflow and safeguards

```text
Request -> Orchestrator -> Sights + Food research (parallel)
        -> Merge research -> Planner -> Critic
        -> Traveler approval / revision -> Export
```

The Orchestrator chooses the next worker after each step, so research and planning can repeat before review. Python checks enforce valid evidence, the requested day count, pace limits and unique stops. A draft with hard failures cannot be offered for approval. Only the exact approved, Critic-checked version can be exported; caps never force approval.

Each accepted place carries a source URL and a quote matched against saved research evidence. Evidence establishes where a place came from; opening hours, prices and availability may change. Budget preference is considered. The app never books, pays or contacts anyone.

## Project layout

| Path | Purpose |
| --- | --- |
| `src/app.py` | Streamlit interface |
| `src/graph.py` | LangGraph workflow, persistence and CLI |
| `src/agents/` | Orchestrator, researchers, Planner, Critic and prompts |
| `src/tools.py` | Web research, weather and guarded export tools |
| `src/state.py`, `src/evidence.py`, `src/coverage.py`, `src/gate.py` | Shared state and validation rules |
| `src/export.py` | JSON, Markdown and manifest export |
| `scripts/` | Provider smoke tests and evaluation |
| `tests/` | Mocked checks and opt-in live tests |
| `docs/` | Architecture and demo documentation |
| `outputs/` | Generated exports and evaluation output |
| `PROMPTS.md` | Development prompts and outcomes |

## Developer commands

Run from Week3 with the virtual environment activated.

```powershell
# MOCKED by default: fake providers; no API keys required.
python -m pytest -q
python -m ruff check src tests

# Generate the workflow diagram without API keys.
python -m src.graph --diagram

# CLI alternative to Streamlit (uses live providers).
python -m src.graph "1 day in Lisbon, relaxed, history"
```

Optional **LIVE** checks use configured services and may incur provider charges:

```powershell
python -m scripts.smoke_test
$env:RUN_LIVE = "1"
python -m pytest -q tests/test_tools_live.py
Remove-Item Env:RUN_LIVE
```

Mocked tests check workflow behavior; they do not establish live itinerary quality or response time. Live provider and browser behavior must be checked separately.

## Troubleshooting

- **Module not found:** use the same virtual environment for installation and startup, and run commands from Week3.
- **Missing credentials or provider errors:** check `.env`, the Nebius base URL/model and any overriding shell variables. Restart Streamlit after changing configuration.
- **No weather:** provide a start date within the forecast window. Missing weather is a visible soft finding.
- **Insufficient research:** review recovery warnings, revise the request or start a new trip. Planning cannot proceed without enough evidenced places.
- **Incomplete export:** use **Retry export**. A lone JSON or Markdown file is not a complete export without its matching manifest.

The previous README is preserved in [README.backup.md](README.backup.md).
