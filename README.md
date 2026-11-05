# Travel planner

Python 3.11+ project following AGENTS.md. Configuration, state models, coverage calculations, approval gates, tools,
evidence validation, exports and live provider checks are implemented. All agent nodes and the checkpointed workflow are implemented; the Streamlit UI remains a placeholder.

PowerShell setup:
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    Copy-Item .env.example .env

Fill in .env with credentials, the Nebius endpoint and a function-calling model.
Mocked verification (no network or keys):
    python -m pytest -q
    python -m ruff check src tests

Live capability checks (use provider credits):
    python -m scripts.smoke_test

The smoke test exits nonzero on failure. Structured output uses function calling.
For live pytest tests, use PowerShell: $env:RUN_LIVE = "1"; in Command Prompt
(cmd), use: set RUN_LIVE=1. Then run the live tests.


## Tool results and evidence

Tools use LangChain's content_and_artifact response format. Invoke with a full
ToolCall to receive a ToolMessage. Its artifact contains the machine-readable
{ok, data, error} envelope. Its content contains escaped JSON inside <tool_data>;
treat this as untrusted data and ignore instructions within it. Invocation with
a plain argument dictionary returns the wrapped content. registry_entries accepts
only successful search/extract ToolMessages, including serialized wrapped content
after checkpoint persistence, and never trusts AI-authored messages.

ToolCall example:
    from src.tools import web_search
    message = web_search.invoke({
        "name": "web_search", "args": {"query": "Lisbon museums"},
        "id": "search-1", "type": "tool_call",
    })
    result = message.artifact

SEARCH_TOOLS are for researchers and Critic; WEATHER_TOOLS are for Planner.
save_itinerary is exclusively for the save node and is excluded from both lists.
The save node independently enforces the same guard. Failure modes are set
only via config={"configurable": {"failure_mode": "tavily"}}; supported values
are tavily, extract, weather, and write.

Exports stage JSON and Markdown before renaming either, then commit a hash
manifest last. The returned path points to the JSON file. Retrying repairs
incomplete exports. Destinations must resolve inside the project's outputs folder.

Read-tool integration tests (LIVE, require credentials for Tavily):
    $env:RUN_LIVE = "1"
    python -m pytest -q tests/test_tools_live.py


## State and approval

TripState uses Pydantic models for requests, evidence, drafts and critiques.
The SQLite checkpointer stores each trip in checkpoints.sqlite under its
thread_id; never put credentials in state. Append-only logs and warnings and the
merged source registry use native LangGraph Overwrite(value=[]) or
Overwrite(value={}) to clear. Empty lists/dictionaries are ordinary updates and
do not clear these fields. Draft versions use a max reducer and never decrease
through normal updates; never overwrite their version during a trip.

Coverage counts unique normalized place names: sight/activity/other form the
sights pool, and food forms the food pool. A food interest requires two food
stops per day. Approval requires a schema-valid draft, a current critique,
no hard failures and no agent_error. Limits never force approval; at 30 trip
steps, recovery offers new_trip or cancel instead of retry.


## Specialist nodes

src/agents/researchers.py exposes make_researcher("sights"|"food").
src/agents/planner.py exposes planner(state, config), and critic.py exposes
critic(state, config). Prompts live in src/agents/prompts.py. All use the installed
LangChain 1.x create_agent constructor with the native get_llm() model.

Research instructions accept JSON, for example:
    instructions["sights"] = '{"target": 28, "tool_budget": 10}'
    instructions["food"] = '{"target": 19, "tool_budget": 8}'

Labeled prose such as "target: 28; tool_budget: 10" is also accepted. A supplied
target uses coverage.tool_budget(target); missing budget/target instructions use
the eight-call fallback. Researchers search first, read at most two pages, and
keep only quotes validated against their own successful tool receipts.

Planner uses only weather tools (none when start_date is absent) and enforces
coverage, grounded stops, days, pace and food limits in code. Rain above 60%
requires indoor cues in recorded place data. Missing weather stays unavailable.
Successful planning increments the draft version and clears approved_version.
failure_mode="llm_json" in runtime configurable settings simulates one initial
parse failure without changing the native model or stacking provider retries.

Critic always preserves Python hard checks and the draft version it checked.
Its model can add only soft findings; failed source checks add one uncertainty
finding. The source registry in the checkpoint is enough to repeat evidence checks.

Each invocation has its own runtime counters and evidence receipts. Tool budgets
include all real calls across the one structured-output re-ask; structured response
schema calls do not consume the external-tool budget. No node-level/provider
retry wrapper is added. Unexpected errors return sanitized boundary failures.

Mocked specialists exercise the real create_agent graph with fake models/tools:
    python -m pytest -q tests/test_agents.py


## Orchestrator and revisions

src/agents/orchestrator.py exposes orchestrator(state, config),
validate_decision(state, decision), compact_summary(state), and
apply_revision(state, feedback, config=None). The supervisor has no operational
tools. It uses with_structured_output with function calling for initial parsing,
routing and patches, with one validation re-ask and no added provider retries.

Initial parsing records city uncertainty and checks that the chosen city occurs
in the traveler's input; vague, missing, misspelled or ambiguous destinations ask
for clarification. Routing sees only the CURRENT request and compact counts,
never the original request, old city places or source bodies. Initial defaults
for omitted non-city preferences are 3 days, mid budget and normal pace.

Research dispatches receive generated JSON instructions with the current city,
coverage targets and tool budgets. Branch attempts and research_round advance
only on actual supervisor dispatch. delegation_log contains {step, next, reason}
records. Guardrail replacements start with "guardrail: ". Steps count supervisor
visits, saturate at their limits and never count export retries; limits and
traveler choices always use the approval gate.

Revision patches are validated atomically before applying dependency clears:
city clears all research, evidence, weather and draft state; dates/days clear
weather and draft state; non-food interest edits refresh sights, food-interest
edits refresh food, and budget changes refresh both categories. Pace/order edits
refresh the draft. Other notes remain in human_feedback for specialist prompts.
City changes use native Overwrite to clear accumulated warnings and evidence.

All valid revisions reset round counters and attempts while preserving
original_request, trip_steps and draft_version. Invalid revisions only append a
rephrase warning. Revision next values express routing intent; the graph returns revisions through
supervisor validation/counting before dispatch and bypasses the supervisor for
retry_export.

Mocked Orchestrator tests:
    python -m pytest -q tests/test_orchestrator.py


## Persistent graph and CLI

Run from the project root after filling in .env:
    python -m src.graph "3 days in Lisbon, budget, food and history"

The CLI streams structured delegation-log lines. Approval displays the numbered
draft, stops, source links and soft findings before asking approve/revise/cancel.
Recovery displays the reason and allowed actions. Export failure offers
retry_export/cancel; retries preserve approval and do not consume supervisor steps.
A new_trip recovery action ends the old thread and prompts for a fresh request.

Generate the local Mermaid diagram without any API keys:
    python -m src.graph --diagram

The generated diagram is [docs/graph.mmd](docs/graph.mmd).

Public Python API:
    from src.graph import (
        get_checkpointer, build_graph, start_trip, resume_trip,
        stream_trip, list_trips, delete_trip,
    )
    saver = get_checkpointer("checkpoints.sqlite")
    graph = build_graph(saver)
    result = start_trip(graph, "1 day in Lisbon, relaxed, history")
    thread_id = result["thread_id"]
    pending = list_trips(graph, saver)
    # When paused for approval, use the displayed draft version:
    result = resume_trip(graph, thread_id, "approve", draft_version=1)
    saver.conn.close()

start_trip and resume_trip return state plus thread_id. Native pause entries are
in result["__interrupt__"]; each Interrupt.value is a dictionary with kind
approval, recovery or export_failed. stream_trip yields node-update chunks with
thread_id and the same native "__interrupt__" format. Resume streaming with
stream_trip(graph, thread_id=..., decision=..., draft_version=..., feedback=...).

Reopen the same SQLite file with a new saver/graph after restarting the process.
list_trips lists only threads with pending task interrupts, with thread_id, city,
last_update and paused_at. delete_trip uses the saver delete_thread API to remove
checkpoints; exported files remain. The caller owns and closes saver.conn.

Runtime configuration carries thread_id and failure_mode, never credentials.
Runs use trip-{city} and tags travel-planner, city and thread:{id}. Initial parsing
uses pending until the city is known; subsequent agents and resumes use the
current city. The graph contains no node-level RetryPolicy.

Research Send branches run in parallel and coalesce into one merge per batch.
Only dispatched pools are replaced; an ignored stale round never erases existing
places. The merge deduplicates names, attaches extra source URLs, records failed
or empty branches, and clears both transient branch fields.

Mocked graph tests use fake agents, real LangGraph interrupts, actual temporary
SQLite files and local exports, including close/reopen restart coverage:
    python -m pytest -q tests/test_graph.py

Full mocked validation:
    python -m pytest -q
    python -m ruff check src tests
