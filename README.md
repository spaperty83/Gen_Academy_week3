# Travel planner

Python 3.11+ project following AGENTS.md. Configuration, state models, coverage calculations, approval gates, tools,
evidence validation, exports and live provider checks are implemented. All agent nodes, the checkpointed workflow and the Streamlit UI are implemented.

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



## Streamlit app

From the project root, with the virtual environment activated and .env filled in:

```powershell
python -m streamlit run src/app.py
```

Open the local URL printed by Streamlit (normally http://localhost:8501).
The graph and SQLite checkpointer are created once with st.cache_resource.
Browser session state holds the current thread and screen; checkpoints.sqlite
keeps paused trips across app/server restarts. New trip always uses a fresh UUID.

The sidebar lists paused trips as city, last update and pause kind. Open loads
that saved thread without starting or resuming work. Demo failure mode is passed
per action as runtime config: None, extract, tavily, weather or llm_json. Include
a start date to exercise weather calls. llm_json makes one initial draft parse
fail so the single re-ask can repair it.

While work runs, all action buttons are disabled and st.status streams one line
per update/tool call. Queries come from actual tool-call events, not reconstructed
source text. A normal rerun or browser refresh never replays a start or approval.
Persisted delegation decisions can be reopened; detailed tool progress is transient.

Approval displays the numbered, Critic-checked draft, soft findings, sources,
budget preference considered, and the delegation log. Approve & export sends
the exact displayed draft version. Request changes and Cancel resume the same
thread. Recovery displays only the backend options and never has an Approve
button. At the trip cap those options are only New trip and Cancel.

Export failure retains approval and offers Retry export / Cancel. Done shows the
export path, but the complete badge and Markdown download require a valid on-disk
manifest and both current file hashes. A metadata-only or subsequently modified
export gets no badge or download. Cancellation shows Trip cancelled.

The UI imports only public functions from src/graph.py. The added read-only APIs
get_trip(graph, thread_id) and get_export(graph, thread_id) open a saved pause and
return validated download bytes respectively. stream_trip adds progress chunks
for actual tool calls while preserving the native __interrupt__ format. No
credentials or raw provider/database exception text are shown in the UI.

### Two-minute manual check

Allow about two minutes of interaction, excluding model/search response time.

1. **Happy path + one revision:** Choose Demo failure mode None. Plan
   "1 day in Lisbon, relaxed, history". At Draft v1, review the sources and
   soft findings. Request changes: "Move museums to the afternoon", then Submit.
   Confirm a newer version in the same trip; Approve & export, check the complete
   badge, and download the .md.
2. **Recovery:** Choose New trip, select demo mode tavily and plan another short
   Lisbon trip. Confirm recovery shows its reason, warnings and only allowed
   actions, with no Approve button. Cancel and confirm Trip cancelled, then
   restore Demo failure mode to None.
3. **City-change revision:** Start a fresh Lisbon trip. Request changes:
   "change Lisbon to Paris". Confirm the new draft says Paris and the old
   Lisbon stops are gone. Leave this draft paused at approval.
4. **Restart and resume:** Stop Streamlit with Ctrl+C, rerun the command above,
   then choose the Paris trip under Resume trip and Open. Confirm the same
   version reappears. Approve & export, check complete, and download Markdown.

An exhausted research round still needs Revise to reset its attempts; Retry does
not bypass research/step caps. Retry export is a separate action and consumes no
Orchestrator steps.

Mocked UI and public-API checks (no keys or live providers):

```powershell
python -m pytest -q tests/test_app.py tests/test_app_api.py
```

The AppTest suite covers caching, all phases, same-thread revisions and approval,
stale versions, pause reopening, disabled controls, interruption recovery,
downloads and friendly errors. API tests use real graph runs, local file hashes,
actual tool progress and a reopened SQLite file.

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
are extract, tavily, weather and llm_json. Filesystem export failures are
tested by mocking disk operations, not by adding another demo mode.

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

Each researcher writes only its own ResearchResult. Its sources and warnings
travel inside that result; merge_research publishes shared evidence and warnings
once, and revalidates candidates before replacing dispatched categories.
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

start_trip and resume_trip return state plus thread_id and screen. The screen
contains kind, options and warnings for pauses and terminal outcomes; it is a
view of state, not additional checkpoint state. Native pause entries are
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


## Failure behavior review

The review covers approval/version guards, agent boundaries, parallel merge,
coverage, evidence, revisions, atomic export, retry placement and untrusted data.
The existing approval and revision rules remain enforced. Schema instances are
revalidated, coverage cannot count one venue in two pools, and export manifests
must match the filename's draft version as well as both file hashes.

Optional page extraction can fail while successful search snippets remain usable.
Research stays branch-local until merge; successful sibling results survive a
failed branch. Critic spot-check queries/pages are restricted in code to at most
three selected stops and two extracted pages. Missing weather and failed checks
always become soft uncertainty, even if the model omits them.

Source text stays untrusted inside <tool_data>. Instruction-like passages cannot
support an evidence quote, including after checkpoint restart. The injection test
deliberately makes the fake model add Place X and checks that validation prevents
it from entering places, the itinerary or the exported JSON. Raw source text can
remain in the evidence registry for provenance; storing it does not authorize it.

Demo failure_mode values are exactly extract, tavily, weather and llm_json and are
read only from runtime configurable settings. llm_json simulates an initial
Planner parse failure; the one re-ask can recover. The two-invalid-output tests
mock bad model responses to exercise each agent's exhausted validation boundary.
Network retries remain inside read tools; model retries remain native ChatOpenAI
retries. There are no node-level retries and no retry wrapper around the model.

All tests in the table are in tests/test_failures.py. They execute real agents,
native structured parsers, tools, graph routes, interrupts and local exports with
mocked models/HTTP providers and no keys. Each asserts the resulting screen kind,
available options and warning. Some tests contain both requested variants.

| Failure | Behavior | Test | Mocked/live |
| --- | --- | --- | --- |
| 1. Page extraction fails | Uses successful snippets; approval warns about fallback. | test_01_extract_failure_uses_snippets | Mocked |
| 2. One researcher fails | Keeps sibling places and warns; only failed focus is dispatched once more. | test_02_one_failed_researcher_keeps_other_and_redispatches_once | Mocked |
| 3. Coverage stays short twice | Recovery offers Revise / Retry / Cancel; Planner never runs. | test_03_short_after_two_attempts_recovers | Mocked |
| 4. Tavily unavailable, zero places | Recovery with research warnings and Retry available. | test_04_tavily_down_zero_places_offers_retry | Mocked |
| 5. Weather down or beyond forecast | Soft finding, unavailable forecast and visible warning; approval remains possible. | test_05_weather_down_or_beyond_window_is_soft | Mocked |
| 6. Critic web check fails | One tool-uncertainty finding; Python hard failures still force recovery. | test_06_critic_web_failure_is_soft_but_hard_checks_remain | Mocked |
| 7. Invalid place evidence | Research candidate is dropped; invalid draft evidence is a hard failure. | test_07_invalid_evidence_is_dropped_and_draft_is_hard_failure | Mocked |
| 8. Invalid Orchestrator choice | Unsupported route or unmet precondition is replaced and logged with guardrail warning. | test_08_invalid_orchestrator_choice_is_replaced_and_logged | Mocked |
| 9. Round, trip or correction cap | Hard failures recover; soft-only checked drafts can reach approval with warnings. | test_09_caps_never_force_hard_approval_soft_caps_warn | Mocked |
| 10. Invalid output twice | Recovery names Orchestrator, Planner or Critic and keeps completed state. | test_10_invalid_output_twice_names_agent_and_keeps_completed_state | Mocked |
| 11. Trip cap | Only New trip / Cancel; Retry is rejected without consuming more steps. | test_11_trip_cap_only_new_trip_cancel_and_rejects_retry | Mocked |
| 12. Vague city | Clarification screen warns and offers a fresh trip or cancel; no places invented. | test_12_vague_city_asks_for_clarification | Mocked |
| 13. Stale or unchecked approval | No export; current approval pause or recovery repeats with warning. | test_13_stale_or_unchecked_approval_never_exports | Mocked |
| 14. Disk failure between files | Export-failed pause retains approval; retry repairs both files and commits manifest last without consuming steps. | test_14_export_between_files_fails_then_retry_repairs_manifest | Mocked |
| 15. City changes | New city is applied before invalidation; stale old-city candidates are rejected. | test_15_city_revision_cannot_restore_old_places | Mocked |
| 16. Web instruction to add Place X | Instruction text cannot authorize a place or stop; export contains no Place X. | test_16_instruction_in_web_result_cannot_become_a_place | Mocked |

Run just these 16 end-to-end cases:
    python -m pytest -q tests/test_failures.py

Normal recovery retains the two-attempt limit within a round: Revise opens a new
round; Retry clears an agent failure but does not reset research attempts or
caps. At the trip cap, Retry is not offered. Clarification is a terminal screen:
start a new thread for the corrected request rather than resuming that thread.

Real-provider integration tests remain in tests/test_tools_live.py and are opt-in;
the 16 rows above do not claim live-provider coverage.


### LIVE evaluation (automatic approval)

Run `python -m scripts.run_eval` from the repository root with Nebius, Tavily,
and LangSmith credentials in `.env`. This uses paid live APIs and exports real
files while simulating the traveler: one scripted revision, then approval of a
matching, Critic-checked version. Recovery is cancelled; an export failure gets
one retry. Each case uses a fresh in-memory checkpointer and at most six pause
responses. No Streamlit server is needed.

The runner creates `travel-planner-eval` in LangSmith, appends only missing exact
examples, and evaluates exactly the six cases in `tests/eval_cases.json`.
Five normal trips include Porto and a seven-day packed trip; the sixth request
is deliberately vague. Results include the experiment URL and a dated
`outputs/eval_results.csv`, labeled **SIMULATED WORKFLOW (live APIs, automatic
approval)** and **5 cases — preliminary**.

A normal trip succeeds only when its complete export matches the approved JSON,
Python hard checks pass, the Nebius judge confirms the revision, and active agent
time is below 600 seconds. The target is 4/5. The 300-second first-draft measure,
delegation check, and usefulness score (1–5) are secondary; usefulness pass rates
use an explicitly secondary threshold of 3/5. Missing judge scores fail the
required revision check. Timing excludes dataset operations and judging; first
draft time is captured when the Planner first produces an itinerary. The vague
case must finish with clarification and is reported separately.

MOCKED runner checks: `python -m pytest -q tests/test_run_eval.py`.
The live experiment is performed only when you invoke the command above.


LangSmith setup troubleshooting: `python -m scripts.run_eval --check-langsmith`
performs only a dataset read check. HTTP 401 means authentication was rejected;
HTTP 403 means the request is forbidden: check the key's dataset/experiment
permissions and workspace access. Set `LANGSMITH_WORKSPACE_ID` in `.env` when
workspace selection is required, and check `LANGSMITH_ENDPOINT` matches your
account region. Existing shell environment values take precedence over `.env`.
The check does not validate write permissions or start the paid workflow.
