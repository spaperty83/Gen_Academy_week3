# City trip planner

My agent helps a traveler plan a city trip in a Streamlit web app, replacing the 3–4 hours of reading travel blogs and juggling tabs it takes today. Five cooperating agents research places, check the weather, build a day-by-day itinerary and critique it on their own using 4 tools, then hand off to the traveler to approve or revise before the final itinerary is exported. I'll know it works when a traveler reaches an approved itinerary that passes every hard check in under 10 minutes for 4 out of 5 test trips.

This is the product hypothesis and success target. The 3–4 hour baseline has not been measured here, and the 4/5 live target has not been demonstrated. Current evidence is **434 mocked tests passing, three live tests skipped**. No live evaluation CSV is present; the most recent recorded LangSmith access check returned HTTP 403.

## Agent framework

The field names below follow **Part 2: The Agent Framework**, page 8 of the Week 3 Project Handout (Oct 2026). Each answer describes this project's implementation in one or two sentences.

| Field | Fill in (1 to 2 sentences max) |
| --- | --- |
| Agent goal (one line) | Help a traveler reach a source-grounded, Critic-checked city itinerary that they can revise and approve before export. |
| Where do people use it? | A Streamlit web app, with a CLI alternative for the same checkpointed workflow. |
| What steps does it take, in order? | 1. Parse the request; 2. research sights and food in parallel and merge accepted evidence; 3. check weather and draft; 4. critique and send back findings as needed; 5. pause for traveler revision or approval; 6. export only the approved checked version. |
| What can it actually do? | READ: web_search and read_page retrieve Tavily evidence, and get_weather retrieves Open-Meteo forecasts. WRITE: save_itinerary exports local JSON/Markdown plus a manifest, exclusively from the guarded save node after approval. |
| What does it need to remember? | Per-thread SQLite checkpoints retain current/original requests, places, source receipts, drafts, critiques, versions, counters and the pending pause across sessions until the trip is deleted. Exported files and remote traces have separate lifecycles; secrets are excluded from workflow state. |
| What should it never do? | Never book, pay, contact anyone, invent places or export an unapproved draft; never offer approval with hard failures. Keep credentials out of logs, checkpoints and traces. |
| Human-in-the-loop | A checked draft pauses for Approve, Request changes or Cancel before any itinerary export. Recovery offers only allowed repair/cancellation options, while export failure permits Retry export or Cancel without discarding approval. |
| What happens when something breaks? | Tools use bounded transient-error retries; optional extraction and unavailable weather can degrade to explicit warnings, while agent failures or insufficient coverage reach recovery with completed state preserved. Caps never force approval, and failed exports are repaired through a separate retry pause. |
| How do you know it worked? | Target: at least 4/5 normal test trips produce complete approved exports that pass every hard check and apply the scripted revision in under 600 seconds. This target remains unmeasured live; simulated approval and LLM usefulness scores do not establish real human acceptance. |

## Architecture

[Generated LangGraph diagram: docs/graph.mmd](docs/graph.mmd). Regenerate it without API keys:

```powershell
python -m src.graph --diagram
```

START enters the Orchestrator. `Send` dispatches one or both researchers; their branch results join at `merge_research`, then return to the supervisor. Planner and Critic also return to the supervisor. The code gate routes a checked draft to human review and other states to recovery. Approval can reach save; failed writes pause separately for export retry.

| Node | Agent or plain node | Responsibility and tool access |
| --- | --- | --- |
| orchestrator | Agent 1: supervisor | Parse request, choose next worker, issue coverage-sized instructions; no operational tools |
| sights_researcher | Agent 2 | Research sights with web_search/read_page |
| food_researcher | Agent 3 | Research food with web_search/read_page |
| planner | Agent 4 | Use accepted places; get_weather only when a start date exists; produce a new draft |
| critic | Agent 5 plus Python checks | Hard checks, then bounded web_search/read_page spot-checks that add soft findings |
| merge_research | Plain | Validate current-round receipts, replace dispatched categories, deduplicate, publish evidence and warnings |
| human_review | Plain interrupt | Display checked draft; approve matching version, revise or cancel |
| recovery | Plain interrupt | Explain failure and offer only allowed recovery options |
| save | Plain writer | Recheck approval guard and invoke save_itinerary |
| export_failed | Plain interrupt | Retry export or cancel while retaining approval |
| end_cancelled | Plain | End the thread as cancelled |

The four tools are `web_search`, `read_page`, `get_weather` and `save_itinerary`. Only the save node receives the write tool. UI actions use the public [graph API](src/graph.py), including `stream_trip`, `get_trip`, `get_export`, `list_trips` and `resume_trip`.

## Why five agents

The supervisor sees a compact summary and delegates to specialists with separate tools and budgets. Sights and food research can run in parallel. Each branch writes only its own ResearchResult; one merge replaces only dispatched categories, so food-only research preserves sights. Results from an old research_round are ignored; normalized names are deduplicated and extra URLs retained.

The Critic separates Python hard checks from model judgment. Findings have an owner (researchers or planner), allowing send-backs before the exact new draft is checked again. Python guardrails replace invalid routes and log `guardrail:` reasons; prompts alone do not authorize planning or export.

Coverage controls research size rather than relying on a fixed handful of suggestions:

| Quantity | Rule |
| --- | --- |
| Stops/day | relaxed 3; normal 4; packed 5 |
| Food/day | 2 when food is an interest, otherwise 1 |
| Needed per category | Per-day count times trip days; other stops belong to sights |
| Research target | ceil(needed times 1.3) |
| Tool budget per researcher | min(14, max(8, ceil(target / 3))) |

A seven-day packed food trip needs 21 sights and 14 food places; targets are 28 and 19, giving tool budgets 10 and 8. The Planner cannot run until both pools meet needed coverage. [Coverage tests](tests/test_coverage.py) check these rules. This architecture's speed and quality advantage over a single agent has not been measured.

## Approval, recovery and export

Approval requires a draft, `critique_version == draft_version`, no hard failures and no agent_error. Hard failures include invalid schema, a stop outside accepted places, invalid evidence, wrong day count, pace-limit violations or duplicate stops. Soft findings include unavailable weather, uncertain spot-checks, thin interest coverage or a possibly unmet budget preference. Soft findings remain visible and can coexist with approval.

Caps are 12 supervisor steps per revision round, 30 per trip, two correction cycles per round and two dispatch attempts per researcher per round. **Caps never force approval.** The gate still decides whether a checked draft can be shown. Recovery normally offers Revise / Retry / Cancel; at the trip cap it offers only New trip / Cancel. Retry clears an agent error and forwards findings; it does not reset exhausted research attempts or caps. Revise resets round limits. Expected and unexpected agent failures become sanitized agent_error updates; researchers return failed results so a sibling's work survives.

Both save and save_itinerary enforce `decision == approve` and non-null `approved_version == draft_version == critique_version`, with no hard failures. Stale or unchecked approval cannot export. JSON and Markdown are staged, renamed, then the SHA-256 manifest is written last. A lone file is incomplete. Retry repairs incomplete bundles; export retries consume no supervisor steps. Destinations must remain inside outputs/. The UI's complete badge/download require a valid current bundle matching the approved itinerary.

## Evidence and its limits

Only real tool-response messages in a workflow populate the checkpoint's source registry: source_id maps to URL, retrieval timestamp and at most 4,000 characters of text. Candidates need a registered source_id, an exactly matching source URL, a quote containing the normalized name, and a whitespace-normalized quote found in the saved source text. Name matching strips accents, punctuation and generic venue words. [Evidence tests](tests/test_evidence.py) exercise forged IDs, URLs, names and quotations.

This proves textual provenance against a retrieved response. It does not establish that a venue is open, prices are current, reservations are available, travel distances are practical, or the source is truthful. Use the wording **budget preference considered**. The export disclaimer says: "Places were checked against their source pages; opening hours and prices may have changed."

Tool text is untrusted and wrapped in escaped `<tool_data>`. Instruction-like evidence is rejected; the mocked Place X injection scenario checks it never reaches accepted places, stops or exported JSON. That regression test is not a general guarantee against every prompt injection. There is no vector store, embedding pipeline or RAG. Checkpoints retain request and evidence data per thread until deletion; never put keys in state. Deleting checkpoints does not delete exports or any remote traces.

## Revisions

A structured RequestPatch is validated and applied to current request before dependent state is cleared. original_request stays unchanged. Invalid patches only add a rephrase warning. [Revision tests](tests/test_orchestrator.py) include Paris replacing Lisbon, dates, length, interests, budget and pace.

| Change | Cleared/refreshed |
| --- | --- |
| City | All places, branch results, source registry, accumulated warnings, weather, itinerary, critique and approval; research both new-city categories |
| Start date or days | Weather, itinerary, critique and approval; replan, with coverage guards still enforced |
| Interests | Affected pool's places plus itinerary, critique and approval; food-interest edits refresh food, other interest edits refresh sights |
| Budget | Both category pools plus itinerary, critique and approval |
| Pace or stop order | Itinerary, critique and approval; retained places may need additional research to meet new coverage |

Valid revisions reset round_steps, correction_cycles and researcher_attempts; trip_steps and draft_version never decrease. City changes use LangGraph Overwrite to clear accumulated registry and warnings. Notes remain in human_feedback for subsequent work.

## Setup, run and resume

Python 3.11+; run from the project root. PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Copy only when creating .env initially; preserve an existing credential file. Fill in NEBIUS_API_KEY, NEBIUS_BASE_URL, NEBIUS_MODEL (tool calling and structured output required), TAVILY_API_KEY and, for tracing/evaluation, LANGSMITH_API_KEY. LANGSMITH_TRACING and LANGSMITH_PROJECT configure tracing. Keys are validated when a service is used. Existing shell variables take precedence over .env. Credentials and generated SQLite files are ignored by Git.

```powershell
python -m streamlit run src/app.py
```

Open the URL Streamlit prints, normally http://localhost:8501. Try "1 day in Lisbon, relaxed, history". Add a real ISO start date to request weather. Review sources and soft findings, request a change, then approve the displayed version. The graph/checkpointer are cached once; buttons are disabled during work. New trip creates a fresh thread UUID.

**Resume trip:** leave a trip paused, stop the app, restart it from the same project directory, select its city / last update / pause kind in the sidebar and click Open. SQLite checkpoints.sqlite restores the same pending thread. Opening is read-only; an action resumes it. Clarification ends the vague thread; start a new trip with the corrected request.

CLI alternative:

```powershell
python -m src.graph "1 day in Lisbon, relaxed, history"
```

## Error handling (Prompt 7)

The following 16 cases are in [tests/test_failures.py](tests/test_failures.py). They use the real agent/graph control flow with fake models and HTTP, and assert screen kind, options and warnings. They do not establish live-provider reliability.

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

Read tools use 15-second timeouts and up to two retries for timeouts, 429 and 5xx, never 400/401. Native ChatOpenAI uses max_retries=2 and timeout=60; no model with_retry wrapper or node RetryPolicy. Invalid structured output gets one re-ask, then an agent boundary failure. read_page is optional and limited to 1–2 calls when snippets lack names/areas. Planner and Critic each have at most four external tool calls.

Demo modes are None, extract, tavily, weather and llm_json. Extract may fall back to snippets; weather is soft; llm_json causes one initial parse failure and may recover via re-ask. Tavily failure can reach recovery. Switch a failed mode back to None before retrying. A round with exhausted research attempts needs Revise or a new trip; Retry does not bypass the cap.

## Testing

MOCKED is the default: fake providers, no keys required. LIVE is explicit and uses network/provider credits.

```powershell
python -m pytest -q
python -m ruff check src tests
python -m pytest -q tests/test_failures.py tests/test_app.py tests/test_app_api.py tests/test_run_eval.py
```

The most recent full run passed 434 tests with three live tests skipped. [tests/test_graph.py](tests/test_graph.py), `test_sqlite_restart_lists_pending_trip_and_resumes_export`, uses a temporary SQLite FILE: pause, close the connection, rebuild saver/graph, list the pending trip, approve and validate the export. UI tests also exercise cached resources, versioned actions, downloads and friendly errors. These are mocked functional checks, not measured human acceptance.

```powershell
python -m scripts.smoke_test
$env:RUN_LIVE = "1"
python -m pytest -q tests/test_tools_live.py
Remove-Item Env:RUN_LIVE
```

The environment assignment above is PowerShell syntax; Command Prompt uses `set RUN_LIVE=1`. The smoke test checks Nebius answers, tool calling, structured output and Tavily search/extract. Real read-tool tests are opt-in.

## Evaluation and results

```powershell
python -m scripts.run_eval --check-langsmith
python -m scripts.run_eval
```

The first command is a read-only LangSmith access check. HTTP 401 indicates rejected authentication; 403 indicates forbidden access. Check key permissions, workspace membership, optional LANGSMITH_WORKSPACE_ID and the region configured by LANGSMITH_ENDPOINT. A successful read check does not establish write permissions. The last recorded check returned 403; no completed live experiment is claimed.

The live runner synchronizes travel-planner-eval without duplicating exact inputs and evaluates five normal trips (Lisbon, Porto, Paris, Kyoto, Rome) plus a vague request. Each normal trip has one fitting scripted revision. It uses a fresh in-memory checkpoint, inspects pause kind, revises the first approval, approves a subsequent matching checked version, cancels recovery, retries export once, and caps pause responses at six.

A normal case succeeds only if the complete on-disk export equals the approved JSON, hard checks pass, the Nebius judge says the revision was applied, and agent time is below 600 seconds. First draft below 300 seconds and delegation are secondary. Usefulness is separately judged 1–5; an optional secondary pass rate uses >=3. A model's rating or simulated approval **is not real human acceptance**. Timers exclude dataset setup and judging.

The runner writes `outputs/eval_results.csv` with date, experiment URL, per-case outcomes, pause kinds, timing, usefulness, failure reasons and summary rates. It labels results **SIMULATED WORKFLOW (live APIs, automatic approval)** and **5 cases — preliminary**. This file is currently absent; no average time, usefulness score or 4/5 success is reported as measured. Upload or share the generated CSV only after a real run. Human review and a larger varied set remain future work.

## Sample itinerary bundles

No live exports existed when this documentation was prepared. These two bundles were produced by the existing mocked graph harness, passed its simulated versioned approval, and were copied byte-for-byte with manifests. **They are synthetic test fixtures, not destination recommendations or evidence of live research.** Their `.test` source links and fixture venue names are deliberately artificial. See [sample provenance](outputs/samples/README.md); an inherited export disclaimer does not turn fixture receipts into real source pages.

| Mocked sample | Markdown | JSON | Manifest |
| --- | --- | --- | --- |
| Lisbon, first checked draft | [Itinerary](outputs/samples/mock_lisbon_v1.md) | [Data](outputs/samples/mock_lisbon_v1.json) | [Hashes](outputs/samples/mock_lisbon_v1.manifest.json) |
| Paris after city revision | [Itinerary](outputs/samples/mock_paris_revision_v2.md) | [Data](outputs/samples/mock_paris_revision_v2.json) | [Hashes](outputs/samples/mock_paris_revision_v2.manifest.json) |

Replace or supplement these with 2–3 actual traveler-approved live bundles when available; copy all three files together and preserve their bytes and `_vN` filename version.

## How I used Codex

[PROMPTS.md](PROMPTS.md) records requests and one-line outcomes. Work progressed through state/coverage/gates, specialists, supervision/revisions, graph persistence, a failure review, Streamlit and evaluation. Git records an initial implementation (`4e4ac78`, 2026-11-04), then UI/failure/evaluation changes (`7b17d7f`, 2026-11-07). Prompt-log entries provide finer detail than those two commits; they are not separate commits.

Codex wrote and revised code, ran mocked checks, and added diagnostics after the observed LangSmith 403. Model suggestions were constrained by Python checks and tested failure scenarios. The decision to omit RAG is part of the design; the history does not show implementing and then removing it. See the [Google Doc outline](docs/GOOGLE_DOC_OUTLINE.md) and [five-minute demo script](docs/DEMO_SCRIPT.md).

## Limitations and next steps

- Source evidence establishes provenance, not opening hours, current prices, booking availability or geographical feasibility. Weather is unavailable without a start date or outside the forecast window.
- Live provider quality/latency, the 4/5 target, time saved and human acceptance remain unmeasured here. Fix LangSmith access, run the preliminary evaluation and collect traveler feedback.
- Coverage, budgets and correction caps bound work but may stop a difficult trip in recovery. Larger trips may need more source diversity or revised coverage policy.
- Injection defenses cover tested patterns, not all hostile content. Review checkpoint/trace retention and access before shared deployment.
- SQLite and local exports serve the current app; multi-user deployment, authentication, distance-aware routing and current opening-hour checks are future work. Pin tested dependency versions before distributing a reproducible release.
- Never book, pay or contact anyone. The traveler retains approval control in the app; the live evaluation uses explicitly labeled simulated approval.


### Repairing research evidence failures

Research candidates now receive the existing single validation re-ask when their
IDs, URLs or exact quotes do not match their own tool receipts. The correction
includes rejection codes and the retained source text inside untrusted tool_data;
it shares the original tool budget. Candidates still invalid after that correction
are dropped, with reason counts in the warning. When minimum category coverage is met and no draft exists, another research
choice is redirected to Planner rather than exhausting attempts chasing targets.
This does not relax quote matching,
injection checks, coverage, approval or the two-dispatch cap.

Search snippets shown to the model are capped at the same 4,000 characters saved
in the checkpoint, preventing quotes from an unseen-to-validation tail. Restart
Streamlit after updating the code. For a previously exhausted trip, use New trip,
or a valid revision to reset round attempts; Retry alone does not reset them.

Mocked regression: tests/test_research_recovery.py checks that an incorrect first
sights quote can be repaired within one dispatch and reach a checked approval
pause. A targeted live sights-only diagnostic accepted three places without warnings.
A subsequent live "1 day in Lisbon, relaxed, history" run reached approval with
three sights, two food places and draft_version=critique_version=1, with no
evidence-drop warnings or agent error. It stopped without approval or export;
this single workflow is not a reliability or evaluation success-rate claim.


### Critic tool-budget failures and Retry

A saved app failure was diagnosed as specialist_step_limit_reached in Critic,
with an existing draft and accepted places preserved. At the external tool budget,
the runtime now removes read tools from the next model request while retaining
the structured-response tool, and asks the model to finish with any uncertainty.
After two extracted pages, read_page is removed while search remains available.
A model that ignores these restrictions still fails closed into recovery.

Retry after a Critic error now clears critique_version as well as agent_error,
so a successful new check is required for approval of the saved draft. Restart
Streamlit, reopen that trip with Resume trip / Open, then select Retry when offered.
The trip cap still permits only New trip / Cancel. A rejected candidate, a
coverage-met routing message and unavailable weather are not by themselves fatal;
agent_error is the blocker in the reported scenario.

Mocked checks in tests/test_critic_recovery.py and tests/test_agents.py cover
four-call finalization, preserved hard failures and actual rechecking on Retry.
A live retest of the saved trip was not performed because automatic approval
review blocked sending checkpointed trip/source data to external providers
without specific permission. This fix has not been confirmed on that live trip.


### Revised drafts must be checked before further research

A local failed beach-interest revision showed draft v2 being sent back to sights
before Critic had checked it; the extra dispatch then exhausted research attempts.
Research choices for an existing unchecked draft now route to Critic first, while
agent errors and overall caps continue to take precedence. Researchers also receive
human_feedback alongside the updated request and are prompted to target new interests.
The mocked revision regression in tests/test_critic_recovery.py asserts v2 reaches a
matching Critic-checked approval without an unnecessary research dispatch, while
preserving original_request and food research. This is not a live quality claim
that any particular beach was found. Restart Streamlit before reopening and retrying
a saved trip; the trip cap still applies.


Approval-button integration is checked by tests/test_app_real_workflow.py: actual
Streamlit buttons drive the real graph through a revision and versioned approval,
with mocked providers and real local export files. The test requires a complete
manifest and download, and asserts approval invokes no agents and consumes no
supervisor steps. This does not reproduce or establish the cause of a separately
reported live approval error; that report still needs its exact current message.


On the exported screen, the export path, completion check and download appear
first. Weather and other travel advisories remain visible; rejected-candidate,
guardrail and reached-limit notices are retained under a collapsed Run log
section as run history. These notices do not indicate export failure. Incomplete
or modified bundles still receive no complete badge or download.


The supervisor cannot repeatedly dispatch Critic for an already approvable draft:
a repeated Critic choice now routes through the gate to traveler review, with soft
findings preserved. Agent errors, unchecked versions and hard failures do not use
this shortcut. A local saved Agra failure exposed several redundant checks of v1
before a later step-limit error; tests/test_critic_recovery.py reproduces that
supervisor pattern for both the initial draft and a revision with mocked providers.
This check avoids redundant work; it does not guarantee live provider reliability.


### Display-only city photos

At the draft approval screen, up to two city photos appear above the daily stops,
with captions and Image source links. They are city inspiration and may not depict
specific itinerary stops. Photo lookup uses Tavily's image results separately from
the four agent tools; it adds no model calls and cannot block the approval gate.
See [Tavily image result options](https://github.com/tavily-ai/skills/blob/main/skills/tavily-best-practices/references/search.md).

The public get_city_images(city) API is read-only and has no trip/checkpointer
argument. Streamlit caches the returned URLs in that browser session by city:
ordinary reruns reuse them; new searches, submitted revisions, new draft versions
and opening another trip clear old photos and trigger a fresh lookup when reviewed.
No image URLs, captions or image bytes are added to TripState, Itinerary, SQLite
checkpoints, JSON, Markdown or manifests. Photos use an extra Tavily search when
not already cached; a restarted browser session may search again. Missing keys,
failed searches and missing image results leave the itinerary available without
photos. External image hosts control loading and availability; source links are
retained, and images are not bundled in downloads.

MOCKED checks: tests/test_city_images.py and tests/test_app.py cover capped and
deduplicated results, URL filtering, attribution, city changes, cache reuse and
optional failures. tests/test_app_real_workflow.py displays mocked images through
the actual UI/graph while asserting they do not enter checkpoints or either export.
Live image selection/loading has not been tested as part of these checks.


Critic's four-call budget also applies within a single model-authored batch.
A saved Hyderabad failure contained three initial calls followed by two more;
the excess fifth call previously aborted the review. Optional checking now declines
that excess call without executing it, records uncertainty, and lets the next model
turn return its structured critique with no external tools exposed. A model that
keeps requesting external tools after finalization is still stopped; hard checks
and agent boundaries remain enforced. Mocked specialist and real-graph regressions
reproduce the exact three-plus-two pattern without approving or exporting.


Weather received from Open-Meteo is shown in each draft day and in a Weather for
your trip section after export. New weather notes include the forecast date, rain
probability and low/high temperatures when those values and units are returned.
New Markdown exports include each approved day's weather_note; JSON already
retained that field. No forecast values are invented when missing. Previously
completed export bundles are left unchanged; a newly generated/revised export
uses this rendering. A local successful Hyderabad trip with start date 2026-11-15
confirmed weather was retrieved but its earlier Markdown renderer omitted the note.


Opening a failed trip is read-only; clicking Retry is a separate action. If a pool
is still below minimum coverage after both research attempts, recovery explains
its available/required counts and that Retry cannot reset the limit. Such a Retry
returns to the pause without model calls or supervisor steps, preserving completed
work. Use a meaningful revision (for example fewer days or a different pace) to
open a new round, or use New trip. Other recoverable agent errors keep their normal
Retry behavior. Research output below minimum coverage now receives the existing
one correction opportunity with remaining tool budget; it cannot invent places or
bypass the final coverage gate. Mocked regressions cover both paths.


A saved Agra planning failure emitted five alternative Itinerary schema calls in
one response. Specialists now bind the native model with parallel_tool_calls=False
(each researcher can still run in parallel with its sibling). Multiple schema
outputs receive a specific instruction to return exactly one complete response;
draft day/stop violations receive the actual validation findings and required
counts during the existing single correction. Persistent invalid output still
reaches recovery with completed research retained. Mocked regressions reproduce
five alternative outputs and a corrected full graph run; live behavior is unverified.


Technical candidate-rejection, guardrail and reached-limit notices appear in the
collapsed Run log rather than above the reviewed draft or successful export.
Critic progress remains in that log too. Material soft findings (such as uncertain
source confirmation or missing weather), hard failures and active recovery reasons
remain visible so the traveler can make an informed decision.

The temporary planning status no longer prints detailed streamed lines above the
result. It is explicitly cleared when work completes; all detailed progress stays
in the collapsed Run log below the result.


Live search progress remains visible in an expanded Run log while work is running.
On completion its temporary panel is cleared. Dedicated result and log containers
keep the final itinerary/export above the collapsed Run log on the result screen.

The initial trip request uses a single-line field: press Enter or click Plan trip
to submit. The revision feedback field remains multiline.

Draft source links are grouped in a collapsed Citations section immediately
before Approve & export, rather than inside each day. Exports retain their source
links; photo attribution stays beside each displayed photo.


Researcher batches use the same capped finalization behavior as optional Critic
checks: an excess call is declined without execution, and successful search
receipts can still produce the final ResearchCandidates response. Budget declines
are not misclassified as Tavily outages; actual required-search failures still
fail the researcher. A saved Yellowstone failure showed successful Food Researcher
searches discarded when repeated batches exceeded eight calls. Mocked specialist
and real-graph regressions reproduce that sequence, retain evidenced food places
and reach review. Existing exhausted rounds still require a valid revision or New
trip; Retry does not reset their dispatch counts. Live reliability remains unmeasured.

The draft heading shows only its version; the Critic-check confirmation is kept
in run history rather than displayed above the result. Approval checks are unchanged.


Checkpoint model compatibility: models restored from an older module identity
(such as a cached graph surviving source reload) are converted to field data only
when their module and model name match the current schema, then fully revalidated.
Patch omission versus explicit null is preserved; invalid values and unrelated
models remain rejected. A saved Yellowstone modification exposed this validation
failure before the revision was applied. Restart the server after code updates;
that already-stopped attempt may require a new trip because no interrupt was saved.
