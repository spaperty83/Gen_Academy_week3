# Google Doc outline: multi-agent city trip planner

This is an outline to paste into a Google Doc; no external document was created.

## 1. Overview and problem

- Quote README.md's opening one-liner verbatim.
- Traveler enters city, 1-7 days, optional start date, budget, interests and pace.
- Five agents research, plan and critique; traveler revises/approves before export.
- Treat 3-4 hours saved as a proposed baseline and 4/5 checked exports in under
  ten minutes as the success target. Neither is measured in this repository.

## 2. Agent framework table

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

## 3. Data used and provenance

| Data | Use | Limits |
| --- | --- | --- |
| Tavily search snippets and optional extracted pages | Candidate places, names, quotes and URLs; checkpoint source registry | Textual provenance, not proof of opening hours, price or availability |
| Open-Meteo geocoding and forecast | Weather notes with a start date; rain-aware planning | No invented forecast beyond the window or on failure |
| tests/eval_cases.json | Five normal trips with one revision each, plus a vague request | Small, scripted set; preliminary, no human acceptance measurement |
| Mocked responses in tests | Reproducible boundary/failure/approval checks | Synthetic fixtures cannot establish live accuracy or latency |

No embeddings, ingestion pipeline or vector database are used. Evidence lives
with each trip's checkpoint. Explain API tracing and checkpoint retention without
including raw keys or private request data in the submission.

## 4. Architecture

- Insert docs/graph.mmd and README.md's table distinguishing agents/plain nodes.
- Orchestrator: compact-summary supervisor, no operational tools.
- Parallel sights/food Send branches: own-result writes, current-round merge,
  category replacement and normalized-name deduplication.
- Planner: accepted places only; weather tools; coverage and pace rules.
- Critic: deterministic hard checks plus bounded soft web checks.
- Versioned human-review gate, recovery and guarded save/export-failed paths.
- SQLite thread persistence; public graph APIs used by Streamlit.

## 5. Vibe-coding prompts and Codex use

- Link PROMPTS.md; quote representative requests rather than inventing a process.
- Examples: state/coverage/gates; specialist tool budgets and boundaries;
  supervisor revision invalidation; graph restart; sixteen failure scenarios;
  Streamlit public-API UI; live evaluation and safe 403 diagnosis.
- Explain the developer's specification, Codex implementation, mocked validation
  and revisions. Test passing is evidence of specified behavior under mocks.

## 6. Implemented iterations supported by this repo

Git has two implementation commits. Prompt entries below are finer-grained work
records, not additional Git commits.

| Recorded evidence | What actually changed | Evidence/validation |
| --- | --- | --- |
| PROMPTS.md state/coverage/gate entry, 2026-11-04 | Added typed models, Overwrite-aware reducers, coverage and gate logic | State/coverage/gate tests; recorded 178 passes at that stage |
| Specialist entry, 2026-11-04 | Added create_agent specialists, bounded tool runtime and deterministic checks | Recorded 231 mocked passes |
| Orchestrator entry, 2026-11-04 | Added supervision, route guardrails, patch-first revisions and counters | Recorded 291 mocked passes |
| Graph entry and commit 4e4ac78, 2026-11-04 | Added Send/fan-in, SQLite interrupts, guarded export and CLI; committed initial implementation | Recorded 324 passes and SQLite close/reopen test |
| Failure review entry, 2026-11-07 | Fixed fallback, branch-local receipts, merge evidence validation, uncertainty, coverage dedupe and injection-pattern rejection | Exactly 16 failure cases; recorded 343 mocked passes |
| Streamlit entry, 2026-11-07 | Added cached resources, pause-based screens, versioned actions, reopening and downloads | Recorded 366 passes; recorded headless startup health HTTP 200 |
| Evaluation/diagnostic entries and commit 7b17d7f, 2026-11-07 | Added six live eval cases, simulated pause loop, judges, CSV reporting, safe HTTP diagnostics and read-only access command; committed UI/failure/eval changes | Current 385 mocked passes, three live skipped; actual read-only LangSmith request returned 403 |
| Current documentation work | Reorganized README, timed demo, this outline and two copied mock export bundles with valid hashes | Sample bundles are mocked; no new live research or acceptance claimed |

### Design decisions, not implemented migrations

- No RAG: part of AGENTS.md's design; history does not show a RAG implementation
  being removed. Describe any earlier discussion about dropping it as design only.
- Nebius, Tavily, Open-Meteo, LangChain/LangGraph and SQLite are the recorded stack.
  Do not claim a migration from another model provider or storage system.
- Native model retries and tool-scoped network retries are design constraints;
  do not invent a previous production incident involving stacked retries.

## 7. Results and evaluation

- Report current mocked suite: 428 passed, three live skipped; Ruff passed.
- Name restart coverage:
  tests/test_graph.py::test_sqlite_restart_lists_pending_trip_and_resumes_export.
- Link all sixteen failure rows in README.md; distinguish hard versus soft and
  show that caps never authorize hard-failure approval.
- No outputs/eval_results.csv is available at documentation time. The live
  experiment is blocked by recorded LangSmith 403; no 4/5, latency or usefulness
  result can be reported as achieved.
- When available, insert the dated CSV, experiment URL and labels SIMULATED
  WORKFLOW (live APIs, automatic approval) and five-case preliminary.
- Success = complete approved JSON export AND passing hard checks AND applied
  revision (Nebius judge) AND active agent time <600 seconds.
- Show secondary first-draft timing, delegation and usefulness separately.
  Usefulness 1-5 from an LLM and scripted approval are not human acceptance.
- Link outputs/samples/README.md. The two manifest-valid mocked bundles illustrate
  export format and revision behavior, not live destination recommendations.

## 8. Learnings, limitations and next steps

- Code guards constrain what model decisions can do; evidence receipts permit
  rechecking after a restart without retrieving a new page.
- Category-specific merge preserves independent work; optional extraction can
  fall back to snippets, and unavailable weather stays an explicit uncertainty.
- Strict coverage and capped retries can end in recovery; Retry is not a cap reset.
- Manifests establish completeness/integrity, not venue accuracy.
- Resolve LangSmith workspace permissions, run the preliminary live experiment,
  collect actual traveler-approved samples and measure human acceptance/time saved.
- Add wider destinations, routing feasibility, current opening-hour checks and
  deployment/retention controls. No booking/payment/contact actions are in scope.
