# Five-minute demo script

Total narration and interaction: **5:00**. Provider waits are variable and are
not evidence of the latency target. Prepare approval/recovery pauses in advance,
keep a second paused trip for the failure section, and have the labeled mocked
sample bundles ready as a fallback. Say when using a prepared state or fixture.
Do not describe simulated approval as a real traveler's acceptance.

| Time | Segment | What to say and show |
| --- | --- | --- |
| 0:00-0:30 | Problem and one-liner | Read the opening paragraph of README.md. Call the blog-reading baseline a hypothesis and 4/5 under ten minutes a target, not an achieved result. |
| 0:30-1:15 | Architecture | Open docs/graph.mmd and the README node table. Point out five agents, four tools, the supervisor and two Send branches. Explain that merge replaces only dispatched categories and that save is a plain guarded writer. |
| 1:15-2:30 | Live workflow and delegation | In Streamlit, select None and start "1 day in Lisbon, relaxed, history"; include a near-term ISO date if showing weather. Show Orchestrator instructions, researcher tool calls and merge counts in the delegation/progress display. Explain that the two branches can run in parallel; their displayed event order alone is not a timing benchmark. Show Planner and Critic versions. If waiting, switch to a prepared live pause and disclose it. |
| 2:30-3:15 | Revision, approval, export | At approval, request "Move museums to the afternoon". Show the newer checked draft and sources, then Approve & export. Show the complete badge and Markdown download. Explain that the matching version and manifest are required. Use a prepared revised pause if needed; identify mocked samples explicitly if used. |
| 3:15-4:00 | Failure, recovery, retry | Open the prepared recovery trip created with tavily mode. Show reason, warnings and allowed options; no Approve button. Change mode to None and click Retry. Explain that Retry preserves attempt/step caps and may return to recovery when research attempts are exhausted. In that case show Revise with a valid changed request or New trip rather than promising Retry will finish. At the trip cap, only New trip/Cancel exist. |
| 4:00-5:00 | Codex and results | Open PROMPTS.md: state/gates, specialists, graph/restart, failure review, UI, evaluation and 403 diagnosis. Cite the two actual commits. Report 439 mocked passes and three skipped live tests; show the restart test name. Explain that no live eval CSV is present and 4/5 remains unmeasured. Point to the runner's simulated-workflow labels and distinguish LLM usefulness from human acceptance. Close with live evaluation, real samples and traveler feedback as next steps. |

## Preparation and honest fallbacks

1. Run the mocked tests and start `python -m streamlit run src/app.py`.
2. Prepare a normal approval pause, a revised approval pause and a tavily
   recovery pause in separate threads. Resume trip/Open switches among them.
3. Confirm credentials privately; never show .env, API keys or raw traces during
   screen sharing. Do not run the full paid evaluation during this five minutes.
4. Keep source links, the diagram and sample provenance visible. A `.test` URL
   belongs to mocked fixtures, not a real source.
5. For a restart demonstration outside the timed segment, stop/restart the app
   from the same directory and reopen a paused trip using Resume trip.
