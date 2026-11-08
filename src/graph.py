"""Persistent travel workflow, interrupt-based review, public API and CLI."""

import argparse
import hashlib
import json
import logging
import sqlite3
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Send, interrupt
from pydantic import BaseModel, ValidationError

from src import export as export_files
from src import gate
from src.agents._runtime import envelope
from src.agents.critic import critic
from src.agents.orchestrator import apply_revision, orchestrator
from src.agents.planner import planner
from src.agents.researchers import make_researcher
from src.coverage import needed
from src.evidence import normalize_name, validate_places
from src.state import (
    MAX_CORRECTION_CYCLES,
    MAX_RESEARCHER_ATTEMPTS,
    MAX_ROUND_STEPS,
    MAX_TRIP_STEPS,
    AgentError,
    Critique,
    Place,
    ResearchResult,
    SourceRecord,
    TripState,
)
from src.tools import save_itinerary, search_city_images

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
FAILURE_MODES = frozenset({"extract", "tavily", "weather", "llm_json"})


class GraphState(TripState, total=False):
    """Workflow outcomes added without changing the shared specialist schema."""

    new_trip: bool
    cancelled: bool
    export_error: str | None


def _json(value: Any) -> Any:
    """Make interrupt payloads JSON-compatible without serializing config or clients."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    return value


def _city(state: Mapping[str, Any]) -> str | None:
    """Read the current city, including restored dictionary requests."""
    request = state.get("request")
    return request.get("city") if isinstance(request, Mapping) else getattr(request, "city", None)


def _config(thread_id: str, city: str | None, failure_mode: str | None = None) -> RunnableConfig:
    """Keep failure injection in runtime config only; allow enough graph supersteps."""
    if failure_mode is not None and failure_mode not in FAILURE_MODES:
        raise ValueError("Unsupported failure_mode")
    label = city or "pending"
    return {
        "configurable": {"thread_id": thread_id, "failure_mode": failure_mode},
        "run_name": f"trip-{label}",
        "tags": ["travel-planner", label, f"thread:{thread_id}"],
        "recursion_limit": 250,
    }


def _node_config(state: TripState, config: RunnableConfig) -> RunnableConfig:
    """Label child agents/tools with the real city once initial parsing completes."""
    configured = config.get("configurable", {})
    return {
        **config,
        "run_name": f"trip-{_city(state) or 'pending'}",
        "tags": ["travel-planner", _city(state) or "pending",
                 f"thread:{configured.get('thread_id', '')}"],
    }


def _agent_node(function: Callable, name: str) -> Callable:
    """Keep unexpected node errors inside the workflow without provider retries."""
    def node(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
        try:
            return function(state, _node_config(state, config))
        except Exception as error:  # noqa: BLE001 - final secret-safe agent boundary
            LOGGER.warning("%s node failed (%s)", name, type(error).__name__)
            update = {
                "agent_error": AgentError(agent=name, message="agent_failed"),
                "warnings": [f"{name} failed; completed state is retained for recovery."],
            }
            if name == "orchestrator":
                update["next"] = "recovery"
            return update
    return node


def _research_node(focus: str) -> Callable:
    """A research branch never writes another branch's result or global errors."""
    function = make_researcher(focus)

    def node(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
        try:
            result = function(state, _node_config(state, config))
            if result.get("agent_error") is None:
                branch = ResearchResult.model_validate(result[f"{focus}_result"])
                # Normalize older/custom branch adapters without allowing shared writes.
                sources = branch.sources | {
                    key: SourceRecord.model_validate(value)
                    for key, value in result.get("source_registry", {}).items()
                }
                return {f"{focus}_result": branch.model_copy(update={"sources": sources})}
        except Exception as error:  # noqa: BLE001 - researchers return failed status
            LOGGER.warning("%s research node failed (%s)", focus, type(error).__name__)
        return {f"{focus}_result": ResearchResult(
            status="failed", error="research_failed", round_id=state["research_round"],
        )}
    return node


def _focus(place: Place) -> str:
    """Match coverage's food/non-food pools, independent of researcher identity."""
    return "food" if place.category == "food" else "sights"


def _dedupe(places: list[Place]) -> list[Place]:
    """Keep the first place and attach other actual source URLs to extra_sources."""
    unique: dict[str, Place] = {}
    for place in places:
        name = normalize_name(place.name)
        if not name:
            continue
        if name not in unique:
            unique[name] = place.model_copy(deep=True)
            continue
        first = unique[name]
        urls = list(dict.fromkeys([
            *first.extra_sources, place.source_url, *place.extra_sources,
        ]))
        unique[name] = first.model_copy(update={
            "extra_sources": [url for url in urls if url != first.source_url],
        })
    return list(unique.values())


def merge_research(state: GraphState) -> dict[str, Any]:
    """Fan in once; replace only current dispatched pools and clear branch fields."""
    try:
        warnings = []
        replacement: dict[str, list[Place]] = {}
        sources: dict[str, SourceRecord] = {}
        for focus in state.get("dispatched", []):
            if focus not in {"sights", "food"}:
                continue
            raw = state.get(f"{focus}_result")
            if raw is None:
                warnings.append(f"{focus} research returned no result.")
                replacement[focus] = []
                continue
            result_round = raw.get("round_id") if isinstance(raw, Mapping) else raw.round_id
            if result_round != state["research_round"]:
                warnings.append(f"{focus} stale research result ignored.")
                continue  # Ignoring a stale response must not erase valid existing places.
            result = ResearchResult.model_validate(raw)
            sources.update(result.sources)
            warnings.extend(result.warnings)
            if result.status != "ok":
                warnings.append(f"{focus} research {result.status}.")
            candidates = [place for place in result.places
                          if result.status == "ok" and _focus(place) == focus]
            replacement[focus], dropped = validate_places(
                candidates, state.get("source_registry", {}) | sources,
            )
            if dropped:
                warnings.append(f"{focus} candidates dropped: invalid evidence.")
        old = [Place.model_validate(raw) for raw in state.get("places", [])]
        kept = [place for place in old if _focus(place) not in replacement]
        incoming = [place for values in replacement.values() for place in values]
        return {
            "places": _dedupe([*kept, *incoming]), "warnings": warnings,
            "source_registry": sources,
            "sights_result": None, "food_result": None, "dispatched": [],
        }
    except Exception as error:  # noqa: BLE001 - keep completed branch state recoverable
        LOGGER.warning("Research merge failed (%s)", type(error).__name__)
        return {"agent_error": AgentError(agent="merge_research", message="merge_failed")}


def _route_supervisor(state: GraphState) -> str | list[Send]:
    """Route the already validated/counted decision, without counting it a second time.

    orchestrator calls validate_decision before incrementing dispatch attempts and
    correction cycles. Revalidating those updated counters here would incorrectly
    block the second allowed attempt/send-back.
    """
    if state.get("agent_error") is not None:
        return "recovery"
    choice = state.get("next")
    if choice in {"both", "sights", "food"}:
        focuses = state.get("dispatched", [])
        if not focuses:
            return "recovery"
        return [Send(f"{focus}_researcher", dict(state)) for focus in focuses
                if focus in {"sights", "food"}]
    if choice == "traveler":
        return "human_review" if gate.approvable(state) else "recovery"
    if choice in {"planner", "critic", "recovery"}:
        return choice
    return END if choice == "stop" else "recovery"


def _gate_destination(state: GraphState) -> Literal["human_review", "recovery", "orchestrator"]:
    """Do not expose an unchecked draft as an approval pause."""
    if state.get("agent_error") is not None:
        return "orchestrator"
    return "human_review" if gate.route_after_limits(state) == "traveler" else "recovery"


def _findings(state: TripState, kind: str) -> list[Any]:
    """Read serialized or typed critiques for interrupt payloads."""
    value = state.get("critique")
    if value is None:
        return []
    try:
        return _json(getattr(Critique.model_validate(value), kind))
    except ValidationError:
        return ([{"issue": "schema invalid critique", "owner": "planner", "day": None}]
                if kind == "hard_failures" else [])


def _screen_warnings(state: Mapping[str, Any], *, approval: bool = False) -> list[str]:
    """Project persisted uncertainty and guardrails onto every human-facing screen."""
    warnings = list(state.get("warnings", []))
    itinerary = _json(state.get("itinerary"))
    if isinstance(itinerary, Mapping):
        warnings.extend(itinerary.get("warnings", []))
    if approval:
        warnings.extend(finding["issue"] for finding in _findings(state, "soft_findings"))
    else:
        warnings.extend("Hard failure: " + finding["issue"]
                        for finding in _findings(state, "hard_failures"))
    error = _json(state.get("agent_error"))
    if isinstance(error, Mapping):
        warnings.append(f"{error.get('agent', 'agent')} failed; completed state is retained for recovery.")
        explanation = {
            "specialist_step_limit_reached": "The agent exhausted its bounded tool/model steps; Retry can recheck the saved draft.",
            "structured_output_invalid_after_one_reask": "The agent returned invalid structured output after one correction.",
            "specialist_tool_failed": "A required tool failed after its allowed retries.",
        }.get(error.get("message"))
        if explanation:
            warnings.append(explanation)
    if state.get("trip_steps", 0) >= MAX_TRIP_STEPS:
        warnings.append("Trip limit reached; start a new trip or cancel if recovery is needed.")
    elif (
        state.get("round_steps", 0) >= MAX_ROUND_STEPS
        or state.get("correction_cycles", 0) >= MAX_CORRECTION_CYCLES
    ):
        warnings.append("Round or correction limit reached; limits never force approval.")
    return list(dict.fromkeys(warnings))


def trip_screen(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return a stable screen contract, including clarification and terminal outcomes."""
    pending = result.get("__interrupt__", ())
    if pending:
        return dict(pending[0].value)
    warnings = _screen_warnings(result)
    if result.get("saved_path"):
        return {"kind": "exported", "options": [], "warnings": warnings}
    if result.get("new_trip"):
        return {"kind": "new_trip", "options": ["new_trip", "cancel"], "warnings": warnings}
    if result.get("cancelled"):
        return {"kind": "cancelled", "options": [], "warnings": warnings}
    if result.get("request") is None and result.get("next") == "stop":
        return {"kind": "clarification", "options": ["new_trip", "cancel"], "warnings": warnings}
    return {"kind": "stopped", "options": [], "warnings": warnings}


def _approval_payload(state: GraphState) -> dict[str, Any]:
    """Build a read-only payload; no model, tools or writes run before a pause."""
    return {
        "kind": "approval", "itinerary": _json(state["itinerary"]),
        "draft_version": state["draft_version"],
        "soft_findings": _findings(state, "soft_findings"),
        "warnings": _screen_warnings(state, approval=True),
        "options": ["approve", "revise", "cancel"],
        "delegation_log": _json(state.get("delegation_log", [])[-10:]),
    }


def _answer(value: Any) -> dict[str, Any]:
    """Malformed resume values are rejected at the pause, never acted upon."""
    return dict(value) if isinstance(value, Mapping) else {}


def _revise(state: GraphState, answer: dict[str, Any], config: RunnableConfig, pause: str) -> Command:
    """Run revision extraction only after an interrupt has been resumed."""
    feedback = answer.get("feedback")
    if not isinstance(feedback, str) or not feedback.strip():
        return Command(goto=pause, update={"warnings": ["Please supply revision feedback."]})
    update = apply_revision(state, feedback, config)
    update["approved_version"] = None
    if update.get("agent_error") is not None or "request" in update:
        return Command(goto="orchestrator", update=update)
    return Command(goto=pause, update=update)


def human_review(
    state: GraphState, config: RunnableConfig,
) -> Command[Literal["save", "human_review", "recovery", "orchestrator", "end_cancelled"]]:
    """Pause first; approval applies only to this exact, currently approvable draft."""
    # This pure safety check is necessary when a checkpoint was changed while paused.
    if not gate.approvable(state):
        return Command(goto=_gate_destination(state), update={"approved_version": None})
    answer = _answer(interrupt(_approval_payload(state)))
    action = answer.get("decision")
    if action == "approve":
        version = answer.get("draft_version")
        if type(version) is not int or version != state["draft_version"]:
            return Command(goto=_gate_destination(state), update={
                "warnings": ["Stale approval rejected; review the current draft version."],
                "approved_version": None, "decision": None,
            })
        if not gate.approvable(state):
            return Command(goto=_gate_destination(state), update={"approved_version": None})
        return Command(goto="save", update={"decision": "approve", "approved_version": version})
    if action == "revise":
        return _revise(state, answer, config, "human_review")
    if action == "cancel":
        return Command(goto="end_cancelled", update={"decision": "cancel"})
    return Command(goto="human_review", update={
        "warnings": ["Choose approve, revise or cancel."],
    })


def _retry_instructions(state: GraphState) -> dict[str, str]:
    """Assign every open finding to its owning specialist(s)."""
    instructions = dict(state.get("instructions", {}))
    critique = state.get("critique")
    if critique is not None:
        checked = Critique.model_validate(critique)
        for finding in [*checked.hard_failures, *checked.soft_findings]:
            owners = ["planner"] if finding.owner == "planner" else ["sights", "food"]
            for owner in owners:
                instructions[owner] = "\n".join(filter(None, (
                    instructions.get(owner, ""), finding.issue,
                )))
    return instructions



def _exhausted_research(state: GraphState) -> list[str]:
    """Explain categories that cannot make progress through same-round Retry."""
    if state.get("request") is None:
        return []
    required = needed(state["request"])
    names = {"sights": set(), "food": set()}
    seen = set()
    for raw in state.get("places", []):
        place = Place.model_validate(raw)
        name = normalize_name(place.name)
        if name and name not in seen:
            names["food" if place.category == "food" else "sights"].add(name)
            seen.add(name)
    return [
        f"{focus.capitalize()} research found {len(names[focus])} of {required[focus]} required places "
        "and has used both attempts this round. Retry cannot reset that limit; "
        "revise your request or use New trip."
        for focus in required
        if len(names[focus]) < required[focus]
        and state.get("researcher_attempts", {}).get(focus, 0) >= MAX_RESEARCHER_ATTEMPTS
    ]


def recovery(
    state: GraphState, config: RunnableConfig,
) -> Command[Literal["recovery", "orchestrator", "end_cancelled", "__end__"]]:
    """Pause before revision or retry work, rejecting options unavailable at caps."""
    answer = _answer(interrupt({
        "kind": "recovery", "reason": (
            f"{_json(state['agent_error'])['agent']} agent_error"
            if state.get("agent_error") is not None
            else gate.recovery_reason(state) or "Recovery requested"
        ),
        "options": gate.recovery_options(state),
        "hard_findings": _findings(state, "hard_failures"),
        "warnings": _screen_warnings(state) + _exhausted_research(state),
    }))
    action = answer.get("decision")
    if action not in gate.recovery_options(state):
        return Command(goto="recovery", update={
            "warnings": ["That recovery option is unavailable. Choose one of the listed options."],
        })
    if action == "revise":
        return _revise(state, answer, config, "recovery")
    if action == "retry":
        blocked = _exhausted_research(state)
        if blocked:
            return Command(goto="recovery")
        error = _json(state.get("agent_error"))
        recheck = ({"critique_version": None}
                   if isinstance(error, Mapping) and error.get("agent") == "critic" else {})
        return Command(goto="orchestrator", update={
            **recheck, "agent_error": None, "approved_version": None, "decision": "retry",
            "instructions": _retry_instructions(state),
        })
    if action == "new_trip":
        return Command(goto=END, update={
            "new_trip": True, "decision": "new_trip", "next": "stop", "approved_version": None,
        })
    return Command(goto="end_cancelled", update={"decision": "cancel"})


def _save_guard(state: GraphState) -> bool:
    """Enforce the approval guard independently of the write tool."""
    approved = state.get("approved_version")
    return (
        state.get("decision") == "approve" and type(approved) is int
        and approved == state.get("draft_version") == state.get("critique_version")
        and gate.approvable(state)
    )


def save(
    state: GraphState, config: RunnableConfig,
) -> Command[Literal["human_review", "recovery", "orchestrator", "export_failed", "__end__"]]:
    """Never interrupt; export only the exact approved draft, retaining failed approval."""
    if not _save_guard(state):
        return Command(goto=_gate_destination(state), update={
            "approved_version": None, "decision": None,
            "warnings": ["Export rejected: this draft has no current approval."],
        })
    try:
        message = save_itinerary.invoke({
            "name": "save_itinerary", "type": "tool_call", "id": f"save-{state['draft_version']}",
            "args": {
                "itinerary": _json(state["itinerary"]),
                "thread_id": config["configurable"]["thread_id"],
                "decision": state["decision"], "draft_version": state["draft_version"],
                "approved_version": state["approved_version"],
                "critique_version": state["critique_version"], "has_hard_failures": False,
            },
        }, config=config)
        result = envelope(message) or {"ok": False, "error": "write_failed"}
    except Exception as error:  # noqa: BLE001 - export failures must remain resumable
        LOGGER.warning("Export tool failed (%s)", type(error).__name__)
        result = {"ok": False, "error": "write_failed"}
    data = result.get("data")
    if (
        result.get("ok") is True and isinstance(data, Mapping)
        and isinstance(data.get("path"), str) and isinstance(data.get("manifest"), dict)
    ):
        return Command(goto=END, update={
            "saved_path": data["path"], "export_manifest": data["manifest"], "export_error": None,
        })
    if result.get("error") == "not_approved":
        return Command(goto=_gate_destination(state), update={
            "approved_version": None, "decision": None, "warnings": ["Export approval was rejected."],
        })
    return Command(goto="export_failed", update={
        "saved_path": None, "export_manifest": None, "export_error": "write_failed",
        "warnings": ["Export failed; approval is retained. Retry export or cancel."],
    })


def export_failed(state: GraphState) -> Command[Literal["save", "export_failed", "end_cancelled"]]:
    """Pause before retrying export; do not revisit supervisor or touch step counts."""
    answer = _answer(interrupt({
        "kind": "export_failed", "reason": state.get("export_error") or "write_failed",
        "options": ["retry_export", "cancel"], "warnings": _screen_warnings(state),
    }))
    if answer.get("decision") == "retry_export":
        return Command(goto="save")
    if answer.get("decision") == "cancel":
        return Command(goto="end_cancelled", update={"decision": "cancel"})
    return Command(goto="export_failed", update={
        "warnings": ["Choose retry_export or cancel."],
    })


def end_cancelled(state: GraphState) -> dict[str, Any]:
    """Record cancellation without deleting checkpoints or exporting anything."""
    return {"cancelled": True, "decision": "cancel", "approved_version": None, "next": "stop"}


def get_checkpointer(path: str | Path = "checkpoints.sqlite") -> SqliteSaver:
    """Return a SQLite saver; the caller closes saver.conn when finished."""
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


def build_graph(checkpointer: BaseCheckpointSaver | None) -> CompiledStateGraph:
    """Build one persistent workflow, with no node-level RetryPolicy."""
    builder = StateGraph(GraphState)
    builder.add_node("orchestrator", _agent_node(orchestrator, "orchestrator"))
    builder.add_node("sights_researcher", _research_node("sights"))
    builder.add_node("food_researcher", _research_node("food"))
    builder.add_node("merge_research", merge_research)
    builder.add_node("planner", _agent_node(planner, "planner"))
    builder.add_node("critic", _agent_node(critic, "critic"))
    builder.add_node("human_review", human_review)
    builder.add_node("recovery", recovery)
    builder.add_node("save", save)
    builder.add_node("export_failed", export_failed)
    builder.add_node("end_cancelled", end_cancelled)
    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges("orchestrator", _route_supervisor, {
        name: name for name in (
            "sights_researcher", "food_researcher", "planner", "critic",
            "human_review", "recovery", END,
        )
    })
    # Individual edges coalesce into one merge task in the next superstep.
    # A static [sights, food] barrier would deadlock food-only dispatch.
    builder.add_edge("sights_researcher", "merge_research")
    builder.add_edge("food_researcher", "merge_research")
    for name in ("merge_research", "planner", "critic"):
        builder.add_edge(name, "orchestrator")
    builder.add_edge("end_cancelled", END)
    return builder.compile(checkpointer=checkpointer)


def _initial_state(user_input: str) -> GraphState:
    """Create isolated empty state; credentials/failure modes never enter checkpoints."""
    return {
        "user_input": user_input, "request": None, "original_request": None, "next": None,
        "instructions": {}, "delegation_log": [], "dispatched": [], "research_round": 0,
        "researcher_attempts": {"sights": 0, "food": 0}, "sights_result": None, "food_result": None,
        "places": [], "source_registry": {}, "weather": None, "itinerary": None, "draft_version": 0,
        "critique": None, "critique_version": None, "approved_version": None, "round_steps": 0,
        "trip_steps": 0, "correction_cycles": 0, "agent_error": None, "human_feedback": None,
        "decision": None, "warnings": [], "saved_path": None, "export_manifest": None,
        "new_trip": False, "cancelled": False, "export_error": None,
    }


def _pending(snapshot: Any) -> list[Any]:
    """Only unfinished task interrupts make a trip resumable."""
    return [item for task in snapshot.tasks for item in task.interrupts]


def _prepare(
    graph: CompiledStateGraph, user_input: str | None, thread_id: str | None,
    decision: str | None, draft_version: int | None, feedback: str | None,
    failure_mode: str | None,
) -> tuple[str, GraphState | Command, RunnableConfig]:
    """Prepare either a fresh thread or a resume without changing saved state."""
    thread = thread_id or str(uuid4())
    snapshot = graph.get_state(_config(thread, None))
    if decision is None:
        if user_input is None:
            raise ValueError("A new trip requires user_input")
        if snapshot.values:
            raise ValueError("thread_id already exists; resume it or start a new thread")
        return thread, _initial_state(user_input), _config(thread, None, failure_mode)
    if thread_id is None or not _pending(snapshot):
        raise ValueError("Trip has no pending interrupt")
    return thread, Command(resume={
        "decision": decision, "draft_version": draft_version, "feedback": feedback,
    }), _config(thread, _city(snapshot.values), failure_mode)


def start_trip(
    graph: CompiledStateGraph, user_input: str, thread_id: str | None = None,
    failure_mode: str | None = None,
) -> dict[str, Any]:
    """Run a fresh trip to completion/pause; return its state and thread_id."""
    thread, value, config = _prepare(graph, user_input, thread_id, None, None, None, failure_mode)
    result = {**graph.invoke(value, config=config), "thread_id": thread}
    return result | {"screen": trip_screen(result)}


def resume_trip(
    graph: CompiledStateGraph, thread_id: str, decision: str,
    draft_version: int | None = None, feedback: str | None = None,
    failure_mode: str | None = None,
) -> dict[str, Any]:
    """Resume exactly one pending workflow action using its current thread."""
    thread, value, config = _prepare(
        graph, None, thread_id, decision, draft_version, feedback, failure_mode,
    )
    result = {**graph.invoke(value, config=config), "thread_id": thread}
    return result | {"screen": trip_screen(result)}


def stream_trip(
    graph: CompiledStateGraph, user_input: str | None = None, thread_id: str | None = None,
    *, decision: str | None = None, draft_version: int | None = None,
    feedback: str | None = None, failure_mode: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream updates; __interrupt__ entries retain native Interrupt.value['kind']."""
    thread, value, config = _prepare(
        graph, user_input, thread_id, decision, draft_version, feedback, failure_mode,
    )
    for mode, chunk in graph.stream(value, config=config, stream_mode=["updates", "custom"]):
        if mode == "custom":
            yield {"progress": _json(chunk), "thread_id": thread}
        else:
            yield {**_json(chunk), "thread_id": thread}



def get_trip(graph: CompiledStateGraph, thread_id: str) -> dict[str, Any]:
    """Open a saved thread read-only, including its current native pause and screen."""
    snapshot = graph.get_state(_config(thread_id, None))
    if not snapshot.values:
        raise ValueError("Trip not found")
    result = {**_json(snapshot.values), "thread_id": thread_id}
    pending = _pending(snapshot)
    if pending:
        result["__interrupt__"] = tuple(pending)
    return result | {"screen": trip_screen(result)}



def get_city_images(city: str) -> list[dict[str, str]]:
    """Look up optional city photos without reading/writing any trip checkpoint."""
    result = search_city_images(city)
    return result["data"] if result["ok"] else []


def get_export(graph: CompiledStateGraph, thread_id: str) -> dict[str, Any]:
    """Return downloadable Markdown only for a complete, still-approved manifest.

    The UI never accesses output files or trusts checkpoint metadata as evidence
    of completion. Validate the actual bytes again before returning the download.
    """
    state = get_trip(graph, thread_id)
    path = state.get("saved_path")
    incomplete = {
        "path": path, "complete": False, "manifest": None,
        "markdown": None, "file_name": None,
    }
    if not isinstance(path, str) or not path.endswith(".json") or not _save_guard(state):
        return incomplete
    base = path.removesuffix(".json")
    try:
        if not export_files.export_is_complete(base):
            return incomplete
        manifest = json.loads(Path(base + ".manifest.json").read_text("utf-8"))
        json_data = Path(base + ".json").read_bytes()
        markdown = Path(base + ".md").read_bytes()
        if (
            manifest != state.get("export_manifest")
            or manifest["files"][".json"] != hashlib.sha256(json_data).hexdigest()
            or manifest["files"][".md"] != hashlib.sha256(markdown).hexdigest()
            or json.loads(json_data) != state.get("itinerary")
        ):
            return incomplete
        return {
            "path": path, "complete": True, "manifest": manifest,
            "markdown": markdown, "file_name": Path(base + ".md").name,
        }
    except (OSError, ValueError, KeyError, TypeError, export_files.ExportError):
        return incomplete


def list_trips(graph: CompiledStateGraph, checkpointer: BaseCheckpointSaver) -> list[dict[str, Any]]:
    """List distinct threads whose latest root snapshot has a pending interrupt."""
    # SQLite list() holds its lock across yields. Exhaust it before get_state
    # reads the same saver, or this would deadlock on its non-reentrant lock.
    threads = list(dict.fromkeys(
        item.config["configurable"]["thread_id"] for item in checkpointer.list(None)
    ))
    trips = []
    for thread in threads:
        snapshot = graph.get_state(_config(thread, None))
        pending = _pending(snapshot)
        if pending:
            trips.append({
                "thread_id": thread, "city": _city(snapshot.values),
                "last_update": snapshot.created_at, "paused_at": pending[0].value["kind"],
            })
    return sorted(trips, key=lambda trip: trip["last_update"] or "", reverse=True)


def delete_trip(checkpointer: BaseCheckpointSaver, thread_id: str) -> None:
    """Delete only the thread's checkpoint records using the installed saver API."""
    checkpointer.delete_thread(thread_id)


def export_graph_diagram(
    graph: CompiledStateGraph | None = None, path: str | Path | None = None,
) -> Path:
    """Write Mermaid locally without contacting an image-rendering service."""
    destination = Path(path) if path is not None else ROOT / "docs" / "graph.mmd"
    destination.parent.mkdir(parents=True, exist_ok=True)
    compiled = graph if graph is not None else build_graph(None)
    destination.write_text(compiled.get_graph().draw_mermaid(), encoding="utf-8")
    return destination


def _display_pause(pause: dict[str, Any]) -> None:
    """Print the review information needed to make an informed CLI decision."""
    if pause["kind"] == "approval":
        LOGGER.info("Draft v%s", pause["draft_version"])
        for day in pause["itinerary"]["days"]:
            LOGGER.info("Day %s: %s", day["day_number"], day["theme"])
            for stop in day["stops"]:
                LOGGER.info("  %s - %s: %s [%s]",
                            stop["time_slot"], stop["place_name"], stop["note"], stop["source_url"])
        for finding in pause["soft_findings"]:
            LOGGER.info("Soft finding: %s", finding["issue"])
    else:
        LOGGER.info("%s: %s", pause["kind"], pause["reason"])
    for warning in pause.get("warnings", []):
        LOGGER.info("Warning: %s", warning)
    if pause["kind"] == "recovery":
        LOGGER.info("Allowed options: %s", ", ".join(pause["options"]))


def main(argv: list[str] | None = None) -> int:
    """Stream the workflow and keep asking at its persistent review pauses."""
    parser = argparse.ArgumentParser(description="Multi-agent travel planner")
    parser.add_argument("request", nargs="?")
    parser.add_argument("--checkpoint", default="checkpoints.sqlite")
    parser.add_argument("--diagram", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.diagram:
        LOGGER.info("Diagram: %s", export_graph_diagram())
        if args.request is None:
            return 0
    if args.request is None:
        parser.error("a travel request is required")
    checkpointer = get_checkpointer(args.checkpoint)
    try:
        graph = build_graph(checkpointer)
        chunks = stream_trip(graph, args.request)
        while True:
            pause = None
            thread = None
            for chunk in chunks:
                thread = chunk["thread_id"]
                for name, update in chunk.items():
                    if name not in {"thread_id", "__interrupt__"} and isinstance(update, dict):
                        for entry in update.get("delegation_log", []):
                            LOGGER.info("Step %s -> %s: %s", entry["step"], entry["next"], entry["reason"])
                    if name == "__interrupt__":
                        pause = update[0].value
            if pause is None:
                values = graph.get_state(_config(thread, None)).values
                if values.get("saved_path"):
                    LOGGER.info("Exported: %s", values["saved_path"])
                elif values.get("new_trip"):
                    request = input("New trip request (or cancel): ").strip()
                    if request.casefold() != "cancel":
                        chunks = stream_trip(graph, request)
                        continue
                else:
                    LOGGER.info("Trip cancelled." if values.get("cancelled") else "Trip ended.")
                    for warning in values.get("warnings", []):
                        LOGGER.info("Warning: %s", warning)
                return 0
            _display_pause(pause)
            choices = {
                "approval": ["approve", "revise", "cancel"],
                "recovery": pause.get("options", []),
                "export_failed": ["retry_export", "cancel"],
            }[pause["kind"]]
            action = input("/".join(choices) + "> ").strip()
            feedback = input("Revision feedback: ") if action == "revise" else None
            chunks = stream_trip(
                graph, thread_id=thread, decision=action, feedback=feedback,
                draft_version=pause.get("draft_version") if action == "approve" else None,
            )
    finally:
        checkpointer.conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
