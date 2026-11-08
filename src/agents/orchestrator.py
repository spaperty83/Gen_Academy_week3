"""Tool-free supervision, deterministic guardrails and atomic request revisions."""

import json
import logging
import re
import unicodedata
from collections.abc import Mapping
from typing import Any, Literal, TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Overwrite
from pydantic import BaseModel, ConfigDict, ValidationError

from src import coverage, gate
from src.agents._runtime import StructuredOutputExhausted, tool_data
from src.agents.prompts import ORCHESTRATOR, PARSE_REQUEST, REVISION
from src.config import get_llm
from src.evidence import normalize_name
from src.state import (
    MAX_CORRECTION_CYCLES,
    MAX_RESEARCHER_ATTEMPTS,
    MAX_ROUND_STEPS,
    MAX_TRIP_STEPS,
    AgentError,
    Critique,
    NextAgent,
    OrchestratorDecision,
    Place,
    RequestPatch,
    TripRequest,
    TripState,
)

LOGGER = logging.getLogger(__name__)
ModelT = TypeVar("ModelT", bound=BaseModel)
FOCUSES = ("sights", "food")
CLARIFY_CITY = "Please clarify the city name and country; do not use a vague or misspelled destination."
REPHRASE = "The revision was not valid. Please rephrase using 1-7 days, a real ISO date, and a known pace/budget."


class ParsedTripRequest(BaseModel):
    """Distinguish destination uncertainty from provider/validation failures."""

    model_config = ConfigDict(extra="forbid")
    request: TripRequest | None
    city_status: Literal["clear", "missing", "vague", "misspelled", "ambiguous"]


def _structured(
    schema: type[ModelT], prompt: str, payload: dict[str, Any], config: RunnableConfig,
) -> ModelT:
    """Extract structured output with one validation re-ask and no provider retries."""
    model = get_llm().with_structured_output(schema, method="function_calling")
    messages = [SystemMessage(content=prompt), HumanMessage(content=tool_data(payload))]
    for attempt in range(2):
        try:
            return schema.model_validate(model.invoke(messages, config=config))
        except (ValidationError, OutputParserException) as error:
            # An unsupported route is a guardrail choice, not a reason to let
            # a model invent an operational tool or repeatedly request save.
            if schema is OrchestratorDecision and isinstance(error, ValidationError):
                failures = error.errors()
                if failures and all(
                    item["loc"] == ("next",) and item["type"] == "literal_error"
                    for item in failures
                ):
                    return OrchestratorDecision(
                        next="recovery", reason="guardrail: invalid routing choice replaced with recovery",
                    )
            if attempt:
                raise StructuredOutputExhausted("Invalid structured output") from None
            messages.append(HumanMessage(
                content="The response did not validate. Return one corrected response matching the schema.",
            ))
    raise StructuredOutputExhausted("Missing structured output")


def _counter(state: Mapping[str, Any], name: str) -> int:
    """Read a nonnegative counter without exposing malformed values."""
    value = state.get(name, 0)
    return value if type(value) is int and value >= 0 else 0


def _limits(state: Mapping[str, Any]) -> bool:
    """All limits route through the approval gate."""
    return (
        _counter(state, "round_steps") >= MAX_ROUND_STEPS
        or _counter(state, "trip_steps") >= MAX_TRIP_STEPS
        or _counter(state, "correction_cycles") >= MAX_CORRECTION_CYCLES
    )


def _city_text(text: str) -> str:
    """Compare explicit city mentions without correcting their spelling."""
    plain = unicodedata.normalize("NFKD", text.casefold())
    return " ".join("".join(
        char if char.isalnum() else " " for char in plain
        if not unicodedata.combining(char)
    ).split())


def _explicit_city(city: str, text: str) -> bool:
    """Require the extracted city to occur in the traveler's actual words."""
    name = _city_text(city)
    return bool(name) and f" {name} " in f" {_city_text(text)} " and name not in {
        "somewhere", "anywhere", "unknown", "unspecified", "a city",
        "europe", "portugal", "france", "spain", "italy", "united states",
    }


def _counts(places: list[Any]) -> dict[str, int]:
    """Match coverage's distinct-name food-versus-non-food accounting."""
    names: dict[str, set[str]] = {"sights": set(), "food": set()}
    seen: set[str] = set()
    for raw in places:
        place = Place.model_validate(raw)
        name = normalize_name(place.name)
        if name and name not in seen:
            names["food" if place.category == "food" else "sights"].add(name)
            seen.add(name)
    return {focus: len(values) for focus, values in names.items()}


def _short(state: Mapping[str, Any]) -> list[str]:
    """Find only categories below the minimum, not the larger research target."""
    required = coverage.needed(state["request"])
    counts = _counts(state.get("places", []))
    return [focus for focus in FOCUSES if counts[focus] < required[focus]]


def _researchers(next_agent: str) -> list[str]:
    """Translate a routing choice into its research branches."""
    return list(FOCUSES) if next_agent == "both" else (
        [next_agent] if next_agent in FOCUSES else []
    )


def validate_decision(
    state: TripState, decision: OrchestratorDecision | Mapping[str, Any] | str,
) -> tuple[NextAgent, str | None]:
    """Replace unsafe choices in code; replacements always have guardrail reasons."""
    chosen = decision if isinstance(decision, str) else (
        decision.next if isinstance(decision, OrchestratorDecision) else decision.get("next")
    )
    if state.get("agent_error") is not None:
        return "recovery", "guardrail: agent_error requires recovery"
    if chosen == "save" or chosen not in {
        "sights", "food", "both", "planner", "critic", "traveler", "recovery", "stop",
    }:
        return "recovery", "guardrail: invalid decision; save is never allowed"
    if _limits(state) or chosen == "traveler":
        routed = gate.route_after_limits(state)
        reason = "guardrail: limits or traveler request routed through approval gate"
        return routed, reason if routed != chosen or _limits(state) else None
    if (
        _researchers(chosen) and state.get("itinerary") is not None
        and state.get("critique_version") != state.get("draft_version")
    ):
        return "critic", "guardrail: check the current draft before dispatching more research"
    if chosen == "critic" and gate.approvable(state):
        return "traveler", "guardrail: current draft already checked; request traveler review"
    reasons = []
    if chosen == "critic" and not state.get("itinerary"):
        chosen = "planner"
        reasons.append("critic requires a draft")
    if chosen == "planner":
        if state.get("request") is None:
            return "recovery", "guardrail: planner requires a request"
        if not coverage.has_enough(state.get("places", []), state["request"]):
            short = _short(state)
            if any(
                state.get("researcher_attempts", {}).get(focus, 0) >= MAX_RESEARCHER_ATTEMPTS
                for focus in short
            ):
                return "recovery", "guardrail: research short of coverage; researcher attempts exhausted"
            chosen = "both" if len(short) == 2 else short[0]
            reasons.append("research short of coverage; dispatch only short researchers")
    if _researchers(chosen) and state.get("request") is not None:
        short = _short(state)
        if not short and not state.get("itinerary"):
            return "planner", "guardrail: coverage met; plan instead of repeating research"
        if short:
            selected = [focus for focus in _researchers(chosen) if focus in short] or short
            replacement = "both" if len(selected) == 2 else selected[0]
            if replacement != chosen:
                reasons.append("keep completed research; dispatch only short researchers")
            chosen = replacement
    if any(
        state.get("researcher_attempts", {}).get(focus, 0) >= MAX_RESEARCHER_ATTEMPTS
        for focus in _researchers(chosen)
    ):
        return "recovery", "guardrail: researcher attempt limit reached"
    return chosen, "guardrail: " + "; ".join(reasons) if reasons else None


def compact_summary(state: TripState) -> dict[str, Any]:
    """Summarize only the current request, never old requests or source bodies."""
    trip = TripRequest.model_validate(state["request"])
    counts, required = _counts(state.get("places", [])), coverage.needed(trip)
    branches = {}
    for focus in FOCUSES:
        result = state.get(f"{focus}_result")
        branches[focus] = {
            "status": result.get("status") if isinstance(result, Mapping) else getattr(result, "status", None),
            "round_id": result.get("round_id") if isinstance(result, Mapping) else getattr(result, "round_id", None),
        }
    critique = Critique.model_validate(state["critique"]) if state.get("critique") is not None else Critique()
    findings = {
        label: {owner: sum(finding.owner == owner for finding in values)
                for owner in ("researchers", "planner")}
        for label, values in (("hard", critique.hard_failures), ("soft", critique.soft_findings))
    }
    return {
        "request": trip.model_dump(mode="json"),
        "coverage": {focus: {"available": counts[focus], "needed": required[focus]}
                     for focus in FOCUSES},
        "branches": branches, "research_round": _counter(state, "research_round"),
        "researcher_attempts": dict(state.get("researcher_attempts", {})),
        "has_draft": state.get("itinerary") is not None,
        "approvable": gate.approvable(state),
        "draft_version": state.get("draft_version", 0),
        "critique_version": state.get("critique_version"), "findings_by_owner": findings,
        "human_feedback": (state.get("human_feedback") or "")[:1500],
        "steps": {name: _counter(state, name) for name in (
            "round_steps", "trip_steps", "correction_cycles",
        )},
        "agent_error": {"present": True} if state.get("agent_error") is not None else None,
    }


def _instructions(
    request: TripRequest, focuses: list[str], supplied: Mapping[str, str],
) -> dict[str, str]:
    """Generate authoritative coverage targets, budgets and current city."""
    instructions = dict(supplied)
    desired = coverage.targets(request)
    for focus in focuses:
        instructions[focus] = json.dumps({
            "city": request.city, "target": desired[focus],
            "tool_budget": coverage.tool_budget(desired[focus]),
            "instructions": supplied.get(focus, f"Research {focus} for the current request."),
        }, ensure_ascii=False)
    return instructions


def _dispatch(state: TripState, next_agent: str, supplied: Mapping[str, str]) -> dict[str, Any]:
    """Advance the research round once and count each dispatched branch once."""
    focuses = _researchers(next_agent)
    update: dict[str, Any] = {"dispatched": focuses, "instructions": dict(supplied)}
    if focuses:
        attempts = dict(state.get("researcher_attempts", {}))
        for focus in focuses:
            attempts[focus] = attempts.get(focus, 0) + 1
        update.update({
            "research_round": _counter(state, "research_round") + 1,
            "researcher_attempts": attempts,
            "instructions": _instructions(TripRequest.model_validate(state["request"]), focuses, supplied),
            "critique_version": None, "approved_version": None, "decision": None,
        })
    elif next_agent == "planner":
        update.update({"approved_version": None, "decision": None})
    return update


def _sendback(state: TripState, next_agent: str) -> bool:
    """Count a correction only when current findings are sent to their owner."""
    if (
        state.get("critique") is None or state.get("critique_version") is None
        or state.get("critique_version") != state.get("draft_version")
    ):
        return False
    critique = Critique.model_validate(state["critique"])
    owners = {finding.owner for finding in [*critique.hard_failures, *critique.soft_findings]}
    return (
        next_agent == "planner" and "planner" in owners
        or bool(_researchers(next_agent)) and "researchers" in owners
    )


def _finish(update: dict[str, Any], next_agent: NextAgent, reason: str) -> dict[str, Any]:
    """Append one structured audit record using the state's accumulating reducer."""
    return update | {
        "next": next_agent,
        "delegation_log": [{"step": update["trip_steps"], "next": next_agent, "reason": reason}],
    }


def orchestrator(state: TripState, config: RunnableConfig) -> dict[str, Any]:
    """Parse once, then supervise summaries; never expose service or write tools."""
    update = {
        "round_steps": _counter(state, "round_steps") + int(_counter(state, "round_steps") < MAX_ROUND_STEPS),
        "trip_steps": _counter(state, "trip_steps") + int(_counter(state, "trip_steps") < MAX_TRIP_STEPS),
    }
    try:
        current = state | update
        if state.get("agent_error") is not None or _limits(current):
            next_agent, reason = validate_decision(current, "recovery")
            update["warnings"] = [reason or "Recovery requested"]
            return _finish(update, next_agent, reason or "Recovery requested")
        if state.get("request") is None:
            parsed = _structured(
                ParsedTripRequest, PARSE_REQUEST, {"user_input": state.get("user_input", "")}, config,
            )
            if (
                parsed.city_status != "clear" or parsed.request is None
                or not _explicit_city(parsed.request.city, state.get("user_input", ""))
            ):
                return _finish(
                    update | {"warnings": [CLARIFY_CITY], "dispatched": []},
                    "stop", "guardrail: city requires clarification",
                )
            update |= {
                "request": parsed.request, "original_request": parsed.request.model_copy(deep=True),
                "agent_error": None,
            }
            current |= update
            decision = OrchestratorDecision(next="both", reason="Parsed request; research both categories.")
        else:
            decision = _structured(OrchestratorDecision, ORCHESTRATOR, compact_summary(state), config)
        next_agent, guardrail = validate_decision(current, decision)
        if guardrail is None and decision.reason.startswith("guardrail: "):
            guardrail = decision.reason
        instructions = dict(decision.instructions)
        if state.get("decision") == "retry":
            for owner, text in state.get("instructions", {}).items():
                instructions[owner] = "\n".join(filter(None, (text, instructions.get(owner, ""))))
        update |= _dispatch(current, next_agent, instructions)
        if _sendback(current, next_agent):
            update["correction_cycles"] = _counter(state, "correction_cycles") + 1
        update["agent_error"] = None
        reason = guardrail or decision.reason
        if guardrail:
            update["warnings"] = [guardrail]
            LOGGER.info("%s", guardrail)
        return _finish(update, next_agent, reason)
    except Exception as error:  # noqa: BLE001 - supervisor follows the agent boundary rule
        LOGGER.warning("Orchestrator failed (%s)", type(error).__name__)
        message = (
            "orchestrator_output_invalid_after_one_reask"
            if isinstance(error, StructuredOutputExhausted) else "orchestrator_failed"
        )
        return _finish(
            update | {"agent_error": AgentError(agent="orchestrator", message=message), "dispatched": [],
                      "warnings": ["orchestrator failed; completed state is retained for recovery."]},
            "recovery", "guardrail: orchestrator failure requires recovery",
        )


def _affected(old: TripRequest, new: TripRequest, changed: set[str]) -> list[str]:
    """Budget affects both pools; food-interest edits affect food, others sights."""
    focuses = set()
    if "budget" in changed:
        focuses.update(FOCUSES)
    if "interests" in changed:
        edits = {interest.strip().casefold() for interest in old.interests} ^ {
            interest.strip().casefold() for interest in new.interests
        }
        if "food" in edits:
            focuses.add("food")
        if edits - {"food"}:
            focuses.add("sights")
    return [focus for focus in FOCUSES if focus in focuses]


def apply_revision(
    state: TripState, feedback: str, config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Validate and apply a patch atomically before clearing dependent state."""
    try:
        old = TripRequest.model_validate(state["request"])
        patch = _structured(
            RequestPatch, REVISION, {"request": old, "feedback": feedback}, config or {},
        )
        clear_date = "dates" in patch.change_types or bool(re.search(
            r"\b(remove|clear|unset|drop)\b.*\b(date|dates)\b", feedback, re.IGNORECASE,
        ))
        values = {
            key: value for key, value in patch.model_dump(exclude_unset=True).items()
            if key in TripRequest.model_fields
            and (value is not None or key == "start_date" and clear_date)
        }
        if "interests" in values and re.search(r"\b(add|include|also)\b", feedback, re.IGNORECASE) and not re.search(
            r"\b(remove|replace|instead|only)\b", feedback, re.IGNORECASE,
        ):
            values["interests"] = list(dict.fromkeys([*old.interests, *values["interests"]]))
        new = TripRequest.model_validate(old.model_dump() | values)
        changed = {key for key in TripRequest.model_fields if getattr(new, key) != getattr(old, key)}
        if "city" in changed and not _explicit_city(new.city, feedback):
            return {"warnings": [CLARIFY_CITY]}
        order_changed = "order" in patch.change_types or bool(patch.other_notes)
        if not changed and not order_changed:
            return {"warnings": ["No request change was identified. Please rephrase the revision."]}
        # The new request is validated FIRST; only then invalidate dependencies.
        update: dict[str, Any] = {
            "request": new, "human_feedback": feedback, "round_steps": 0,
            "correction_cycles": 0, "researcher_attempts": {"sights": 0, "food": 0},
            "agent_error": None, "dispatched": [], "instructions": {},
            "itinerary": None, "critique": None, "critique_version": None,
            "approved_version": None, "decision": None, "saved_path": None, "export_manifest": None,
        }
        if "city" in changed:
            focuses = list(FOCUSES)
            update |= {
                "places": [], "sights_result": None, "food_result": None,
                "source_registry": Overwrite(value={}), "warnings": Overwrite(value=[]), "weather": None,
            }
        else:
            focuses = _affected(old, new, changed)
            if focuses:
                update["places"] = [
                    raw for raw in state.get("places", [])
                    if ("food" if Place.model_validate(raw).category == "food" else "sights") not in focuses
                ]
                update.update({f"{focus}_result": None for focus in focuses})
            if changed & {"start_date", "days"}:
                update["weather"] = None
        if focuses:
            update["next"] = "both" if len(focuses) == 2 else focuses[0]
            update["instructions"] = _instructions(new, focuses, {})
        else:
            update["next"] = "planner"
        # original_request, trip_steps and draft_version stay untouched.
        # Routing intent is set here; actual dispatch is counted by orchestrator.
        return update
    except (ValidationError, OutputParserException, StructuredOutputExhausted):
        return {"warnings": [REPHRASE]}
    except Exception as error:  # noqa: BLE001 - failures keep the old request intact
        LOGGER.warning("Revision failed (%s)", type(error).__name__)
        return {
            "agent_error": AgentError(agent="orchestrator", message="revision_failed"),
            "next": "recovery",
        }
