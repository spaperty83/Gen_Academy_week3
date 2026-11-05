"""Pure approval and recovery decisions; limits never bypass hard failures."""

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import ValidationError

from src.coverage import has_enough
from src.state import MAX_TRIP_STEPS, Critique, HumanDecision, Itinerary, TripState


def _draft(state: TripState | Mapping[str, Any]) -> Itinerary | None:
    """Fail closed on absent or schema-invalid drafts."""
    value = state.get("itinerary")
    if value is None:
        return None
    try:
        return Itinerary.model_validate(value)
    except ValidationError:
        return None


def _critique(state: TripState | Mapping[str, Any]) -> Critique | None:
    """Missing hard-failure data is unchecked, never implicitly clean."""
    value = state.get("critique")
    if isinstance(value, Mapping) and "hard_failures" not in value:
        return None
    if value is None:
        return None
    try:
        return Critique.model_validate(value)
    except ValidationError:
        return None


def _checked(state: TripState | Mapping[str, Any]) -> bool:
    """Require an explicit nonnegative draft version and matching critique."""
    version = state.get("draft_version")
    checked_version = state.get("critique_version")
    return (
        type(version) is int and version >= 0
        and type(checked_version) is int and checked_version == version
        and _critique(state) is not None
    )


def approvable(state: TripState | Mapping[str, Any]) -> bool:
    """Allow only an existing, current, hard-failure-free draft with no agent error."""
    critique = _critique(state)
    return (
        state.get("agent_error") is None
        and _draft(state) is not None
        and _checked(state)
        and critique is not None
        and not critique.hard_failures
    )


def route_after_limits(
    state: TripState | Mapping[str, Any],
) -> Literal["traveler", "recovery"]:
    """At any limit, offer only an approvable draft to the traveler."""
    return "traveler" if approvable(state) else "recovery"


def recovery_reason(state: TripState | Mapping[str, Any]) -> str | None:
    """Explain recovery, prioritizing the trip cap, agent errors and research gaps."""
    if state.get("trip_steps", 0) >= MAX_TRIP_STEPS:
        return "trip limit reached"
    if state.get("agent_error") is not None:
        return "agent_error"
    request = state.get("request")
    if request is not None and not has_enough(state.get("places", []), request):
        return "research short of coverage"
    if _draft(state) is None:
        return "no draft"
    if not _checked(state):
        return "unchecked draft"
    critique = _critique(state)
    if critique is not None and critique.hard_failures:
        return "hard failures"
    return None


def recovery_options(state: TripState | Mapping[str, Any]) -> list[HumanDecision]:
    """Offer a new trip instead of a futile retry once the trip cap is reached."""
    if state.get("trip_steps", 0) >= MAX_TRIP_STEPS:
        return ["new_trip", "cancel"]
    return ["revise", "retry", "cancel"]
