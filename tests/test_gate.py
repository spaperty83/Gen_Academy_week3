"""MOCKED: fail-closed approval, version freshness and recovery at limits."""

import pytest

from src.gate import approvable, recovery_options, recovery_reason, route_after_limits
from src.state import AgentError, Critique, Finding, Itinerary, TripRequest


@pytest.fixture
def ready():
    return {
        "itinerary": Itinerary(
            city="Lisbon", budget_tips=["Use transit.", "Consider free hours."],
            days=[{
                "day_number": 1, "theme": "Art", "stops": [{
                    "time_slot": "morning", "place_name": "MAAT",
                    "note": "Art by the river.", "source_url": "https://example.org",
                }],
            }],
        ),
        "draft_version": 2, "critique_version": 2, "critique": Critique(),
        "agent_error": None, "round_steps": 0, "trip_steps": 0, "correction_cycles": 0,
    }


@pytest.mark.parametrize(("round_steps", "trip_steps", "correction_cycles"), [
    (0, 0, 0), (12, 0, 0), (0, 30, 0), (12, 30, 2), (13, 31, 3),
])
def test_hard_failures_never_approvable_even_at_caps(
    ready, round_steps, trip_steps, correction_cycles,
):
    state = ready | {
        "round_steps": round_steps, "trip_steps": trip_steps,
        "correction_cycles": correction_cycles,
        "critique": Critique(hard_failures=[
            Finding(issue="Stop has no evidence", owner="researchers", day=1),
        ]),
    }
    assert not approvable(state)
    assert route_after_limits(state) == "recovery"


def test_soft_only_draft_is_approvable(ready):
    ready["critique"] = Critique(soft_findings=[
        Finding(issue="Weather unavailable", owner="planner"),
    ])
    assert approvable(ready)
    assert route_after_limits(ready) == "traveler"
    assert recovery_reason(ready) is None


def test_approvable_draft_at_caps_can_still_go_to_traveler(ready):
    assert route_after_limits(ready | {
        "round_steps": 12, "trip_steps": 30, "correction_cycles": 2,
    }) == "traveler"


def test_agent_error_forces_recovery(ready):
    ready["agent_error"] = AgentError(agent="planner", message="provider_failed")
    assert not approvable(ready)
    assert route_after_limits(ready) == "recovery"
    assert recovery_reason(ready) == "agent_error"


@pytest.mark.parametrize("changes", [
    {"itinerary": None}, {"itinerary": {}}, {"critique": None},
    {"critique": {}}, {"critique": {"hard_failures": None}},
    {"critique_version": None}, {"critique_version": 1},
    {"draft_version": None, "critique_version": None},
    {"draft_version": True, "critique_version": True},
    {"draft_version": -1, "critique_version": -1},
])
def test_missing_invalid_or_unchecked_state_fails_closed(ready, changes):
    assert not approvable(ready | changes)
    assert route_after_limits(ready | changes) == "recovery"


def test_serialized_checkpoint_models_are_supported(ready):
    serialized = ready | {
        "itinerary": ready["itinerary"].model_dump(),
        "critique": ready["critique"].model_dump(),
    }
    assert approvable(serialized)


@pytest.mark.parametrize("steps", [30, 31, 100])
def test_trip_cap_recovery_options(steps):
    assert recovery_options({"trip_steps": steps}) == ["new_trip", "cancel"]
    assert recovery_reason({"trip_steps": steps}) == "trip limit reached"


def test_normal_recovery_options():
    assert recovery_options({}) == ["revise", "retry", "cancel"]
    assert recovery_options({"trip_steps": 29, "round_steps": 12}) == [
        "revise", "retry", "cancel",
    ]


def test_recovery_reason_no_draft():
    assert recovery_reason({}) == "no draft"


def test_recovery_reason_unchecked_draft(ready):
    assert recovery_reason(ready | {"critique_version": 1}) == "unchecked draft"


def test_recovery_reason_hard_failures(ready):
    ready["critique"] = Critique(hard_failures=[
        Finding(issue="Duplicate stop", owner="planner"),
    ])
    assert recovery_reason(ready) == "hard failures"


def test_recovery_reason_research_short_of_coverage():
    trip = TripRequest(
        city="Lisbon", days=1, budget="budget", interests=[], pace="relaxed",
    )
    assert recovery_reason({"request": trip, "places": []}) == "research short of coverage"


def test_recovery_reason_priority(ready):
    state = ready | {
        "trip_steps": 30, "agent_error": AgentError(agent="critic", message="failed"),
    }
    assert recovery_reason(state) == "trip limit reached"


def test_mutated_model_cannot_bypass_schema_gate(ready):
    ready["itinerary"].days[0].stops[0].time_slot = "midnight"
    assert not approvable(ready)
    assert route_after_limits(ready) == "recovery"
