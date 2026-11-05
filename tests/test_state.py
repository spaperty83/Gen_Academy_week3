"""MOCKED: schemas and actual LangGraph reducer/Overwrite behavior; no services."""

from typing import get_type_hints

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Overwrite
from pydantic import ValidationError

from src.state import (
    AgentError,
    Critique,
    DayPlan,
    Finding,
    Itinerary,
    OrchestratorDecision,
    Place,
    RequestPatch,
    ResearchResult,
    SourceRecord,
    Stop,
    TripRequest,
    TripState,
    merge_source_registry,
    preserve_draft_version,
)


@pytest.fixture
def request_data():
    return {
        "city": "Lisbon", "days": 4, "budget": "budget",
        "interests": ["food", "history"], "pace": "normal",
    }


@pytest.fixture
def place_data():
    return {
        "name": "MAAT", "category": "sight", "description": "Art by the river.",
        "found_by": "sights", "source_id": "real", "source_url": "https://example.org",
        "source_title": "MAAT", "retrieved_at": "2026-11-04T12:00:00Z",
        "evidence_quote": "Visit MAAT by the river.",
    }


def make_source(text="Visit MAAT by the river."):
    return SourceRecord(
        url="https://example.org", retrieved_at="2026-11-04T12:00:00Z", text=text,
    )


def make_itinerary():
    return Itinerary(
        city="Lisbon",
        days=[DayPlan(
            day_number=1, theme="Art",
            stops=[Stop(
                time_slot="morning", place_name="MAAT", note="Art by the river.",
                source_url="https://example.org",
            )],
        )],
        budget_tips=["Use transit.", "Consider free museum hours."],
    )


def test_valid_models_and_json_round_trip(request_data, place_data):
    request = TripRequest(**request_data, start_date="2026-11-04")
    assert request.start_date == "2026-11-04"
    assert TripRequest.model_validate_json(request.model_dump_json()) == request
    place = Place(**place_data)
    assert ResearchResult(status="ok", places=[place], round_id=1).places == [place]
    assert ResearchResult(status="failed", error="service_failed", round_id=1).places == []
    assert OrchestratorDecision(next="both", reason="Research both categories").next == "both"
    finding = Finding(issue="Missing evidence", owner="researchers", day=1)
    critique = Critique(hard_failures=[finding])
    assert critique.soft_findings == []
    assert AgentError(agent="planner", message="provider_failed").agent == "planner"
    itinerary = make_itinerary()
    assert Itinerary.model_validate_json(itinerary.model_dump_json()) == itinerary


@pytest.mark.parametrize("changes", [
    {"days": 0}, {"days": 8}, {"days": True}, {"city": ""},
    {"budget": "cheap"}, {"pace": "rushed"}, {"interests": "food"},
    {"start_date": "2026-02-30"}, {"start_date": "20261104"},
    {"start_date": "04/11/2026"}, {"start_date": "2026-11-04T12:00:00"},
    {"unexpected": "typo"},
])
def test_invalid_trip_requests(request_data, changes):
    with pytest.raises(ValidationError):
        TripRequest.model_validate(request_data | changes)


def test_request_patch_fields_are_optional_and_date_can_be_cleared():
    patch = RequestPatch()
    assert patch.model_dump(exclude_unset=True) == {}
    assert patch.change_types == []
    revision = RequestPatch(days=7, other_notes="Move museums later.", change_types=["days", "order"])
    assert revision.days == 7
    assert RequestPatch(start_date=None).model_fields_set == {"start_date"}


@pytest.mark.parametrize("data", [
    {"days": 0}, {"days": 8}, {"budget": "cheap"}, {"pace": "rushed"},
    {"start_date": "2026-02-30"}, {"change_types": ["save"]},
])
def test_invalid_request_patch(data):
    with pytest.raises(ValidationError):
        RequestPatch(**data)


@pytest.mark.parametrize(("model", "data"), [
    (SourceRecord, {"url": "https://example.org", "retrieved_at": "now", "text": "x" * 4001}),
    (ResearchResult, {"status": "pending", "round_id": 1}),
    (ResearchResult, {"status": "ok", "round_id": -1}),
    (Stop, {"time_slot": "midnight", "place_name": "MAAT", "note": "", "source_url": "url"}),
    (DayPlan, {"day_number": 0, "theme": "Art", "stops": []}),
    (Finding, {"issue": "Missing evidence", "owner": "orchestrator"}),
    (OrchestratorDecision, {"next": "save", "reason": "Write"}),
    (OrchestratorDecision, {"next": "export", "reason": "Write"}),
    (Itinerary, {"city": "Lisbon", "days": [], "budget_tips": ["Only one"]}),
    (Itinerary, {"city": "Lisbon", "days": [], "budget_tips": ["a", "b", "c", "d"]}),
    (AgentError, {"agent": "planner"}),
])
def test_other_invalid_models(model, data):
    with pytest.raises(ValidationError):
        model.model_validate(data)


@pytest.mark.parametrize("changes", [
    {"category": "hotel"}, {"found_by": "critic"},
    {"source_id": ""}, {"evidence_quote": ""},
])
def test_invalid_place(place_data, changes):
    with pytest.raises(ValidationError):
        Place.model_validate(place_data | changes)


def test_source_limit_includes_exactly_4000_characters():
    assert len(make_source("x" * 4000).text) == 4000


def test_mutable_defaults_are_not_shared_between_trips(place_data):
    first, second = Place(**place_data), Place(**place_data)
    first.extra_sources.append("https://extra.example.org")
    assert second.extra_sources == []
    first_critique, second_critique = Critique(), Critique()
    first_critique.hard_failures.append(Finding(issue="missing", owner="planner"))
    assert second_critique.hard_failures == []
    first_patch, second_patch = RequestPatch(), RequestPatch()
    first_patch.change_types.append("city")
    assert second_patch.change_types == []


def test_registry_reducer_merges_without_mutating_inputs():
    first, second = {"one": make_source()}, {"two": make_source("Another source")}
    merged = merge_source_registry(first, second)
    assert set(merged) == {"one", "two"}
    assert set(first) == {"one"} and set(second) == {"two"}
    assert merged["two"] == second["two"]
    assert merge_source_registry(first, {}) == first
    serialized = {"three": make_source().model_dump()}
    assert isinstance(merge_source_registry(first, serialized)["three"], SourceRecord)


def test_registry_reducer_preserves_source_constraints():
    with pytest.raises(ValidationError):
        merge_source_registry({}, {
            "bad": {"url": "https://example.org", "retrieved_at": "now", "text": "x" * 4001},
        })


def test_langgraph_native_overwrite_clears_accumulators():
    builder = StateGraph(TripState)

    def ordinary_empty_update(state):
        assert state["source_registry"] and state["warnings"] == ["old warning"]
        return {"source_registry": {}, "warnings": [], "delegation_log": []}

    def clear_accumulators(state):
        assert state["source_registry"] and state["warnings"] == ["old warning"]
        assert state["delegation_log"] == ["old decision"]
        return {
            "source_registry": Overwrite(value={}),
            "warnings": Overwrite(value=[]),
            "delegation_log": Overwrite(value=[]),
        }

    builder.add_node("empty_update", ordinary_empty_update)
    builder.add_node("clear", clear_accumulators)
    builder.add_edge(START, "empty_update")
    builder.add_edge("empty_update", "clear")
    builder.add_edge("clear", END)
    result = builder.compile().invoke({
        "source_registry": {"one": make_source()},
        "warnings": ["old warning"], "delegation_log": ["old decision"],
    })
    assert result["source_registry"] == {}
    assert result["warnings"] == []
    assert result["delegation_log"] == []


def test_draft_version_cannot_decrease_through_normal_updates():
    builder = StateGraph(TripState)
    builder.add_node("stale_version", lambda state: {"draft_version": 1})
    builder.add_edge(START, "stale_version")
    builder.add_edge("stale_version", END)
    assert builder.compile().invoke({"draft_version": 3})["draft_version"] == 3
    assert preserve_draft_version(3, 4) == 4
    with pytest.raises(ValueError, match="nonnegative"):
        preserve_draft_version(3, -1)


def test_checkpoint_threads_do_not_share_accumulating_state():
    builder = StateGraph(TripState)
    builder.add_node("note", lambda state: {"warnings": [state["user_input"]]})
    builder.add_edge(START, "note")
    builder.add_edge("note", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    first = {"configurable": {"thread_id": "trip-one"}}
    second = {"configurable": {"thread_id": "trip-two"}}
    graph.invoke({"user_input": "first"}, config=first)
    graph.invoke({"user_input": "second"}, config=second)
    assert graph.get_state(first).values["warnings"] == ["first"]
    assert graph.get_state(second).values["warnings"] == ["second"]


def test_trip_state_has_exact_requested_field_names():
    assert set(get_type_hints(TripState)) == {
        "user_input", "original_request", "request", "next", "instructions",
        "delegation_log", "dispatched", "research_round", "researcher_attempts",
        "sights_result", "food_result", "places", "source_registry", "weather",
        "itinerary", "draft_version", "critique", "critique_version", "approved_version",
        "round_steps", "trip_steps", "correction_cycles", "agent_error",
        "human_feedback", "decision", "warnings", "saved_path", "export_manifest",
    }
