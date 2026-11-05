"""MOCKED: tool-free supervisor decisions, atomic revisions and bounded routing."""

import copy
import json
from html import unescape
from unittest.mock import Mock

import pytest
import requests
from langgraph.graph import END, START, StateGraph
from langgraph.types import Overwrite

from src.agents import orchestrator as module
from src.agents.orchestrator import (
    ParsedTripRequest,
    apply_revision,
    compact_summary,
    orchestrator,
    validate_decision,
)
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
)
from tests.test_agents import ScriptedModel, structured


class FakeModel:
    """Return scripted structured values; no provider retries, tools or network."""

    def __init__(self, values, by_schema=None):
        self.values = list(values)
        self.by_schema = by_schema or {}
        self.calls = []
        self.bindings = []
        self.index = 0

    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        self.bindings.append((schema, kwargs))
        return self

    def invoke(self, messages, config=None):
        self.calls.append((self.schema, list(messages), config))
        if self.schema in self.by_schema:
            value = self.by_schema[self.schema]
        else:
            value = self.values[min(self.index, len(self.values) - 1)]
            self.index += 1
        if isinstance(value, Exception):
            raise value
        return value


def use_model(monkeypatch, values, **kwargs):
    model = FakeModel(values, **kwargs)
    monkeypatch.setattr(module, "get_llm", lambda: model)
    return model


def decision(next_agent, **instructions):
    return OrchestratorDecision(next=next_agent, instructions=instructions, reason="Scripted decision")


def payload(message):
    return json.loads(unescape(
        message.content.removeprefix("<tool_data>").removesuffix("</tool_data>")
    ))


def updated(state, update):
    """Apply the relevant real-state reducer semantics for sequential node tests."""
    result = copy.deepcopy(state)
    for key, value in update.items():
        if isinstance(value, Overwrite):
            result[key] = value.value
        elif key in {"warnings", "delegation_log"}:
            result[key] = result.get(key, []) + value
        elif key == "source_registry":
            result[key] = result.get(key, {}) | value
        else:
            result[key] = value
    return result


@pytest.fixture(autouse=True)
def no_services(monkeypatch):
    for key in ("NEBIUS_API_KEY", "TAVILY_API_KEY", "LANGSMITH_API_KEY", "LANGSMITH_TRACING"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        requests.sessions.Session, "request",
        Mock(side_effect=AssertionError("Mocked test attempted networking")),
    )


@pytest.fixture
def state():
    trip = TripRequest(
        city="Lisbon", days=3, start_date="2027-05-01", budget="mid",
        interests=["history"], pace="relaxed",
    )
    places = []
    registry = {}
    for index in range(9):
        food = index >= 6
        name = f"Restaurant {index}" if food else f"Museum {index}"
        text = f"{name} is an indoor venue in Lisbon."
        place = Place(
            name=name, category="food" if food else "sight", description=text,
            found_by="food" if food else "sights", source_id=f"source-{index}",
            source_url=f"https://example.org/{index}", source_title=name,
            retrieved_at="2026-11-04T12:00:00Z", evidence_quote=text,
        )
        places.append(place)
        registry[place.source_id] = SourceRecord(
            url=place.source_url, retrieved_at=place.retrieved_at, text=text,
        )
    itinerary = Itinerary(
        city="Lisbon", budget_tips=["Use transit.", "Consider free hours."],
        days=[DayPlan(
            day_number=index + 1, theme="Explore",
            stops=[Stop(
                time_slot=slot, place_name=place.name, source_url=place.source_url, note="Visit.",
            ) for slot, place in zip(
                ("morning", "afternoon", "evening"),
                [places[index * 2], places[index * 2 + 1], places[index + 6]], strict=True,
            )],
        ) for index in range(3)],
    )
    return {
        "user_input": "3 days in Lisbon with history", "request": trip,
        "original_request": trip.model_copy(deep=True), "places": places,
        "source_registry": registry, "itinerary": itinerary,
        "draft_version": 8, "critique_version": 8, "approved_version": 8,
        "critique": Critique(), "decision": "approve", "weather": {"city": "Lisbon"},
        "research_round": 2, "researcher_attempts": {"sights": 1, "food": 1},
        "sights_result": ResearchResult(status="ok", places=places[:6], round_id=2),
        "food_result": ResearchResult(status="ok", places=places[6:], round_id=2),
        "round_steps": 4, "trip_steps": 9, "correction_cycles": 1,
        "warnings": ["An old warning"], "delegation_log": [],
        "saved_path": "outputs/old.json", "export_manifest": {"old": "export"},
        "agent_error": None,
    }


@pytest.mark.parametrize(("feedback", "patch", "field", "expected", "next_agent", "clears_weather"), [
    ("change Lisbon to Paris", {"city": "Paris", "change_types": ["city"]},
     "city", "Paris", "both", True),
    ("Start on 2027-06-02", {"start_date": "2027-06-02", "change_types": ["dates"]},
     "start_date", "2027-06-02", "planner", True),
    ("Change from 3 to 5 days", {"days": 5, "change_types": ["days"]},
     "days", 5, "planner", True),
    ("Add nightlife", {"interests": ["nightlife"], "change_types": ["interests"]},
     "interests", ["history", "nightlife"], "sights", False),
    ("Change budget to luxury", {"budget": "luxury", "change_types": ["budget"]},
     "budget", "luxury", "both", False),
    ("Change relaxed to packed", {"pace": "packed", "change_types": ["pace"]},
     "pace", "packed", "planner", False),
])
def test_requested_revisions(
    state, monkeypatch, feedback, patch, field, expected, next_agent, clears_weather,
):
    before = copy.deepcopy(state)
    use_model(monkeypatch, [RequestPatch(**patch)])
    update = apply_revision(state, feedback)
    after = updated(state, update)
    assert getattr(after["request"], field) == expected
    assert after["original_request"] == before["original_request"]
    assert "original_request" not in update
    assert after["next"] == next_agent
    assert after["draft_version"] == 8 and "draft_version" not in update
    assert after["trip_steps"] == 9 and "trip_steps" not in update
    assert after["round_steps"] == after["correction_cycles"] == 0
    assert after["researcher_attempts"] == {"sights": 0, "food": 0}
    assert after["itinerary"] is after["critique"] is after["approved_version"] is None
    assert after["critique_version"] is None and after["decision"] is None
    assert after["saved_path"] is after["export_manifest"] is None
    assert state == before  # no mutation, including nested models
    if clears_weather:
        assert after["weather"] is None
    else:
        assert after["weather"] == before["weather"]
    if field == "city":
        assert after["places"] == [] and after["source_registry"] == {}
        assert after["sights_result"] is after["food_result"] is None
        assert isinstance(update["warnings"], Overwrite)
        assert isinstance(update["source_registry"], Overwrite)
        assert after["warnings"] == []
    elif field == "interests":
        assert all(place.category == "food" for place in after["places"])
        assert after["food_result"] == before["food_result"]
        assert after["sights_result"] is None
        assert after["source_registry"] == before["source_registry"]
    elif field == "budget":
        assert after["places"] == []
        assert after["sights_result"] is after["food_result"] is None
        assert after["source_registry"] == before["source_registry"]
    else:
        assert after["places"] == before["places"]
        assert after["source_registry"] == before["source_registry"]
        assert after["sights_result"] == before["sights_result"]


def test_paris_is_never_reparsed_as_lisbon(state, monkeypatch):
    model = use_model(monkeypatch, [
        RequestPatch(city="Paris", change_types=["city"]), decision("both"),
    ])
    revised = updated(state, apply_revision(state, "change Lisbon to Paris"))
    result = orchestrator(revised, {})
    assert revised["request"].city == "Paris"
    assert result["next"] == "both" and "request" not in result
    assert revised["original_request"].city == "Lisbon"
    summary = payload(model.calls[-1][1][-1])
    assert summary["request"]["city"] == "Paris"
    assert "original_request" not in summary and "user_input" not in summary
    assert all(json.loads(value)["city"] == "Paris" for value in result["instructions"].values())


def test_city_revision_uses_native_langgraph_overwrite(state, monkeypatch):
    use_model(monkeypatch, [RequestPatch(city="Paris", change_types=["city"])])
    builder = StateGraph(TripState)
    builder.add_node("revise", lambda state: apply_revision(state, "change Lisbon to Paris"))
    builder.add_edge(START, "revise")
    builder.add_edge("revise", END)
    result = builder.compile().invoke(state)
    assert result["request"].city == "Paris" and result["original_request"].city == "Lisbon"
    assert result["source_registry"] == {} and result["warnings"] == []
    assert result["draft_version"] == 8


@pytest.mark.parametrize("bad", [
    {"days": 12}, {"start_date": "2027-02-30"}, {"pace": "rushed"}, {"budget": "cheap"},
])
def test_invalid_patch_changes_nothing_and_warns(state, monkeypatch, bad):
    before = copy.deepcopy(state)
    model = use_model(monkeypatch, [bad])
    update = apply_revision(state, "change my trip")
    assert set(update) == {"warnings"} and "rephrase" in update["warnings"][0].lower()
    assert state == before and model.index == 2


def test_revision_one_reask_can_recover(state, monkeypatch):
    model = use_model(monkeypatch, [{"days": 12}, RequestPatch(days=5)])
    assert apply_revision(state, "make it 5 days")["request"].days == 5
    assert len(model.calls) == 2


def test_food_interest_refreshes_only_food(state, monkeypatch):
    use_model(monkeypatch, [RequestPatch(interests=["food"], change_types=["interests"])])
    result = updated(state, apply_revision(state, "add food"))
    assert result["next"] == "food"
    assert result["request"].interests == ["history", "food"]
    assert all(place.category != "food" for place in result["places"])
    assert result["sights_result"] == state["sights_result"]
    assert result["food_result"] is None


def test_order_revision_only_invalidates_draft(state, monkeypatch):
    use_model(monkeypatch, [RequestPatch(change_types=["order"], other_notes="Swap the first two stops.")])
    result = updated(state, apply_revision(state, "Swap the first two stops."))
    assert result["request"] == state["request"]
    assert result["next"] == "planner" and result["itinerary"] is None
    assert result["weather"] == state["weather"] and result["places"] == state["places"]


def test_combined_revision_applies_all_invalidations(state, monkeypatch):
    use_model(monkeypatch, [RequestPatch(days=5, interests=["food"], change_types=["days", "interests"])])
    result = updated(state, apply_revision(state, "5 days and add food"))
    assert result["request"].days == 5 and "food" in result["request"].interests
    assert result["weather"] is None and result["next"] == "food"
    assert all(place.category != "food" for place in result["places"])


def test_date_can_be_explicitly_removed(state, monkeypatch):
    use_model(monkeypatch, [RequestPatch(start_date=None, change_types=["dates"])])
    assert apply_revision(state, "remove the start date")["request"].start_date is None


@pytest.mark.parametrize("change", [
    {"round_steps": 12}, {"trip_steps": 30}, {"correction_cycles": 2},
])
@pytest.mark.parametrize("route", ["traveler", "recovery"])
def test_caps_route_through_gate(state, monkeypatch, change, route):
    route_mock = Mock(return_value=route)
    monkeypatch.setattr(module.gate, "route_after_limits", route_mock)
    next_agent, reason = validate_decision(state | change, decision("planner"))
    assert next_agent == route and reason.startswith("guardrail: ")
    route_mock.assert_called_once()


def test_agent_error_always_routes_recovery(state):
    next_agent, reason = validate_decision(state | {
        "agent_error": AgentError(agent="planner", message="provider_failed"),
    }, decision("traveler"))
    assert next_agent == "recovery" and reason.startswith("guardrail: ")


@pytest.mark.parametrize("with_hard_failure", [True, False])
def test_actual_gate_never_forces_hard_failure_approval_at_caps(state, with_hard_failure):
    state["trip_steps"] = 30
    if with_hard_failure:
        state["critique"] = Critique(hard_failures=[Finding(issue="bad evidence", owner="researchers")])
    assert validate_decision(state, "planner")[0] == (
        "recovery" if with_hard_failure else "traveler"
    )


def test_traveler_request_is_also_gate_checked(state):
    state["critique_version"] = None
    assert validate_decision(state, decision("traveler"))[0] == "recovery"


@pytest.mark.parametrize(("pool", "expected"), [
    ("sights", "food"), ("food", "sights"), ("none", "both"),
])
def test_planner_dispatches_only_short_categories(state, pool, expected):
    state["places"] = [
        place for place in state["places"]
        if (place.category == "food") == (pool == "food")
    ] if pool != "none" else []
    state["researcher_attempts"] = {"sights": 0, "food": 0}
    next_agent, reason = validate_decision(state, decision("planner"))
    assert next_agent == expected and "guardrail: research short of coverage" in reason


def test_research_short_twice_recovers_never_plans(state, monkeypatch):
    state["places"] = []
    state["researcher_attempts"] = {"sights": 0, "food": 0}
    state["correction_cycles"] = 0
    use_model(monkeypatch, [decision("planner")])
    for _ in range(2):
        result = orchestrator(state, {})
        assert result["next"] == "both"
        state = updated(state, result)
    result = orchestrator(state, {})
    assert result["next"] == "recovery"
    assert state["researcher_attempts"] == {"sights": 2, "food": 2}
    assert "research short of coverage" in result["delegation_log"][-1]["reason"]


def test_direct_research_dispatch_cannot_exceed_attempt_cap(state):
    state["researcher_attempts"]["sights"] = 2
    assert validate_decision(state, "sights")[0] == "recovery"


def test_critic_without_draft_redirects_and_rechecks_planner_coverage(state):
    state["itinerary"] = None
    assert validate_decision(state, "critic")[0] == "planner"
    state["places"] = []
    state["researcher_attempts"] = {"sights": 0, "food": 0}
    assert validate_decision(state, "critic")[0] == "both"


@pytest.mark.parametrize("invalid", ["save", "export", "book", "unknown"])
def test_invalid_choices_never_route_to_save(state, invalid):
    next_agent, reason = validate_decision(state, {"next": invalid})
    assert next_agent == "recovery" and reason.startswith("guardrail: ")


def test_dispatch_counts_round_and_budgets_from_current_request(state, monkeypatch):
    state["request"] = state["request"].model_copy(update={
        "days": 7, "pace": "packed", "interests": ["food"],
    })
    state["correction_cycles"] = 0
    use_model(monkeypatch, [decision("both")])
    result = orchestrator(state, {})
    assert result["dispatched"] == ["sights", "food"] and result["research_round"] == 3
    assert result["researcher_attempts"] == {"sights": 2, "food": 2}
    assert json.loads(result["instructions"]["sights"])["target"] == 28
    assert json.loads(result["instructions"]["sights"])["tool_budget"] == 10
    assert json.loads(result["instructions"]["food"])["target"] == 19
    assert json.loads(result["instructions"]["food"])["tool_budget"] == 8
    assert result["round_steps"] == 5 and result["trip_steps"] == 10
    assert set(result["delegation_log"][0]) == {"step", "next", "reason"}


def test_correction_cycle_counts_current_critic_owner_sendback(state, monkeypatch):
    state["correction_cycles"] = 0
    state["critique"] = Critique(hard_failures=[Finding(issue="Reorder stops.", owner="planner")])
    use_model(monkeypatch, [decision("planner")])
    assert orchestrator(state, {})["correction_cycles"] == 1


def test_stale_critique_does_not_count_as_new_correction(state, monkeypatch):
    state["critique_version"] = 7
    state["critique"] = Critique(hard_failures=[Finding(issue="Reorder stops.", owner="planner")])
    use_model(monkeypatch, [decision("planner")])
    assert "correction_cycles" not in orchestrator(state, {})


def test_compact_summary_has_counts_without_places_or_source_text(state):
    state["critique"] = Critique(
        hard_failures=[Finding(issue="Evidence", owner="researchers")],
        soft_findings=[Finding(issue="Weather", owner="planner")],
    )
    summary = compact_summary(state)
    assert summary["coverage"] == {
        "sights": {"available": 6, "needed": 6}, "food": {"available": 3, "needed": 3},
    }
    assert summary["findings_by_owner"]["hard"]["researchers"] == 1
    assert summary["findings_by_owner"]["soft"]["planner"] == 1
    assert not {"places", "itinerary", "source_registry", "original_request", "user_input"} & summary.keys()


def test_forty_scripted_decisions_cannot_exceed_thirty_trip_steps(state, monkeypatch):
    state |= {"round_steps": 0, "trip_steps": 0, "correction_cycles": 0, "critique_version": None}
    model = use_model(
        monkeypatch, [decision("planner")] * 40,
        by_schema={RequestPatch: RequestPatch(change_types=["order"], other_notes="Adjust order.")},
    )
    for _ in range(40):
        result = orchestrator(state, {})
        assert result["next"] != "save"
        state = updated(state, result)
        if state["next"] in {"recovery", "traveler", "stop"}:
            break
        if state["round_steps"] == 10:
            state = updated(state, apply_revision(state, "Adjust order."))
    else:
        pytest.fail("Forty decisions failed to stop")
    assert state["trip_steps"] == 30
    assert sum(schema is OrchestratorDecision for schema, _, _ in model.calls) == 29


def test_initial_request_sets_original_and_current_independently(monkeypatch):
    request = TripRequest(city="Lisbon", days=3, budget="mid", interests=[], pace="normal")
    model = use_model(monkeypatch, [ParsedTripRequest(request=request, city_status="clear")])
    config = {"configurable": {"marker": "forwarded"}}
    result = orchestrator({"user_input": "3 days in Lisbon", "request": None}, config)
    assert result["request"] == result["original_request"] == request
    assert result["request"] is not result["original_request"]
    assert result["next"] == "both" and result["research_round"] == 1
    assert model.calls[0][2] == config
    assert all(kwargs == {"method": "function_calling"} for _, kwargs in model.bindings)


@pytest.mark.parametrize("status", ["missing", "vague", "misspelled", "ambiguous"])
def test_uncertain_city_stops_with_clarification(monkeypatch, status):
    use_model(monkeypatch, [ParsedTripRequest(request=None, city_status=status)])
    result = orchestrator({"user_input": "3 days somewhere warm"}, {})
    assert result["next"] == "stop" and "clarify" in result["warnings"][0]
    assert "request" not in result and "original_request" not in result


def test_silently_corrected_city_is_rejected_even_if_model_says_clear(monkeypatch):
    request = TripRequest(city="Lisbon", days=3, budget="mid", interests=[], pace="normal")
    use_model(monkeypatch, [ParsedTripRequest(request=request, city_status="clear")])
    assert orchestrator({"user_input": "3 days in Lisbom"}, {})["next"] == "stop"


def test_model_city_invention_is_rejected(monkeypatch):
    request = TripRequest(city="Lisbon", days=3, budget="mid", interests=[], pace="normal")
    use_model(monkeypatch, [ParsedTripRequest(request=request, city_status="clear")])
    assert orchestrator({"user_input": "3 days somewhere sunny"}, {})["next"] == "stop"


@pytest.mark.parametrize("invalid", [{"next": "save"}, {"next": "planner", "instructions": []}])
def test_invalid_decisions_twice_return_agent_error(state, monkeypatch, invalid):
    model = use_model(monkeypatch, [invalid])
    result = orchestrator(state, {})
    assert result["next"] == "recovery" and result["agent_error"].agent == "orchestrator"
    assert len(model.calls) == 2


def test_provider_failure_does_not_retry_or_leak_secrets(state, monkeypatch, caplog):
    model = use_model(monkeypatch, [RuntimeError("private-secret")])
    result = orchestrator(state, {})
    assert result["agent_error"] and result["next"] == "recovery"
    assert len(model.calls) == 1
    assert "private-secret" not in caplog.text
    assert "private-secret" not in result["agent_error"].model_dump_json()


def test_revision_provider_failure_keeps_request(state, monkeypatch):
    before = copy.deepcopy(state)
    use_model(monkeypatch, [RuntimeError("private-secret")])
    result = apply_revision(state, "change my plans")
    assert result["next"] == "recovery" and result["agent_error"]
    assert "request" not in result and state == before


def test_caps_and_agent_error_do_not_call_model(state, monkeypatch):
    factory = Mock(side_effect=AssertionError("Model must not be called"))
    monkeypatch.setattr(module, "get_llm", factory)
    assert orchestrator(state | {"trip_steps": 30}, {})["trip_steps"] == 30
    result = orchestrator(state | {"agent_error": AgentError(agent="planner", message="failed")}, {})
    assert result["next"] == "recovery"
    factory.assert_not_called()


def test_native_structured_output_for_decision_has_no_operational_tools(state, monkeypatch):
    model = ScriptedModel(responses=[structured(OrchestratorDecision, decision("critic"))])
    monkeypatch.setattr(module, "get_llm", lambda: model)
    assert orchestrator(state, {})["next"] == "critic"
    assert model.bound_names == [["OrchestratorDecision"]]


def test_native_structured_patch_preserves_omitted_date(state, monkeypatch):
    model = ScriptedModel(responses=[structured(RequestPatch, {"pace": "packed"})])
    monkeypatch.setattr(module, "get_llm", lambda: model)
    result = apply_revision(state, "change relaxed to packed")
    assert result["request"].pace == "packed"
    assert result["request"].start_date == state["request"].start_date
    assert model.bound_names == [["RequestPatch"]]


def test_default_null_fields_do_not_clear_unrelated_date(state, monkeypatch):
    # Some models emit every optional field, even though asked to omit unchanged ones.
    use_model(monkeypatch, [{
        "city": None, "days": None, "start_date": None, "budget": None,
        "interests": None, "pace": "packed", "other_notes": None,
        "change_types": ["pace"],
    }])
    result = apply_revision(state, "change relaxed to packed")
    assert result["request"].pace == "packed"
    assert result["request"].start_date == state["request"].start_date
    assert "weather" not in result


def test_two_sendbacks_are_allowed_then_correction_limit_stops(state, monkeypatch):
    state["correction_cycles"] = 0
    state["critique"] = Critique(hard_failures=[Finding(issue="Reorder.", owner="planner")])
    model = use_model(monkeypatch, [decision("planner")])
    first = orchestrator(state, {})
    assert first["next"] == "planner" and first["correction_cycles"] == 1
    state = updated(state, first)
    # A new draft and its new Critic check trigger the second send-back.
    state["draft_version"] = state["critique_version"] = 9
    second = orchestrator(state, {})
    assert second["next"] == "planner" and second["correction_cycles"] == 2
    state = updated(state, second)
    terminal = orchestrator(state, {})
    assert terminal["next"] == "recovery" and len(model.calls) == 2


def test_research_dispatch_invalidates_prior_approval_and_critique_version(state, monkeypatch):
    use_model(monkeypatch, [decision("food")])
    result = orchestrator(state, {})
    assert result["approved_version"] is None and result["critique_version"] is None
    assert result["decision"] is None
    assert state["critique_version"] == 8  # original state was not mutated


def test_existing_over_limit_counters_never_decrease(state, monkeypatch):
    factory = Mock(side_effect=AssertionError("No further model work at cap"))
    monkeypatch.setattr(module, "get_llm", factory)
    result = orchestrator(state | {"trip_steps": 35, "round_steps": 15}, {})
    assert result["trip_steps"] == 35 and result["round_steps"] == 15
    factory.assert_not_called()


def test_retry_keeps_open_finding_instructions_even_if_model_omits_them(state, monkeypatch):
    state["decision"] = "retry"
    state["instructions"] = {"planner": "Fix the duplicated morning stop."}
    use_model(monkeypatch, [decision("planner")])
    result = orchestrator(state, {})
    assert "Fix the duplicated morning stop." in result["instructions"]["planner"]
