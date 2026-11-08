"""MOCKED: full LangGraph, real interrupts, SQLite restart, and local exports."""

from collections import Counter
from pathlib import Path
from threading import Barrier
from unittest.mock import Mock

import pytest
import requests
from langgraph.checkpoint.memory import InMemorySaver

from src import export
from src import graph as workflow
from src.agents import orchestrator as supervisor
from src.agents._checks import hard_checks
from src.agents.orchestrator import ParsedTripRequest
from src.state import (
    AgentError,
    Critique,
    DayPlan,
    Finding,
    Itinerary,
    OrchestratorDecision,
    Place,
    ResearchResult,
    SourceRecord,
    Stop,
    TripRequest,
)

REQUEST = "1 day in Lisbon, relaxed, history"


def paused(result):
    return result["__interrupt__"][0].value


def config(thread):
    return {"configurable": {"thread_id": thread}}


@pytest.fixture
def facts():
    places = []
    registry = {}
    for index, (name, category, found_by) in enumerate([
        ("Museum Alpha", "sight", "sights"),
        ("Gallery Beta", "activity", "sights"),
        ("Restaurant Gamma", "food", "food"),
    ]):
        quote = f"{name} is an indoor venue in Lisbon."
        place = Place(
            name=name, category=category, found_by=found_by, description=quote,
            source_id=f"source-{index}", source_url=f"https://example.org/{index}",
            source_title=name, retrieved_at="2026-11-04T12:00:00Z", evidence_quote=quote,
        )
        places.append(place)
        registry[place.source_id] = SourceRecord(
            url=place.source_url, retrieved_at=place.retrieved_at, text=quote,
        )
    return places, registry


@pytest.fixture
def fake_agents(monkeypatch, tmp_path, facts):
    """Use real supervisor guards and fake specialists; never call providers."""
    for key in ("NEBIUS_API_KEY", "TAVILY_API_KEY", "LANGSMITH_API_KEY", "LANGSMITH_TRACING"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        requests.sessions.Session, "request",
        Mock(side_effect=AssertionError("Mocked graph attempted network access")),
    )
    monkeypatch.setattr(export, "OUTPUTS_DIR", tmp_path / "outputs")
    calls = Counter()
    controls = {
        "failed_focus": None, "planner_error": False, "hard_failure": False,
        "unchecked": False, "observed_retry": None,
    }
    barrier = Barrier(2)
    places, registry = facts
    trip = TripRequest(
        city="Lisbon", days=1, budget="mid", interests=["history"], pace="relaxed",
    )

    def extract(schema, prompt, payload, run_config):
        calls["llm"] += 1
        if schema is ParsedTripRequest:
            return ParsedTripRequest(request=trip, city_status="clear")
        if schema is OrchestratorDecision:
            if payload["coverage"]["food"]["available"] == 0 or not payload["has_draft"]:
                next_agent = "planner"
            elif payload["critique_version"] != payload["draft_version"]:
                next_agent = "traveler" if controls["unchecked"] else "critic"
            else:
                next_agent = "traveler"
            return OrchestratorDecision(next=next_agent, reason="Mocked supervisor choice")
        raise AssertionError("Unexpected extraction")

    def researcher_factory(focus):
        def researcher(state, run_config):
            calls[focus] += 1
            if len(state["dispatched"]) == 2:
                barrier.wait(timeout=5)  # proves both Send branches really run concurrently
            if controls["failed_focus"] == focus:
                return {f"{focus}_result": ResearchResult(
                    status="failed", error="fake_failure", round_id=state["research_round"],
                )}
            found = [place for place in places if place.found_by == focus]
            return {
                f"{focus}_result": ResearchResult(
                    status="ok", places=found, round_id=state["research_round"],
                ),
                "source_registry": {place.source_id: registry[place.source_id] for place in found},
            }
        return researcher

    def fake_planner(state, run_config):
        calls["planner"] += 1
        if state.get("decision") == "retry":
            controls["observed_retry"] = dict(state["instructions"])
        if controls["planner_error"]:
            return {"agent_error": AgentError(agent="planner", message="fake_failure")}
        itinerary = Itinerary(
            city="Lisbon", budget_tips=["Use transit.", "Consider free hours."],
            days=[DayPlan(
                day_number=1, theme="Art and food",
                stops=[Stop(
                    time_slot=slot, place_name=place.name, note="Visit this indoor venue.",
                    source_url=place.source_url,
                ) for slot, place in zip(
                    ("morning", "afternoon", "evening"), state["places"], strict=True,
                )],
            )],
        )
        return {
            "itinerary": itinerary, "draft_version": state["draft_version"] + 1,
            "approved_version": None, "agent_error": None, "weather": None,
        }

    def fake_critic(state, run_config):
        calls["critic"] += 1
        hard = hard_checks(state)
        if controls["hard_failure"]:
            hard.append(Finding(issue="A stop has invalid evidence.", owner="researchers"))
        return {
            "critique": Critique(hard_failures=hard, soft_findings=[
                Finding(issue="Opening hours may have changed.", owner="planner"),
            ]),
            "critique_version": state["draft_version"], "agent_error": None,
        }

    original_merge = workflow.merge_research

    def merge(state):
        calls["merge"] += 1
        return original_merge(state)

    monkeypatch.setattr(supervisor, "_structured", extract)
    monkeypatch.setattr(workflow, "make_researcher", researcher_factory)
    monkeypatch.setattr(workflow, "planner", fake_planner)
    monkeypatch.setattr(workflow, "critic", fake_critic)
    monkeypatch.setattr(workflow, "merge_research", merge)
    return calls, controls, trip, original_merge


@pytest.fixture
def graph(fake_agents):
    return workflow.build_graph(InMemorySaver())


def test_both_branches_run_and_merge_once(graph, fake_agents):
    result = workflow.start_trip(graph, REQUEST, "both-branch")
    assert paused(result)["kind"] == "approval"
    calls = fake_agents[0]
    assert calls["sights"] == calls["food"] == calls["merge"] == 1
    assert len(result["places"]) == 3 and len(result["source_registry"]) == 3
    assert result["sights_result"] is result["food_result"] is None
    assert result["dispatched"] == []


def test_failed_branch_keeps_successful_places_and_warns(graph, fake_agents):
    fake_agents[1]["failed_focus"] = "food"
    result = workflow.start_trip(graph, REQUEST, "failed-food")
    assert paused(result)["kind"] == "recovery"
    assert len(result["places"]) == 2
    assert all(place.category != "food" for place in result["places"])
    assert any("food research failed" in warning for warning in result["warnings"])
    assert fake_agents[0]["sights"] == 1 and fake_agents[0]["food"] == 2


def test_food_only_merge_keeps_every_sight_and_dedupes_sources(fake_agents, facts):
    places, registry = facts
    extra = places[2].model_copy(update={
        "source_id": "food-extra", "source_url": "https://another.example.org/gamma",
    })
    registry = registry | {"food-extra": SourceRecord(
        url=extra.source_url, retrieved_at=extra.retrieved_at, text=extra.evidence_quote,
    )}
    state = {
        "places": places, "dispatched": ["food"], "research_round": 4,
        "food_result": ResearchResult(status="ok", places=[places[2], extra], round_id=4),
        "sights_result": None, "source_registry": registry,
    }
    result = fake_agents[3](state)
    assert result["places"][:2] == places[:2]
    assert len(result["places"]) == 3
    assert result["places"][2].extra_sources == ["https://another.example.org/gamma"]
    assert places[2].extra_sources == []


def test_stale_round_is_ignored_without_erasing_existing_places(fake_agents, facts):
    places, _ = facts
    stale = places[2].model_copy(update={"name": "Stale venue"})
    result = fake_agents[3]({
        "places": places, "dispatched": ["food"], "research_round": 4,
        "food_result": ResearchResult(status="ok", places=[stale], round_id=3),
    })
    assert result["places"] == places
    assert any("stale" in warning for warning in result["warnings"])
    assert result["food_result"] is None


@pytest.mark.parametrize("mode", ["hard_failure", "planner_error", "unchecked"])
def test_hard_failures_errors_and_unchecked_drafts_never_pause_for_approval(graph, fake_agents, mode):
    fake_agents[1][mode] = True
    result = workflow.start_trip(graph, REQUEST, f"blocked-{mode}")
    assert paused(result)["kind"] == "recovery"
    assert result["saved_path"] is None
    rejected = workflow.resume_trip(graph, result["thread_id"], "approve", draft_version=1)
    assert paused(rejected)["kind"] == "recovery"
    assert rejected["saved_path"] is None


def test_current_approval_exports_complete_manifest(graph):
    initial = workflow.start_trip(graph, REQUEST, "approve-current")
    version = paused(initial)["draft_version"]
    result = workflow.resume_trip(graph, initial["thread_id"], "approve", version)
    assert "__interrupt__" not in result
    assert result["approved_version"] == result["critique_version"] == result["draft_version"]
    base = str(result["saved_path"]).removesuffix(".json")
    assert export.export_is_complete(base)
    assert Path(result["saved_path"]).exists()
    assert result["export_manifest"]["draft_version"] == version


def test_stale_approval_warns_and_pauses_again_without_export(graph):
    initial = workflow.start_trip(graph, REQUEST, "stale-approval")
    stale = workflow.resume_trip(graph, initial["thread_id"], "approve", draft_version=0)
    assert paused(stale)["kind"] == "approval"
    assert any("Stale approval" in warning for warning in stale["warnings"])
    assert stale["saved_path"] is None and stale["approved_version"] is None
    current = workflow.resume_trip(graph, initial["thread_id"], "approve", draft_version=1)
    assert current["saved_path"]


@pytest.mark.parametrize("change", [
    {"approved_version": None}, {"approved_version": 99},
    {"critique_version": None}, {"decision": "revise"},
    {"agent_error": AgentError(agent="critic", message="failed")},
])
def test_save_independently_rejects_invalid_approval(graph, monkeypatch, change):
    initial = workflow.start_trip(graph, REQUEST, "save-guard")
    state = initial | {"decision": "approve", "approved_version": 1} | change
    writer = Mock()
    writer.invoke.side_effect = AssertionError("Writer must not run")
    monkeypatch.setattr(workflow, "save_itinerary", writer)
    command = workflow.save(state, config(initial["thread_id"]))
    assert command.goto in {"recovery", "human_review", "orchestrator"}
    writer.invoke.assert_not_called()


def test_export_failure_retry_preserves_approval_and_counters(graph, monkeypatch):
    fail_next_export(monkeypatch)
    initial = workflow.start_trip(graph, REQUEST, "retry-export")
    failed = workflow.resume_trip(
        graph, initial["thread_id"], "approve", draft_version=1,
    )
    assert paused(failed)["kind"] == "export_failed"
    assert failed["approved_version"] == 1 and failed["decision"] == "approve"
    assert failed["saved_path"] is None
    counters = (failed["round_steps"], failed["trip_steps"])
    done = workflow.resume_trip(graph, initial["thread_id"], "retry_export")
    assert done["saved_path"]
    assert (done["round_steps"], done["trip_steps"]) == counters
    assert export.export_is_complete(str(done["saved_path"]).removesuffix(".json"))



def test_trip_cap_rejects_retry_then_allows_new_trip(graph, fake_agents):
    thread = "at-trip-cap"
    state = workflow._initial_state(REQUEST) | {
        "request": fake_agents[2], "original_request": fake_agents[2], "trip_steps": 30,
    }
    initial = graph.invoke(state, config=config(thread))
    assert paused(initial)["options"] == ["new_trip", "cancel"]
    rejected = workflow.resume_trip(graph, thread, "retry")
    assert paused(rejected)["options"] == ["new_trip", "cancel"]
    assert rejected["trip_steps"] == 30
    done = workflow.resume_trip(graph, thread, "new_trip")
    assert done["new_trip"] is True and done["trip_steps"] == 30
    assert "__interrupt__" not in done


def test_sqlite_restart_lists_pending_trip_and_resumes_export(fake_agents, tmp_path):
    database = tmp_path / "restart.sqlite"
    saver = workflow.get_checkpointer(database)
    first = workflow.build_graph(saver)
    initial = workflow.start_trip(first, REQUEST, "restart-thread")
    assert paused(initial)["kind"] == "approval"
    counters = initial["round_steps"], initial["trip_steps"]
    saver.conn.close()

    replacement = workflow.get_checkpointer(database)
    try:
        second = workflow.build_graph(replacement)
        trips = workflow.list_trips(second, replacement)
        assert len(trips) == 1
        assert trips[0] == {
            "thread_id": "restart-thread", "city": "Lisbon",
            "last_update": trips[0]["last_update"], "paused_at": "approval",
        }
        assert trips[0]["last_update"]
        done = workflow.resume_trip(second, "restart-thread", "approve", draft_version=1)
        assert done["saved_path"]
        assert export.export_is_complete(str(done["saved_path"]).removesuffix(".json"))
        assert (done["round_steps"], done["trip_steps"]) == counters
        assert workflow.list_trips(second, replacement) == []
    finally:
        replacement.conn.close()


def test_list_dedupes_threads_and_delete_removes_checkpoint_records(fake_agents, tmp_path):
    saver = workflow.get_checkpointer(tmp_path / "listing.sqlite")
    try:
        graph = workflow.build_graph(saver)
        workflow.start_trip(graph, REQUEST, "one-thread")
        workflow.start_trip(graph, REQUEST, "two-thread")
        assert {item["thread_id"] for item in workflow.list_trips(graph, saver)} == {
            "one-thread", "two-thread",
        }
        workflow.delete_trip(saver, "one-thread")
        assert [item["thread_id"] for item in workflow.list_trips(graph, saver)] == ["two-thread"]
        assert not graph.get_state(config("one-thread")).values
    finally:
        saver.conn.close()


def test_streams_native_interrupt_chunks_with_kind(graph):
    chunks = list(workflow.stream_trip(graph, REQUEST, "stream-trip"))
    assert chunks[-1]["__interrupt__"][0].value["kind"] == "approval"
    assert all(chunk["thread_id"] == "stream-trip" for chunk in chunks)
    assert any("orchestrator" in chunk for chunk in chunks)
    resumed = list(workflow.stream_trip(
        graph, thread_id="stream-trip", decision="approve", draft_version=1,
    ))
    assert any("save" in chunk for chunk in resumed)
    assert graph.get_state(config("stream-trip")).values["saved_path"]


@pytest.mark.parametrize("kind", ["approval", "export_failed", "recovery"])
def test_cancel_from_each_pause_never_exports(graph, fake_agents, kind, monkeypatch):
    fail_next_export(monkeypatch)
    if kind == "recovery":
        fake_agents[1]["planner_error"] = True
    initial = workflow.start_trip(graph, REQUEST, f"cancel-{kind}")
    if kind == "export_failed":
        initial = workflow.resume_trip(
            graph, initial["thread_id"], "approve", draft_version=1,
        )
    assert paused(initial)["kind"] == kind
    result = workflow.resume_trip(graph, initial["thread_id"], "cancel")
    assert result["cancelled"] is True and result["saved_path"] is None
    assert result["approved_version"] is None
    assert not graph.get_state(config(initial["thread_id"])).next



def test_diagram_contains_all_required_nodes(graph, tmp_path):
    path = workflow.export_graph_diagram(graph, tmp_path / "graph.mmd")
    mermaid = path.read_text("utf-8")
    for node in (
        "orchestrator", "sights_researcher", "food_researcher", "merge_research",
        "planner", "critic", "human_review", "recovery", "save", "export_failed", "end_cancelled",
    ):
        assert node in mermaid


def test_start_rejects_existing_thread_and_resume_requires_pause(graph):
    workflow.start_trip(graph, REQUEST, "existing")
    with pytest.raises(ValueError, match="already exists"):
        workflow.start_trip(graph, REQUEST, "existing")
    with pytest.raises(ValueError, match="no pending interrupt"):
        workflow.resume_trip(graph, "missing", "approve", draft_version=1)


def test_cli_displays_draft_sources_findings_and_exports(fake_agents, tmp_path, monkeypatch, caplog):
    monkeypatch.setattr("builtins.input", lambda prompt: "approve")
    caplog.set_level("INFO", logger="src.graph")
    assert workflow.main([REQUEST, "--checkpoint", str(tmp_path / "cli.sqlite")]) == 0
    assert "Draft v1" in caplog.text and "https://example.org/" in caplog.text
    assert "Soft finding:" in caplog.text and "Exported:" in caplog.text
    assert "Step 1 -> both" in caplog.text


@pytest.mark.parametrize("kind", ["approval", "recovery", "export_failed"])
def test_list_trips_reports_each_pending_pause_kind(fake_agents, kind, monkeypatch):
    fail_next_export(monkeypatch)
    saver = InMemorySaver()
    graph = workflow.build_graph(saver)
    if kind == "recovery":
        fake_agents[1]["planner_error"] = True
    result = workflow.start_trip(graph, REQUEST, f"listed-{kind}")
    if kind == "export_failed":
        result = workflow.resume_trip(
            graph, result["thread_id"], "approve", draft_version=1,
        )
    assert workflow.list_trips(graph, saver)[0]["paused_at"] == kind
    workflow.resume_trip(graph, result["thread_id"], "cancel")
    assert workflow.list_trips(graph, saver) == []



def test_recovery_retry_clears_error_and_can_complete(graph, fake_agents):
    fake_agents[1]["planner_error"] = True
    initial = workflow.start_trip(graph, REQUEST, "retry-agent")
    assert paused(initial)["kind"] == "recovery" and initial["agent_error"]
    fake_agents[1]["planner_error"] = False
    retried = workflow.resume_trip(graph, initial["thread_id"], "retry")
    assert retried["agent_error"] is None and paused(retried)["kind"] == "approval"
    assert workflow.resume_trip(graph, initial["thread_id"], "approve", draft_version=1)["saved_path"]


def test_human_revision_runs_only_after_resume_and_invalid_patch_pauses_again(graph, monkeypatch):
    revision = Mock(return_value={"warnings": ["Please rephrase the revision."]})
    monkeypatch.setattr(workflow, "apply_revision", revision)
    initial = workflow.start_trip(graph, REQUEST, "invalid-revision")
    revision.assert_not_called()
    result = workflow.resume_trip(
        graph, initial["thread_id"], "revise", feedback="12 days",
    )
    assert paused(result)["kind"] == "approval"
    assert result["request"] == initial["request"] and result["original_request"] == initial["original_request"]
    assert result["draft_version"] == initial["draft_version"]
    assert revision.call_count == 1


def test_export_failed_rejects_non_export_action_without_counting_steps(graph, monkeypatch):
    fail_next_export(monkeypatch)
    initial = workflow.start_trip(graph, REQUEST, "invalid-export-action")
    failed = workflow.resume_trip(
        graph, initial["thread_id"], "approve", draft_version=1,
    )
    rejected = workflow.resume_trip(graph, initial["thread_id"], "retry")
    assert paused(rejected)["kind"] == "export_failed"
    assert rejected["saved_path"] is None and rejected["approved_version"] == 1
    assert (rejected["trip_steps"], rejected["round_steps"]) == (
        failed["trip_steps"], failed["round_steps"],
    )



def test_city_tags_are_set_after_parsing(graph, monkeypatch, fake_agents):
    seen = []
    original = workflow.planner

    def capture(state, run_config):
        seen.append(run_config)
        return original(state, run_config)

    monkeypatch.setattr(workflow, "planner", capture)
    rebuilt = workflow.build_graph(InMemorySaver())
    workflow.start_trip(rebuilt, REQUEST, "tagged-trip")
    assert seen[0]["run_name"] == "trip-Lisbon"
    assert seen[0]["tags"] == ["travel-planner", "Lisbon", "thread:tagged-trip"]
    assert "failure_mode" not in rebuilt.get_state(config("tagged-trip")).values


def fail_next_export(monkeypatch):
    """Mock a disk failure, never add a nonstandard production failure_mode."""
    original = export._write_pair
    attempts = 0

    def write_pair(*args):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("Mocked export disk failure")
        return original(*args)

    monkeypatch.setattr(export, "_write_pair", write_pair)
