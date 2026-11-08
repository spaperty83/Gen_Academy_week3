"""MOCKED real-graph Critic budget and retry regressions."""

from unittest.mock import Mock

import pytest
import requests

from src.graph import resume_trip, trip_screen
from tests.test_failures import FailureRun, output


@pytest.fixture
def run(monkeypatch, tmp_path):
    for key in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGCHAIN_TRACING_V2"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(requests.sessions.Session, "request", Mock(side_effect=AssertionError("No network")))
    return FailureRun(monkeypatch, tmp_path / "outputs")


def test_critic_four_checks_then_structured_review_reaches_approval(run):
    respond = run.respond

    def bounded_checks(role, payload, messages, call_number, bound):
        if role == "critic" and call_number <= 4:
            return output("web_search", {"query": payload["doubtful_stops"][0]["place_name"]})
        return respond(role, payload, messages, call_number, bound)

    run.respond = bounded_checks
    result = run.start()
    assert trip_screen(result)["kind"] == "approval"
    assert result["agent_error"] is None
    assert result["critique_version"] == result["draft_version"]
    assert not result["critique"].hard_failures
    assert run.calls["service:sights:1"] >= 1
    assert result["saved_path"] is None


def test_retry_failed_critic_rechecks_saved_draft_instead_of_skipping_review(run):
    run.invalid_agent = "critic"
    failed = run.start()
    assert trip_screen(failed)["kind"] == "recovery"
    assert failed["agent_error"].agent == "critic"
    calls_before = run.calls["model:critic"]
    run.invalid_agent = None
    retried = resume_trip(run.graph, failed["thread_id"], "retry")
    assert run.calls["model:critic"] > calls_before
    assert trip_screen(retried)["kind"] == "approval"
    assert retried["agent_error"] is None
    assert retried["draft_version"] == failed["draft_version"]
    assert retried["itinerary"] == failed["itinerary"]
    assert retried["saved_path"] is None


def test_interest_revision_checks_new_draft_before_repeated_research(run):
    first = run.start()
    assert trip_screen(first)["kind"] == "approval"
    respond = run.respond

    def revision_routing(role, payload, messages, call_number, bound):
        if role == "orchestrator" and bound[0] == "RequestPatch":
            return output("RequestPatch", {"interests": ["history", "beaches"], "change_types": ["interests"]})
        if (role == "orchestrator" and bound[0] == "OrchestratorDecision"
                and payload["human_feedback"] and payload["has_draft"]
                and payload["critique_version"] != payload["draft_version"]):
            return output("OrchestratorDecision", {"next": "sights", "instructions": {},
                                                  "reason": "Find more beach places"})
        if role == "researcher":
            assert payload["human_feedback"] == "Add beaches to my interests"
            assert "beaches" in payload["request"]["interests"]
        return respond(role, payload, messages, call_number, bound)

    run.respond = revision_routing
    revised = resume_trip(run.graph, first["thread_id"], "revise", feedback="Add beaches to my interests")
    assert trip_screen(revised)["kind"] == "approval"
    assert revised["critique_version"] == revised["draft_version"] == first["draft_version"] + 1
    assert run.dispatches == {"sights": 2, "food": 1}
    assert revised["original_request"] == first["original_request"]
    assert revised["saved_path"] is None
    assert any("before dispatching more research" in entry["reason"] for entry in revised["delegation_log"])


def test_supervisor_repeated_critic_choice_checks_each_draft_once(run):
    respond = run.respond

    def keep_requesting_critic(role, payload, messages, call_number, bound):
        if (role == "orchestrator" and bound[0] == "OrchestratorDecision" and payload["has_draft"]):
            return output("OrchestratorDecision", {"next": "critic", "instructions": {},
                                                  "reason": "Needs checking again"})
        return respond(role, payload, messages, call_number, bound)

    run.respond = keep_requesting_critic
    first = run.start()
    assert trip_screen(first)["kind"] == "approval"
    assert run.calls["model:critic"] == 2  # one tool call and one structured review
    assert first["critique_version"] == first["draft_version"]
    revised = resume_trip(run.graph, first["thread_id"], "revise", feedback="change Lisbon to Paris")
    assert trip_screen(revised)["kind"] == "approval"
    assert run.calls["model:critic"] == 4
    assert revised["critique_version"] == revised["draft_version"] == 2
    assert revised["agent_error"] is None
    assert revised["saved_path"] is None
    assert any("already checked" in row["reason"] for row in revised["delegation_log"])


def test_saved_failure_pattern_three_then_two_tool_requests_reaches_approval(run):
    from langchain_core.messages import AIMessage

    respond = run.respond

    def batched_checks(role, payload, messages, call_number, bound):
        if role == "critic" and call_number in (1, 2):
            target = payload["doubtful_stops"][0]
            names = ["web_search", "web_search", "read_page"] if call_number == 1 else ["web_search", "read_page"]
            return AIMessage(content="", tool_calls=[{
                "name": name,
                "args": ({"query": "spotcheck " + target["place_name"]} if name == "web_search"
                         else {"url": target["source_url"]}),
                "id": f"critic-{call_number}-{index}", "type": "tool_call",
            } for index, name in enumerate(names)])
        return respond(role, payload, messages, call_number, bound)

    run.respond = batched_checks
    result = run.start()
    assert trip_screen(result)["kind"] == "approval"
    assert result["agent_error"] is None
    assert result["critique_version"] == result["draft_version"]
    assert not result["critique"].hard_failures
    assert any("tool failed" in finding.issue.casefold() for finding in result["critique"].soft_findings)
    assert run.calls["critic_web"] == 3
    assert run.calls["service:extract"] == 1  # second extract is declined before execution
    assert result["saved_path"] is None


def test_retry_exhausted_research_explains_gap_without_model_calls_or_steps(run):
    run.short_focus = "sights"
    failed = run.start()
    assert trip_screen(failed)["kind"] == "recovery"
    assert failed["researcher_attempts"]["sights"] == 2
    assert any("required places" in warning for warning in trip_screen(failed)["warnings"])
    before = run.calls.copy()
    retried = resume_trip(run.graph, failed["thread_id"], "retry")
    assert trip_screen(retried)["kind"] == "recovery"
    assert run.calls == before
    assert retried["trip_steps"] == failed["trip_steps"]
    assert retried["researcher_attempts"] == failed["researcher_attempts"]
    assert any("Retry cannot reset" in warning for warning in trip_screen(retried)["warnings"])
    assert retried["places"] == failed["places"]


def test_planner_five_alternative_tool_calls_corrects_once_and_reaches_review(run):
    from langchain_core.messages import AIMessage

    respond = run.respond
    duplicated = False

    def alternatives(role, payload, messages, call_number, bound):
        nonlocal duplicated
        answer = respond(role, payload, messages, call_number, bound)
        if (role == "planner" and not duplicated and answer.tool_calls
                and answer.tool_calls[0]["name"] == "Itinerary"):
            duplicated = True
            original = answer.tool_calls[0]
            return AIMessage(content="", tool_calls=[original | {"id": f"alternative-{i}"} for i in range(5)])
        return answer

    run.respond = alternatives
    result = run.start()
    assert duplicated
    assert trip_screen(result)["kind"] == "approval"
    assert result["agent_error"] is None
    assert result["critique_version"] == result["draft_version"] == 1
    assert result["saved_path"] is None
