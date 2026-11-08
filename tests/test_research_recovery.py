"""MOCKED real-graph regression for repairable research evidence failures."""

from unittest.mock import Mock

import requests

from src.graph import trip_screen
from tests.test_failures import FailureRun


def test_bad_first_sights_quotes_corrected_without_second_dispatch(monkeypatch, tmp_path):
    for key in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGCHAIN_TRACING_V2"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(requests.sessions.Session, "request", Mock(side_effect=AssertionError("No network")))
    run = FailureRun(monkeypatch, tmp_path / "outputs")
    respond = run.respond
    corrupted = False

    def first_bad_quote(role, payload, messages, call_number, bound):
        nonlocal corrupted
        answer = respond(role, payload, messages, call_number, bound)
        if (role == "researcher" and payload["focus"] == "sights" and not corrupted
                and answer.tool_calls and answer.tool_calls[0]["name"] == "ResearchCandidates"):
            answer = answer.model_copy(deep=True)
            for place in answer.tool_calls[0]["args"]["places"]:
                place["evidence_quote"] = place["name"] + " has a made-up description."
            corrupted = True
        return answer

    run.respond = first_bad_quote
    result = run.start()
    assert corrupted
    assert trip_screen(result)["kind"] == "approval"
    assert run.dispatches == {"sights": 1, "food": 1}
    assert not result["critique"].hard_failures and result["agent_error"] is None
    assert not any("attempt limit" in warning for warning in result["warnings"])
    assert result["saved_path"] is None  # no approval or export in this diagnostic


def test_repeated_research_choice_with_enough_places_reaches_approval(monkeypatch, tmp_path):
    from tests.test_failures import output

    for key in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGCHAIN_TRACING_V2"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(requests.sessions.Session, "request", Mock(side_effect=AssertionError("No network")))
    run = FailureRun(monkeypatch, tmp_path / "outputs")
    respond = run.respond

    def repeat_research(role, payload, messages, call_number, bound):
        if (role == "orchestrator" and bound[0] == "OrchestratorDecision"
                and not payload["has_draft"]):
            return output("OrchestratorDecision", {"next": "both", "instructions": {}, "reason": "Find more"})
        return respond(role, payload, messages, call_number, bound)

    run.respond = repeat_research
    result = run.start()
    assert trip_screen(result)["kind"] == "approval"
    assert run.dispatches == {"sights": 1, "food": 1}
    assert any("coverage met" in entry["reason"] for entry in result["delegation_log"])
