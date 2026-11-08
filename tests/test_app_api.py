"""MOCKED API integration for the UI: real graph, native pauses and local files."""

from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

from src import graph as api
from tests.test_failures import FailureRun


@pytest.fixture
def run(monkeypatch, tmp_path):
    """Keep graph/agents/parsers/tools real, but prevent provider calls."""
    for key in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGCHAIN_TRACING_V2"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        requests.sessions.Session, "request",
        Mock(side_effect=AssertionError("Mocked API test attempted network access")),
    )
    return FailureRun(monkeypatch, tmp_path / "outputs")


def test_get_trip_opens_pause_read_only_and_returns_detached_data(run):
    started = run.start()
    before = run.calls.copy()
    saved = api.get_trip(run.graph, started["thread_id"])
    assert saved["screen"]["kind"] == "approval"
    assert saved["__interrupt__"][0].value["draft_version"] == saved["draft_version"]
    assert saved["request"]["city"] == "Lisbon"
    saved["request"]["city"] = "Changed only in UI memory"
    assert api.get_trip(run.graph, started["thread_id"])["request"]["city"] == "Lisbon"
    assert not api.get_export(run.graph, started["thread_id"])["complete"]
    assert run.calls == before


def test_get_export_validates_real_bytes_before_returning_download(run):
    started = run.start()
    done = api.resume_trip(
        run.graph, started["thread_id"], "approve", draft_version=started["draft_version"],
    )
    artifact = api.get_export(run.graph, started["thread_id"])
    assert artifact["complete"] and artifact["file_name"].endswith(".md")
    assert b"Lisbon" in artifact["markdown"]
    markdown = Path(done["saved_path"]).with_suffix(".md")
    assert artifact["markdown"] == markdown.read_bytes()
    markdown.write_bytes(b"Modified after export")
    invalid = api.get_export(run.graph, started["thread_id"])
    assert not invalid["complete"] and invalid["markdown"] is None
    assert done["export_manifest"]  # metadata alone never enables the badge/download


def test_public_stream_exposes_actual_tool_queries_and_native_pause(run):
    thread = "ui-progress"
    chunks = list(api.stream_trip(
        run.graph, "1 day in Lisbon, relaxed, history", thread_id=thread,
    ))
    events = [chunk["progress"] for chunk in chunks if "progress" in chunk]
    assert any(event["agent"] == "food_researcher" and event["tool"] == "web_search"
               and "Lisbon food" in event["detail"] for event in events)
    assert any(event["agent"] == "planner" and event["tool"] == "get_weather" for event in events)
    assert chunks[-1]["__interrupt__"][0].value["kind"] == "approval"
    assert all(chunk["thread_id"] == thread for chunk in chunks)
    saved = api.get_trip(run.graph, thread)
    assert saved["screen"]["draft_version"] == 1


def test_get_trip_reopens_sqlite_pause_after_saver_restart(run, tmp_path):
    database = tmp_path / "ui-restart.sqlite"
    first = api.get_checkpointer(database)
    try:
        run.graph = api.build_graph(first)
        result = run.start()
    finally:
        first.conn.close()
    second = api.get_checkpointer(database)
    try:
        graph = api.build_graph(second)
        trips = api.list_trips(graph, second)
        assert trips[0]["thread_id"] == result["thread_id"]
        saved = api.get_trip(graph, trips[0]["thread_id"])
        assert saved["screen"]["kind"] == trips[0]["paused_at"] == "approval"
        assert saved["screen"]["itinerary"]["city"] == "Lisbon"
        api.resume_trip(graph, saved["thread_id"], "approve", draft_version=saved["draft_version"])
        assert api.get_export(graph, saved["thread_id"])["complete"]
    finally:
        second.conn.close()
