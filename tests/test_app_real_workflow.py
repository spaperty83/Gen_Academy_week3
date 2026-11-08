"""MOCKED providers, real Streamlit buttons, graph, interrupts and export files."""

from pathlib import Path
from unittest.mock import Mock

import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import graph as api
from tests.test_failures import FailureRun


def test_revision_then_approve_button_exports_without_running_agents(monkeypatch, tmp_path):
    for key in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGCHAIN_TRACING_V2"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(requests.sessions.Session, "request", Mock(side_effect=AssertionError("No network")))
    run = FailureRun(monkeypatch, tmp_path / "outputs")
    monkeypatch.setattr(api, "get_checkpointer", lambda path: run.graph.checkpointer)
    monkeypatch.setattr(api, "build_graph", lambda saver: run.graph)
    monkeypatch.setattr(api, "get_city_images", lambda city: [{
        "url": f"https://images.test/{city}.jpg", "caption": f"{city} city photo",
        "source_url": f"https://tourism.test/{city}",
    }])
    st.cache_resource.clear()
    try:
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "src/app.py", default_timeout=20).run()
        app.text_input(key="request_text").set_value("1 day in Lisbon, relaxed, history")
        app.button(key="plan_trip").click().run()
        assert not app.exception and app.session_state["phase"] == "approval"
        thread = app.session_state["thread_id"]
        app.button(key="approval_revise").click().run()
        app.text_area(key="revision_text").set_value("change Lisbon to Paris")
        app.button(key="approval_revision_submit").click().run()
        assert not app.exception and app.session_state["phase"] == "approval"
        version = app.session_state["interrupt_payload"]["draft_version"]
        assert version == 2
        calls_before = run.calls.copy()
        steps_before = api.get_trip(run.graph, thread)["trip_steps"]
        app.button(key=f"approve_{thread}_{version}").click().run()
        assert not app.exception and app.session_state["phase"] == "done"
        assert run.calls == calls_before
        final = api.get_trip(run.graph, thread)
        assert final["trip_steps"] == steps_before
        assert final["approved_version"] == final["critique_version"] == final["draft_version"] == 2
        assert api.get_export(run.graph, thread)["complete"]
        assert "city_photos" not in final
        assert "images.test" not in Path(final["saved_path"]).read_text(encoding="utf-8")
        assert "images.test" not in api.get_export(run.graph, thread)["markdown"].decode("utf-8")
        assert any(":green-badge[complete]" in item.value for item in app.markdown)
        assert len(app.get("download_button")) == 1
    finally:
        st.cache_resource.clear()
