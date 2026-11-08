"""MOCKED Streamlit UI tests; no providers, browser or credentials required."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import graph as api

APP = Path(__file__).resolve().parents[1] / "src" / "app.py"


class FakeAPI:
    """A public-API contract fake; UI tests cannot touch a graph or saver internals."""

    def __init__(self):
        self.photos_by_city = {}
        self.image_calls = []
        self.graph = object()
        self.saver = object()
        self.store = {}
        self.calls = []
        self.created = 0
        self.complete = True
        self.export_failure = False
        self.raise_error = False
        self.stop_stream = False
        self.recovery_options = ["revise", "retry", "cancel"]

    def get_city_images(self, city):
        self.image_calls.append(city)
        return copy.deepcopy(self.photos_by_city.get(city, []))

    def get_checkpointer(self, path):
        self.created += 1
        return self.saver

    def build_graph(self, saver):
        assert saver is self.saver
        return self.graph

    def list_trips(self, graph, saver):
        assert graph is self.graph and saver is self.saver
        return [{
            "thread_id": thread, "city": item["request"]["city"],
            "last_update": "2026-11-07T12:00:00+00:00", "paused_at": item["screen"]["kind"],
        } for thread, item in self.store.items()
            if item["screen"]["kind"] in {"approval", "recovery", "export_failed"}]

    def snapshot(self, thread, *, city="Lisbon", kind="approval", version=1, options=None):
        itinerary = {
            "city": city, "budget_tips": ["Use transit.", "Consider free hours."],
            "warnings": [], "days": [{
                "day_number": 1, "theme": "Food and art", "weather_note": None,
                "stops": [{
                    "time_slot": "morning", "place_name": f"{city} Museum",
                    "note": "Explore the collections.", "source_url": "https://example.org/museum",
                }],
            }],
        }
        warnings = ["Opening hours may have changed."] if kind == "approval" else []
        view = {"kind": kind, "warnings": warnings}
        if kind == "approval":
            view |= {
                "itinerary": itinerary, "draft_version": version,
                "options": ["approve", "revise", "cancel"],
                "soft_findings": [{"issue": warnings[0], "owner": "planner", "day": 1}],
                "delegation_log": [{"step": 1, "next": "food", "reason": "Find food"}],
            }
        elif kind == "recovery":
            view |= {
                "reason": "research short of coverage",
                "options": options or list(self.recovery_options),
                "hard_findings": [{"issue": "More evidenced places needed.", "owner": "researchers"}],
                "warnings": ["Search is unavailable."],
            }
        elif kind == "export_failed":
            view |= {
                "reason": "write_failed", "options": ["retry_export", "cancel"],
                "warnings": ["Export failed; approval is retained."],
            }
        return {
            "thread_id": thread, "request": {"city": city}, "draft_version": version,
            "itinerary": itinerary, "screen": view, "warnings": view["warnings"],
            "delegation_log": [{"step": 1, "next": "food", "reason": "Find food"}],
            "saved_path": None, "export_manifest": None, "new_trip": False, "cancelled": False,
        }

    def get_trip(self, graph, thread):
        assert graph is self.graph
        if thread not in self.store:
            raise ValueError("No saved trip")
        return copy.deepcopy(self.store[thread])

    def get_export(self, graph, thread):
        assert graph is self.graph
        self.calls.append({"read_export": thread})
        return {
            "path": self.store[thread]["saved_path"], "complete": self.complete,
            "manifest": {"present": True}, "markdown": b"# Approved itinerary\n",
            "file_name": "Lisbon_trip_v1.md",
        }

    def stream_trip(self, graph, user_input=None, thread_id=None, *, decision=None,
                    draft_version=None, feedback=None, failure_mode=None):
        assert graph is self.graph
        call = {
            "thread_id": thread_id, "user_input": user_input, "decision": decision,
            "draft_version": draft_version, "feedback": feedback, "failure_mode": failure_mode,
        }
        self.calls.append(call)
        if self.raise_error:
            raise RuntimeError("private-secret-provider-details")
        if decision is None:
            kind = "recovery" if failure_mode == "tavily" else "approval"
            item = self.snapshot(thread_id, kind=kind)
        else:
            old = self.store[thread_id]
            if decision == "revise":
                city = "Paris" if "Paris" in feedback else old["request"]["city"]
                item = self.snapshot(thread_id, city=city, version=old["draft_version"] + 1)
            elif decision == "approve" and draft_version != old["draft_version"]:
                item = copy.deepcopy(old)
                item["screen"]["warnings"].append("Stale approval rejected; review the current version.")
            elif decision == "approve" and self.export_failure:
                item = self.snapshot(thread_id, kind="export_failed")
            elif decision in {"approve", "retry_export"}:
                item = copy.deepcopy(old)
                item |= {
                    "saved_path": "outputs/Lisbon_trip_v1.json", "export_manifest": {"present": True},
                    "screen": {"kind": "exported", "options": [], "warnings": []}, "warnings": [],
                }
            elif decision == "cancel":
                item = copy.deepcopy(old)
                item |= {"cancelled": True, "screen": {"kind": "cancelled", "options": [], "warnings": []}}
            elif decision == "new_trip":
                item = copy.deepcopy(old)
                item |= {"new_trip": True, "screen": {"kind": "new_trip", "options": [], "warnings": []}}
            else:
                item = self.snapshot(thread_id, city=old["request"]["city"])
        self.store[thread_id] = item
        yield {
            "orchestrator": {
                "research_round": 2, "dispatched": ["food"],
                "delegation_log": [{"step": 1, "next": "food", "reason": "Find food"}],
            }, "thread_id": thread_id,
        }
        if self.stop_stream:
            st.stop()
        yield {"progress": {
            "kind": "tool_call", "agent": "food_researcher", "tool": "web_search", "detail": "Lisbon food",
        }, "thread_id": thread_id}
        yield {"merge_research": {
            "dispatched": [], "places": [{"category": "sight"}, {"category": "food"}],
        }, "thread_id": thread_id}
        yield {"planner": {"draft_version": item["draft_version"]}, "thread_id": thread_id}
        yield {"critic": {
            "critique_version": item["draft_version"], "critique": {"hard_failures": [], "soft_findings": []},
        }, "thread_id": thread_id}
        if item["screen"]["kind"] in {"approval", "recovery", "export_failed"}:
            yield {"__interrupt__": (SimpleNamespace(value=item["screen"]),), "thread_id": thread_id}
        else:
            yield {"save": {"saved_path": item["saved_path"]}, "thread_id": thread_id}


@pytest.fixture
def ui(monkeypatch):
    st.cache_resource.clear()
    fake = FakeAPI()
    for name in ("get_checkpointer", "build_graph", "list_trips", "get_trip", "get_export", "stream_trip", "get_city_images"):
        monkeypatch.setattr(api, name, getattr(fake, name))
    app = AppTest.from_file(APP, default_timeout=10)
    app.run()
    assert not app.exception
    yield app, fake
    st.cache_resource.clear()


def start(app):
    app.text_input(key="request_text").set_value("1 day in Lisbon, relaxed, history")
    app.button(key="plan_trip").click().run()
    assert not app.exception
    return app.session_state["thread_id"]


def badge_present(app):
    return any(":green-badge[complete]" in item.value for item in app.markdown)


def test_initial_setup_cached_and_new_trip_uses_fresh_uuid(ui):
    app, fake = ui
    first = app.session_state["thread_id"]
    app.run()
    app.sidebar.button(key="new_trip").click().run()
    assert app.session_state["thread_id"] != first
    assert fake.created == 1
    assert app.session_state["phase"] == "input"
    assert not fake.calls


def test_start_streams_once_and_approval_has_sources_warnings_and_budget_label(ui):
    app, fake = ui
    thread = start(app)
    assert app.session_state["phase"] == "approval"
    assert app.session_state["interrupt_payload"]["kind"] == "approval"
    assert any("Draft v1" in item.value for item in app.subheader)
    assert any("Opening hours" in item.value for item in app.warning)
    assert any("budget preference considered" in item.value for item in app.markdown)
    assert len(app.get("link_button")) == 1
    assert any("web_search 'Lisbon food'" in line for line in app.session_state["log_lines"])
    assert any("food replaced, sights kept" in line for line in app.session_state["log_lines"])
    run_log = next(section for section in app.expander if section.label == "Run log")
    assert any("web_search" in item.value for item in run_log.text)
    assert not app.get("status")  # temporary progress is removed before the result
    app.run()  # ordinary reruns must never replay the queued action
    starts = [call for call in fake.calls if call.get("user_input")]
    assert len(starts) == 1 and starts[0]["thread_id"] == thread


def test_approve_resumes_displayed_version_then_downloads_validated_markdown(ui):
    app, fake = ui
    thread = start(app)
    app.button(key=f"approve_{thread}_1").click().run()
    assert not app.exception and app.session_state["phase"] == "done"
    action = next(call for call in fake.calls if call.get("decision") == "approve")
    assert action["thread_id"] == thread and action["draft_version"] == 1
    assert badge_present(app)
    download = app.get("download_button")
    assert len(download) == 1 and download[0].proto.label == "Download Markdown"
    assert not download[0].proto.disabled
    assert any("outputs/Lisbon_trip_v1.json" in item.value for item in app.code)


def test_incomplete_manifest_never_gets_badge_or_download(ui):
    app, fake = ui
    thread = start(app)
    fake.complete = False  # a manifest exists, but its hashes do not match
    app.button(key=f"approve_{thread}_1").click().run()
    assert not badge_present(app) and not app.get("download_button")
    assert any("incomplete or has changed" in item.value for item in app.warning)


def test_complete_badge_is_rechecked_on_every_done_rerun(ui):
    app, fake = ui
    thread = start(app)
    app.button(key=f"approve_{thread}_1").click().run()
    assert badge_present(app)
    fake.complete = False
    app.run()
    assert not badge_present(app) and not app.get("download_button")


def test_city_revision_resumes_same_thread_and_displays_new_version(ui):
    app, fake = ui
    thread = start(app)
    before = len(fake.calls)
    app.button(key="approval_revise").click().run()
    assert len(fake.calls) == before
    app.text_area(key="revision_text").set_value("change Lisbon to Paris")
    app.button(key="approval_revision_submit").click().run()
    assert not app.exception
    assert app.session_state["thread_id"] == thread and app.session_state["phase"] == "approval"
    assert app.session_state["interrupt_payload"]["itinerary"]["city"] == "Paris"
    assert any("Draft v2" in item.value for item in app.subheader)
    assert next(call for call in fake.calls if call.get("decision") == "revise")["thread_id"] == thread


def test_recovery_renders_only_backend_options_and_no_approve(ui):
    app, fake = ui
    fake.recovery_options = ["new_trip", "cancel"]
    app.sidebar.selectbox(key="demo_mode").select("tavily").run()
    thread = start(app)
    assert app.session_state["phase"] == "recovery"
    assert {button.key for button in app.button if button.key.startswith("recovery_")} == {
        "recovery_new_trip", "recovery_cancel",
    }
    assert not any(button.label == "Approve & export" for button in app.button)
    assert any("More evidenced places" in item.value for item in app.error)
    app.button(key="recovery_new_trip").click().run()
    assert app.session_state["phase"] == "input" and app.session_state["thread_id"] != thread
    assert next(call for call in fake.calls if call.get("decision") == "new_trip")["thread_id"] == thread


def test_recovery_retry_passes_current_demo_mode_per_resume(ui):
    app, fake = ui
    app.sidebar.selectbox(key="demo_mode").select("tavily").run()
    thread = start(app)
    app.sidebar.selectbox(key="demo_mode").select(None).run()
    app.button(key="recovery_retry").click().run()
    action = next(call for call in fake.calls if call.get("decision") == "retry")
    assert action["thread_id"] == thread and action["failure_mode"] is None
    assert app.session_state["phase"] == "approval"


def test_recovery_revision_uses_text_and_resumes_current_thread(ui):
    app, fake = ui
    app.sidebar.selectbox(key="demo_mode").select("tavily").run()
    thread = start(app)
    app.button(key="recovery_revise").click().run()
    app.text_area(key="revision_text").set_value("change Lisbon to Paris")
    app.button(key="recovery_revision_submit").click().run()
    action = next(call for call in fake.calls if call.get("decision") == "revise")
    assert action["feedback"] == "change Lisbon to Paris" and action["thread_id"] == thread


def test_export_failure_retry_is_a_resume_not_a_new_trip(ui):
    app, fake = ui
    thread = start(app)
    fake.export_failure = True
    app.button(key=f"approve_{thread}_1").click().run()
    assert app.session_state["phase"] == "export_failed"
    assert any(button.label == "Retry export" for button in app.button)
    assert not any(button.label == "Approve & export" for button in app.button)
    app.button(key="retry_export").click().run()
    action = next(call for call in fake.calls if call.get("decision") == "retry_export")
    assert action["thread_id"] == thread
    assert app.session_state["phase"] == "done" and badge_present(app)
    assert len([call for call in fake.calls if call.get("user_input")]) == 1


def test_cancel_shows_cancelled_and_never_downloads(ui):
    app, fake = ui
    thread = start(app)
    app.button(key="approval_cancel").click().run()
    assert app.session_state["phase"] == "done"
    assert any(item.value == "Trip cancelled" for item in app.success)
    assert not app.get("download_button")
    assert next(call for call in fake.calls if call.get("decision") == "cancel")["thread_id"] == thread


def test_open_saved_pause_after_new_browser_session_does_not_resume(ui):
    _app, fake = ui
    saved = "saved-thread"
    fake.store[saved] = fake.snapshot(saved, city="Paris", kind="approval", version=3)
    # A new AppTest has no browser state, just the same persistent public trip API.
    second = AppTest.from_file(APP, default_timeout=10).run()
    assert not second.exception
    select = second.sidebar.selectbox(key="resume_choice")
    assert any("Paris" in option and "approval" in option and "\u00b7" in option for option in select.options)
    select.select_index(0).run()
    second.sidebar.button(key="open_trip").click().run()
    assert second.session_state["thread_id"] == saved
    assert second.session_state["phase"] == "approval"
    assert any("Draft v3" in item.value for item in second.subheader)
    assert not fake.calls  # Open is a read, never a start/resume/write
    assert fake.created == 1


def test_all_buttons_are_disabled_during_running_and_refresh_never_replays(ui):
    app, fake = ui
    fake.stop_stream = True
    start(app)
    assert app.session_state["phase"] == "running"
    assert app.button and all(button.disabled for button in app.button)
    assert all(select.disabled for select in app.sidebar.selectbox)
    before = len(fake.calls)
    fake.stop_stream = False
    app.run()  # interrupted render reloads the persisted pause without replay
    assert app.session_state["phase"] == "approval" and len(fake.calls) == before


def test_empty_request_and_revision_are_friendly_and_make_no_calls(ui):
    app, fake = ui
    app.button(key="plan_trip").click().run()
    assert app.session_state["phase"] == "input" and not fake.calls
    assert any("Please enter a city" in item.value for item in app.error)
    start(app)
    before = len(fake.calls)
    app.button(key="approval_revise").click().run()
    app.button(key="approval_revision_submit").click().run()
    assert any("Please describe the changes" in item.value for item in app.error)
    assert len(fake.calls) == before


def test_unexpected_errors_are_friendly_without_raw_exception_or_stuck_running(ui, caplog):
    app, fake = ui
    fake.raise_error = True
    start(app)
    assert app.session_state["phase"] == "input" and not app.exception
    assert any("couldn't finish" in item.value for item in app.error)
    assert "private-secret" not in " ".join(item.value for item in app.error)
    assert "private-secret" not in caplog.text


def test_app_imports_only_public_graph_api():
    tree = ast.parse(APP.read_text("utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src"):
            assert node.module == "src.graph"
            assert all(not alias.name.startswith("_") for alias in node.names)
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"get_state", "invoke", "conn", "_config"}


def test_stale_approval_sends_displayed_version_and_shows_new_pause(ui):
    app, fake = ui
    thread = start(app)
    fake.store[thread] = fake.snapshot(thread, version=2)
    app.button(key=f"approve_{thread}_1").click().run()
    assert not app.exception and app.session_state["phase"] == "approval"
    action = next(call for call in fake.calls if call.get("decision") == "approve")
    assert action["draft_version"] == 1 and action["thread_id"] == thread
    assert any("Draft v2" in item.value for item in app.subheader)
    assert any("Stale approval" in item.value for item in app.warning)
    assert not badge_present(app) and not app.get("download_button")


@pytest.mark.parametrize("kind", ["recovery", "export_failed"])
def test_open_restores_phase_directly_from_pause_kind(ui, kind):
    app, fake = ui
    fake.store["restored"] = fake.snapshot("restored", kind=kind)
    app.run()
    app.sidebar.selectbox(key="resume_choice").select_index(0).run()
    app.sidebar.button(key="open_trip").click().run()
    assert not app.exception
    assert app.session_state["phase"] == app.session_state["interrupt_payload"]["kind"] == kind
    assert app.session_state["thread_id"] == "restored"
    assert not any(button.label == "Approve & export" for button in app.button)
    assert not fake.calls


def test_city_photos_display_once_and_change_with_revised_city(ui):
    app, fake = ui
    fake.photos_by_city = {
        city: [{"url": f"https://images.test/{city}/{i}.jpg", "caption": f"{city} skyline {i}",
                "source_url": f"https://tourism.test/{city}/{i}"} for i in range(2)]
        for city in ("Lisbon", "Paris")
    }
    thread = start(app)
    assert len(app.get("image")) == 2
    assert fake.image_calls == ["Lisbon"]
    assert sum(button.proto.label == "Image source" for button in app.get("link_button")) == 2
    app.run()
    assert fake.image_calls == ["Lisbon"]
    app.button(key="approval_revise").click().run()
    app.text_area(key="revision_text").set_value("change Lisbon to Paris")
    app.button(key="approval_revision_submit").click().run()
    assert fake.image_calls == ["Lisbon", "Paris"]
    assert set(app.session_state["city_photos"]) == {"paris"}
    assert len(app.get("image")) == 2
    assert any("Photos of Paris" in item.value for item in app.caption)
    assert "city_photos" not in fake.store[thread]
    app.button(key=f"approve_{thread}_2").click().run()
    assert app.session_state["phase"] == "done"
    assert badge_present(app)
    app.sidebar.button(key="new_trip").click().run()
    assert app.session_state["city_photos"] == {}
    assert not app.get("image")
    start(app)
    assert fake.image_calls == ["Lisbon", "Paris", "Lisbon"]
    assert len(app.get("image")) == 2


def test_missing_photos_dont_block_review_and_cache_empty_result(ui):
    app, fake = ui
    start(app)
    app.run()
    assert app.session_state["phase"] == "approval"
    assert not app.get("image")
    assert fake.image_calls == ["Lisbon"]
    assert not app.exception


def test_weather_visible_during_review_and_after_export(ui):
    app, fake = ui
    thread = start(app)
    note = "Forecast for 2026-11-15: Precipitation probability: 2%; High: 29 C."
    fake.store[thread]["itinerary"]["days"][0]["weather_note"] = note
    app.sidebar.selectbox(key="resume_choice").select(thread).run()
    app.sidebar.button(key="open_trip").click().run()
    assert any("Weather: " + note in item.value for item in app.info)
    app.button(key=f"approve_{thread}_1").click().run()
    assert app.session_state["phase"] == "done"
    assert any(item.value == "Weather for your trip" for item in app.subheader)
    assert any(note in item.value for item in app.info)
