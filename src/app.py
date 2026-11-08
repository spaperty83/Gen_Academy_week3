"""Streamlit UI using only the public workflow API; credentials stay in .env."""

import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import streamlit as st

from src.graph import (
    build_graph,
    get_checkpointer,
    get_city_images,
    get_export,
    get_trip,
    list_trips,
    stream_trip,
)

LOGGER = logging.getLogger(__name__)
CHECKPOINT_PATH = str(Path(__file__).resolve().parents[1] / "checkpoints.sqlite")
PAUSES = {"approval", "recovery", "export_failed"}
LABELS = {
    "sights": "Sights researcher", "sights_researcher": "Sights Researcher",
    "food": "Food researcher", "food_researcher": "Food Researcher",
    "both": "Sights + Food researchers", "planner": "Planner", "critic": "Critic",
    "traveler": "Traveler", "recovery": "Recovery", "stop": "End",
}


@st.cache_resource(show_spinner="Opening your trip store\u2026")
def resources(path: str = CHECKPOINT_PATH) -> tuple[Any, Any]:
    """Share one thread-safe saver/graph; trip data remains isolated by thread_id."""
    checkpointer = get_checkpointer(path)
    return checkpointer, build_graph(checkpointer)


def _init_session() -> None:
    """Initialize independent per-browser UI state, never credentials."""
    defaults = {
        "thread_id": str(uuid4()), "phase": "input", "interrupt_payload": None,
        "warnings": [], "log_lines": [], "snapshot": {}, "pending_action": None,
        "show_revision": False, "ui_error": None, "resume_trips": [], "city_photos": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _new_trip() -> None:
    """Create a new thread without deleting or resuming any existing paused trip."""
    if st.session_state.phase == "running":
        return
    st.session_state.update({
        "thread_id": str(uuid4()), "phase": "input", "interrupt_payload": None,
        "warnings": [], "log_lines": [], "snapshot": {}, "pending_action": None,
        "show_revision": False, "ui_error": None, "request_text": "", "revision_text": "",
        "city_photos": {},
    })


def _queue_start() -> None:
    """Queue one start action; rerun renders disabled controls before streaming."""
    if st.session_state.phase != "input":
        return
    request = st.session_state.get("request_text", "").strip()
    if not request:
        st.session_state.ui_error = "Please enter a city and a short trip request."
        return
    st.session_state.city_photos = {}
    st.session_state.pending_action = {
        "thread_id": st.session_state.thread_id, "user_input": request,
        "failure_mode": st.session_state.get("demo_mode"),
    }
    st.session_state.phase = "running"
    st.session_state.log_lines = []
    st.session_state.ui_error = None


def _options(payload: dict[str, Any]) -> list[str]:
    """Recovery options always come from the backend, never from UI guesses."""
    defaults = {
        "approval": ["approve", "revise", "cancel"],
        "export_failed": ["retry_export", "cancel"],
    }
    return payload.get("options", defaults.get(payload.get("kind"), []))


def _queue_resume(
    thread_id: str, decision: str, draft_version: int | None = None,
    with_feedback: bool = False,
) -> None:
    """Queue only an allowed action for the displayed current thread and version."""
    if st.session_state.phase == "running":
        return
    if thread_id != st.session_state.thread_id:
        st.session_state.ui_error = "The selected trip changed. Please review its current screen."
        return
    payload = st.session_state.interrupt_payload or {}
    if decision not in _options(payload):
        st.session_state.ui_error = "That action is not available for this trip."
        return
    feedback = st.session_state.get("revision_text", "").strip() if with_feedback else None
    if with_feedback and not feedback:
        st.session_state.ui_error = "Please describe the changes you want."
        return
    if decision == "revise":
        st.session_state.city_photos = {}
    st.session_state.pending_action = {
        "thread_id": thread_id, "decision": decision, "draft_version": draft_version,
        "feedback": feedback, "failure_mode": st.session_state.get("demo_mode"),
    }
    st.session_state.phase = "running"
    st.session_state.ui_error = None


def _request_changes() -> None:
    """Reveal revision entry without invoking or restarting the graph."""
    if st.session_state.phase != "running":
        st.session_state.show_revision = True


def _apply_snapshot(result: dict[str, Any]) -> None:
    """The current interrupt's kind is the sole source of paused UI phase."""
    view = result["screen"]
    kind = view["kind"]
    if kind == "new_trip":
        st.session_state.phase = "done"
        _new_trip()
        return
    if kind == "clarification":
        request = st.session_state.get("request_text", "")
        st.session_state.phase = "done"
        _new_trip()
        st.session_state.request_text = request
        st.session_state.warnings = view.get("warnings", [])
        return
    old = st.session_state.snapshot
    if old.get("thread_id") != result["thread_id"] or old.get("draft_version") != result.get("draft_version"):
        st.session_state.revision_text = ""
        st.session_state.city_photos = {}
    st.session_state.snapshot = result
    st.session_state.thread_id = result["thread_id"]
    st.session_state.warnings = view.get("warnings", [])
    st.session_state.interrupt_payload = view if kind in PAUSES else None
    st.session_state.phase = kind if kind in PAUSES else "done"
    st.session_state.show_revision = False


def _open_trip(graph: Any, thread_id: str | None) -> None:
    """Load a persisted pause read-only; opening never submits a resume decision."""
    if st.session_state.phase == "running" or not thread_id:
        return
    try:
        result = get_trip(graph, thread_id)
        if result["screen"]["kind"] not in PAUSES:
            st.session_state.ui_error = "That trip is no longer paused. Please select another trip."
            return
        _apply_snapshot(result)
        st.session_state.log_lines = [
            f"\U0001f9ed Step {entry['step']}: Orchestrator \u2192 "
            f"{LABELS.get(entry['next'], entry['next'])}: {entry['reason']}"
            for entry in result.get("delegation_log", [])
        ]
        st.session_state.ui_error = None
    except Exception as error:  # noqa: BLE001 - no provider/database details in the UI
        LOGGER.warning("Open trip failed (%s)", type(error).__name__)
        st.session_state.ui_error = "We couldn't open that saved trip. Refresh and try again."


def progress_line(chunk: dict[str, Any], context: dict[str, Any]) -> str:
    """Format one honest line per public stream chunk, without dumping source data."""
    if "__interrupt__" in chunk:
        return "\u23f8\ufe0f Waiting for " + chunk["__interrupt__"][0].value["kind"].replace("_", " ")
    if "progress" in chunk:
        event = chunk["progress"]
        agent = LABELS.get(event.get("agent"), event.get("agent", "Specialist"))
        detail = " ".join(str(event.get("detail", "")).split())[:180]
        return f"\U0001f50e {agent}: {event.get('tool', 'tool')} '{detail}'"
    for name, update in chunk.items():
        if name == "thread_id" or not isinstance(update, dict):
            continue
        previous = dict(context)
        context.update(update)
        if update.get("agent_error"):
            agent = update["agent_error"].get("agent", LABELS.get(name, name))
            return f"\u26a0\ufe0f {agent.capitalize()} error \u2192 recovery"
        if name == "orchestrator":
            entry = (update.get("delegation_log") or [{}])[-1]
            reason = entry.get("reason", "Supervising next step")
            if reason.startswith("guardrail: "):
                return "\U0001f6e1\ufe0f Guardrail: " + reason.removeprefix("guardrail: ")
            target = LABELS.get(entry.get("next"), entry.get("next", "Next step"))
            round_id = update.get("research_round", context.get("research_round", 0))
            return f"\U0001f9ed Orchestrator \u2192 {target} (round {round_id}): {reason}"
        if name.endswith("_researcher"):
            focus = name.removesuffix("_researcher")
            result = update.get(f"{focus}_result", {})
            return (
                f"\U0001f50e {LABELS.get(name, name)}: {result.get('status', 'completed')} "
                f"({len(result.get('places', []))} places)"
            )
        if name == "merge_research":
            places = update.get("places", [])
            food = sum(place.get("category") == "food" for place in places)
            sights = len(places) - food
            selected = previous.get("dispatched", [])
            action = "food replaced, sights kept" if selected == ["food"] else (
                "sights replaced, food kept" if selected == ["sights"] else "research combined"
            )
            return f"\U0001f500 Merge: {action} ({sights} sights, {food} food)"
        if name == "planner":
            return f"\U0001f5d3\ufe0f Planner: draft v{update.get('draft_version', '?')}"
        if name == "critic":
            critique = update.get("critique", {})
            return (
                f"\U0001f9ea Critic v{update.get('critique_version', '?')}: "
                f"{len(critique.get('hard_failures', []))} hard, "
                f"{len(critique.get('soft_findings', []))} soft"
            )
        if name == "save":
            return "\U0001f4be Export completed" if update.get("saved_path") else "\u26a0\ufe0f Export needs attention"
        if name == "end_cancelled":
            return "Trip cancelled"
        return name.replace("_", " ").capitalize() + ": updated"
    return "Workflow updated"


def _run_job(graph: Any) -> None:
    """Consume one queued action exactly once, then reload its persisted screen."""
    job = st.session_state.pending_action
    st.session_state.pending_action = None
    context = dict(st.session_state.snapshot)
    progress = st.empty()
    with progress.status("Run log - planning your trip", expanded=True) as status:
        try:
            if job is None:
                # A browser rerun can interrupt a previous streaming script. Never
                # blindly replay a start or an approval after such an interruption.
                result = get_trip(graph, st.session_state.thread_id)
            else:
                for chunk in stream_trip(graph, **job):
                    line = progress_line(chunk, context)
                    st.session_state.log_lines.append(line)
                    st.text(line)
                result = get_trip(graph, job["thread_id"])
            _apply_snapshot(result)
            status.update(label="Trip updated", state="complete", expanded=False)
        except Exception as error:  # noqa: BLE001 - keep raw errors and credentials out of UI
            LOGGER.warning("Trip action failed (%s)", type(error).__name__)
            status.update(label="Trip action stopped", state="error")
            st.session_state.ui_error = (
                "We couldn't finish that action. Saved progress is retained. "
                "Use Resume trip to reopen an available pause."
            )
            try:
                _apply_snapshot(get_trip(graph, st.session_state.thread_id))
            except Exception:  # noqa: BLE001 - the store itself may be temporarily unavailable
                st.session_state.phase = "input"
                st.session_state.thread_id = str(uuid4())
    progress.empty()
    st.rerun()


def _sidebar(graph: Any, checkpointer: Any) -> None:
    """Resume existing pauses or create a fresh thread, with no actions during runs."""
    busy = st.session_state.phase == "running"
    with st.sidebar:
        st.header("Your trips")
        if not busy:
            try:
                st.session_state.resume_trips = list_trips(graph, checkpointer)
            except Exception as error:  # noqa: BLE001 - friendly store errors only
                LOGGER.warning("List trips failed (%s)", type(error).__name__)
                st.warning("Saved trips are temporarily unavailable.")
                st.session_state.resume_trips = []
        trips = {item["thread_id"]: item for item in st.session_state.resume_trips}

        def label(thread: str) -> str:
            # Streamlit/testing can format an already-rendered selection.
            # Keep formatting idempotent and tolerate a removed list entry.
            if thread not in trips:
                return str(thread)
            trip = trips[thread]
            return (
                f"{trip.get('city') or 'City pending'} \u00b7 {trip.get('last_update') or ''}"
                f" \u00b7 {trip.get('paused_at', 'unavailable').replace('_', ' ')}"
            )

        selection = st.selectbox(
            "Resume trip", options=list(trips), format_func=label, index=None,
            placeholder="Choose a paused trip", disabled=busy or not trips, key="resume_choice",
        )
        st.button(
            "Open", key="open_trip", disabled=busy or selection is None,
            on_click=_open_trip, args=(graph, selection),
        )
        st.selectbox(
            "Demo failure mode", options=[None, "extract", "tavily", "weather", "llm_json"],
            format_func=lambda value: "None" if value is None else value,
            disabled=busy, key="demo_mode",
        )
        st.button("New trip", key="new_trip", on_click=_new_trip, disabled=busy)
        st.caption("Paused trips survive app restarts.")



def _workflow_notice(message: str) -> bool:
    """Identify technical run history, preserving traveler-facing uncertainty."""
    return (
        message.startswith(("guardrail:", "Round or correction limit reached", "Trip limit reached",
                            "Research tool budget reached"))
        or "candidates dropped:" in message
    )


def _warn(values: list[str], exclude: set[str] | None = None) -> None:
    """Show each warning once; never enable unsafe HTML for model/source text."""
    for value in dict.fromkeys(values):
        if value not in (exclude or set()):
            st.warning(value)


def _revision_form(thread: str, key: str, busy: bool) -> None:
    """Submit feedback as a resume of this same thread."""
    with st.form(key):
        st.text_area("What would you like to change?", key="revision_text", disabled=busy)
        st.form_submit_button(
            "Submit", key=f"{key}_submit", disabled=busy, on_click=_queue_resume,
            args=(thread, "revise", None, True),
        )



def _city_photos(city: str) -> None:
    """Cache optional display photos per city in this browser session only."""
    cache = st.session_state.city_photos
    key = city.strip().casefold()
    if key not in cache:
        try:
            cache[key] = get_city_images(city)
        except Exception:  # noqa: BLE001 - photos must not block review or export.
            LOGGER.info("City photos unavailable")
            cache[key] = []
    photos = cache[key][:2]
    if not photos:
        return
    st.caption(f"Photos of {city} for inspiration; they may not show your itinerary stops.")
    for column, photo in zip(st.columns(len(photos)), photos, strict=True):
        with column:
            try:
                st.image(photo["url"], width="stretch", alt=photo["caption"])
                st.text(photo["caption"])
                st.link_button("Image source", photo["source_url"])
            except Exception:  # noqa: BLE001 - keep the itinerary usable if rendering fails.
                st.caption("This photo is unavailable.")


def _approval(payload: dict[str, Any], busy: bool) -> None:
    """Display only the backend-approved pause, and send its displayed version."""
    version = payload["draft_version"]
    thread = st.session_state.thread_id
    st.subheader(f"Draft v{version}")
    soft = [finding["issue"] for finding in payload.get("soft_findings", [])]
    _warn([message for message in payload.get("warnings", [])
           if not _workflow_notice(message)], set(soft))
    _warn(soft)
    itinerary = payload["itinerary"]
    _city_photos(itinerary["city"])
    for index, day in enumerate(itinerary["days"]):
        with st.expander(f"Day {day['day_number']} \u00b7 {day['theme']}", expanded=index == 0):
            if day.get("weather_note"):
                st.info("Weather: " + day["weather_note"])
            for stop in day["stops"]:
                st.write(f"{stop['time_slot'].capitalize()} \u00b7 {stop['place_name']}")
                st.text(stop["note"])
    st.markdown("**budget preference considered**")
    for tip in itinerary.get("budget_tips", []):
        st.write("\u2022 " + tip)
    with st.expander("Delegation log", expanded=False):
        for entry in payload.get("delegation_log", []):
            st.text(f"Step {entry['step']}: {entry['next']} \u2014 {entry['reason']}")
    with st.expander("Citations", expanded=False):
        for day in itinerary["days"]:
            for index, stop in enumerate(day["stops"]):
                url = stop["source_url"]
                if urlparse(url).scheme in {"http", "https"}:
                    st.link_button(
                        f"Day {day['day_number']} - {stop['place_name']}", url, disabled=busy,
                        key=f"citation_{day['day_number']}_{index}",
                    )
    st.button(
        "Approve & export", key=f"approve_{thread}_{version}", type="primary", disabled=busy,
        on_click=_queue_resume, args=(thread, "approve", version),
    )
    st.button("Request changes", key="approval_revise", disabled=busy, on_click=_request_changes)
    if st.session_state.show_revision:
        _revision_form(thread, "approval_revision", busy)
    st.button(
        "Cancel", key="approval_cancel", disabled=busy,
        on_click=_queue_resume, args=(thread, "cancel"),
    )


def _recovery(payload: dict[str, Any], busy: bool) -> None:
    """Offer exactly the recovery options returned by the backend; never Approve."""
    st.subheader("This trip needs your input")
    st.info(payload["reason"])
    _warn(payload.get("warnings", []))
    for finding in payload.get("hard_findings", []):
        st.error(finding["issue"])
    thread = st.session_state.thread_id
    for option in _options(payload):
        if option == "revise":
            st.button("Revise", key="recovery_revise", disabled=busy, on_click=_request_changes)
            if st.session_state.show_revision:
                _revision_form(thread, "recovery_revision", busy)
        elif option in {"retry", "new_trip", "cancel"}:
            st.button(
                {"retry": "Retry", "new_trip": "New trip", "cancel": "Cancel"}[option],
                key=f"recovery_{option}", disabled=busy,
                on_click=_queue_resume, args=(thread, option),
            )


def _export_failed(payload: dict[str, Any], busy: bool) -> None:
    """Retry the approved export without restarting planning or changing threads."""
    st.subheader("Export needs another try")
    st.info(payload["reason"])
    _warn(payload.get("warnings", []))
    thread = st.session_state.thread_id
    st.button(
        "Retry export", key="retry_export", disabled=busy, type="primary",
        on_click=_queue_resume, args=(thread, "retry_export"),
    )
    st.button(
        "Cancel", key="export_cancel", disabled=busy,
        on_click=_queue_resume, args=(thread, "cancel"),
    )


def _done(graph: Any, busy: bool) -> None:
    """Read validated bytes through the API; never trust a path or manifest alone."""
    snapshot = st.session_state.snapshot
    if snapshot.get("cancelled"):
        st.success("Trip cancelled")
        _warn(st.session_state.warnings)
        return
    if not snapshot.get("saved_path"):
        st.info("Trip ended. Start a new trip when you're ready.")
        _warn(st.session_state.warnings)
        return
    st.subheader("Your itinerary is exported")
    st.code(snapshot["saved_path"], language=None)
    try:
        artifact = get_export(graph, st.session_state.thread_id)
        if artifact["complete"]:
            st.badge("complete", color="green")
            st.download_button(
                "Download Markdown", data=artifact["markdown"], file_name=artifact["file_name"],
                mime="text/markdown", disabled=busy, on_click="ignore", key="download_markdown",
            )
        else:
            st.warning("The export is incomplete or has changed. No download is available.")
    except Exception as error:  # noqa: BLE001 - do not expose disk/database exceptions
        LOGGER.warning("Read export failed (%s)", type(error).__name__)
        st.error("We couldn't read the export. Refresh and try again.")

    weather_days = [day for day in (snapshot.get("itinerary") or {}).get("days", [])
                    if day.get("weather_note")]
    if weather_days:
        st.subheader("Weather for your trip")
        for day in weather_days:
            st.info(f"Day {day['day_number']}: {day['weather_note']}")
    _warn([message for message in st.session_state.warnings if not _workflow_notice(message)])


def main() -> None:
    """Render the phase obtained from the current saved thread's interrupt."""
    st.set_page_config(page_title="Travel planner", page_icon="\U0001f9ed", layout="wide")
    _init_session()
    st.title("Travel planner")
    st.caption("Research, plan, review \u2014 export only after your approval.")
    try:
        checkpointer, graph = resources()
    except Exception as error:  # noqa: BLE001 - setup errors must also be friendly
        LOGGER.warning("Trip store unavailable (%s)", type(error).__name__)
        st.error("The trip store could not be opened. Check the project folder is writable and restart the app.")
        st.stop()
    _sidebar(graph, checkpointer)
    if st.session_state.ui_error:
        st.error(st.session_state.ui_error)
    busy = st.session_state.phase == "running"
    result_area = st.container()
    log_area = st.container()
    if busy:
        with log_area:
            _run_job(graph)
    with result_area:
        if st.session_state.phase == "input":
            _warn(st.session_state.warnings)
            with st.form("new_request", enter_to_submit=True):
                st.text_input(
                    "Where would you like to go?", key="request_text",
                    placeholder="3 days in Lisbon, budget, food and history", disabled=busy,
                )
                st.caption(
                    "Include a start date (YYYY-MM-DD) to plan your trip with a weather forecast. "
                    "Without a start date, your itinerary will be planned without weather information. "
                    "Forecasts depend on availability and are generally limited to the next 16 days."
                )
                st.form_submit_button(
                    "Plan trip", key="plan_trip", type="primary", disabled=busy, on_click=_queue_start,
                )
        elif st.session_state.phase == "approval":
            _approval(st.session_state.interrupt_payload, busy)
        elif st.session_state.phase == "recovery":
            _recovery(st.session_state.interrupt_payload, busy)
        elif st.session_state.phase == "export_failed":
            _export_failed(st.session_state.interrupt_payload, busy)
        else:
            _done(graph, busy)
    run_lines = list(dict.fromkeys([
        *st.session_state.log_lines,
        *(message for message in st.session_state.warnings if _workflow_notice(message)),
    ]))
    with log_area:
        if run_lines:
            with st.expander("Run log", expanded=False):
                for line in run_lines:
                    st.text(line)


if __name__ == "__main__":
    main()
