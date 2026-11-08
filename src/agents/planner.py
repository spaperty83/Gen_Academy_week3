"""Weather-aware, grounded Planner with a non-raising agent boundary."""

import logging
import re
from datetime import date, timedelta
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from src.agents._checks import hard_checks
from src.agents._runtime import OutputInvalid, SpecialistRun, envelope, failure_message
from src.agents.prompts import PLANNER
from src.config import get_llm
from src.coverage import food_per_day, has_enough, stops_per_day
from src.evidence import normalize_name
from src.state import AgentError, Critique, Itinerary, Place, TripRequest, TripState
from src.tools import WEATHER_TOOLS

LOGGER = logging.getLogger(__name__)


def _weather(run: SpecialistRun) -> dict[str, Any] | None:
    """Use only successful weather-tool receipts from this planning invocation."""
    for message in reversed(run.tool_messages):
        data = envelope(message)
        if (
            message.name == "get_weather" and data and data.get("ok") is True
            and isinstance(data.get("data"), dict)
        ):
            return data["data"]
    return None


def _indoor(place: Place) -> bool:
    """Require indoor cues in recorded place data, without inventing a venue."""
    text = f"{place.name} {place.description} {place.evidence_quote}".casefold()
    if re.search(r"\b(outdoor|open.air|beach|hiking|terrace)\b", text):
        return False
    return bool(re.search(
        r"\b(indoor|indoors|museum|gallery|covered|interior|aquarium|palace|"
        r"cathedral|church|restaurant|cafe|caf\u00e9|bar)\b", text,
    ))


def _validate_draft(state: TripState, run: SpecialistRun, draft: BaseModel) -> None:
    """Treat violated grounded task constraints as structured validation errors."""
    itinerary = Itinerary.model_validate(draft)
    findings = hard_checks(state | {"itinerary": itinerary})
    if findings:
        trip = TripRequest.model_validate(state["request"])
        run.validation_feedback = {
            "findings": [{"issue": finding.issue, "day": finding.day} for finding in findings],
            "required_days": trip.days, "required_stops_per_day": stops_per_day(trip.pace),
            "maximum_food_per_day": food_per_day(trip.interests),
        }
        raise OutputInvalid("Draft violates grounded itinerary constraints")
    weather = _weather(run)
    if weather is None:
        return
    daily = weather.get("daily", {})
    probabilities = dict(zip(
        daily.get("time", []), daily.get("precipitation_probability_max", []),
        strict=True,
    ))
    trip = TripRequest.model_validate(state["request"])
    places = {
        normalize_name(place.name): place
        for place in (Place.model_validate(raw) for raw in state["places"])
    }
    for day in itinerary.days:
        stamp = (date.fromisoformat(trip.start_date) + timedelta(days=day.day_number - 1)).isoformat()
        probability = probabilities.get(stamp)
        if isinstance(probability, (int, float)) and probability > 60 and any(
            not _indoor(places[normalize_name(stop.place_name)]) for stop in day.stops
        ):
            run.validation_feedback = {
                "findings": [{"issue": "rainy day requires recorded indoor places", "day": day.day_number}],
            }
            raise OutputInvalid("Rainy days require grounded indoor stops")


def _weather_notes(
    itinerary: Itinerary, weather: dict[str, Any] | None, run: SpecialistRun,
) -> Itinerary:
    """Replace generated weather claims with notes derived from actual tool data."""
    daily = weather.get("daily", {}) if weather else {}
    days = []
    for index, day in enumerate(itinerary.days):
        note = None
        chances = daily.get("precipitation_probability_max", [])
        if index < len(chances):
            parts = [f"Precipitation probability: {chances[index]}%"]
            for field, label in (("temperature_2m_min", "Low"), ("temperature_2m_max", "High")):
                values = daily.get(field, [])
                if index < len(values) and isinstance(values[index], (int, float)):
                    unit = (weather.get("units") or {}).get(field, "")
                    parts.append(f"{label}: {values[index]}{unit}")
            dates = daily.get("time", [])
            stamp = f" for {dates[index]}" if index < len(dates) else ""
            note = f"Forecast{stamp}: " + "; ".join(parts) + "."
        days.append(day.model_copy(update={"weather_note": note}))
    warnings = list(itinerary.warnings)
    if weather is None:
        beyond = any(
            (envelope(message) or {}).get("error") == "beyond_forecast_window"
            for message in run.tool_messages if message.name == "get_weather"
        )
        warnings.append(
            "Weather beyond forecast window; no forecast was invented."
            if beyond else "Weather unavailable; outdoor plans may need adjustment."
        )
    return itinerary.model_copy(update={"days": days, "warnings": warnings})


def planner(state: TripState, config: RunnableConfig) -> dict[str, Any]:
    """Build one grounded version, or return agent_error while keeping prior state."""
    try:
        trip = TripRequest.model_validate(state["request"])
        places = [Place.model_validate(place) for place in state.get("places", [])]
        if not has_enough(places, trip):
            raise OutputInvalid("Planner requires enough places in each category")
        tools = WEATHER_TOOLS if trip.start_date else []
        run = SpecialistRun(
            tools, 4,
            weather_request={
                "city": trip.city, "start_date": trip.start_date, "days": trip.days,
            } if trip.start_date else None,
            simulate_first_parse=(
                config.get("configurable", {}).get("failure_mode") == "llm_json"
            ),
        )
        run.validator = lambda draft: _validate_draft(state, run, draft)
        critique = state.get("critique")
        findings = []
        if critique is not None:
            prior = Critique.model_validate(critique)
            findings = [
                finding for finding in [*prior.hard_failures, *prior.soft_findings]
                if finding.owner == "planner"
            ]
        itinerary = run.execute(
            get_llm(), tools, Itinerary, PLANNER,
            {
                "request": trip, "places": places,
                "stops_per_day": stops_per_day(trip.pace),
                "food_per_day": food_per_day(trip.interests),
                "planner_findings": findings,
                "human_feedback": state.get("human_feedback"),
                "instructions": state.get("instructions", {}).get("planner", ""),
            }, config, "planner",
        )
        weather = _weather(run)
        return {
            "itinerary": _weather_notes(itinerary, weather, run), "weather": weather,
            "draft_version": state.get("draft_version", 0) + 1,
            "approved_version": None, "agent_error": None,
        }
    except Exception as error:  # noqa: BLE001 - specialists never raise past their boundary
        LOGGER.warning("Planner failed (%s)", type(error).__name__)
        return {
            "agent_error": AgentError(agent="planner", message=failure_message(error)),
            "approved_version": None,
        }
