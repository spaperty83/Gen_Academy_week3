"""Deterministic draft checks shared by Planner validation and Critic."""

from typing import Any

from pydantic import ValidationError

from src.coverage import food_per_day, stops_per_day
from src.evidence import normalize_name, validate_places
from src.state import Finding, Itinerary, Place, TripRequest, TripState


def hard_checks(state: TripState) -> list[Finding]:
    """Check schema, membership, persisted evidence, day counts, pace and reuse."""
    findings: list[Finding] = []
    try:
        trip = TripRequest.model_validate(state.get("request"))
        itinerary = Itinerary.model_validate(state.get("itinerary"))
    except ValidationError:
        return [Finding(issue="schema invalid", owner="planner")]
    places = []
    for raw in state.get("places", []):
        try:
            places.append(Place.model_validate(raw))
        except ValidationError:
            findings.append(Finding(issue="place schema invalid", owner="researchers"))
    registry = state.get("source_registry", {})
    for place in places:
        kept, _ = validate_places([place], registry)
        if not kept:
            findings.append(Finding(
                issue=f"place without valid evidence: {place.name}", owner="researchers",
            ))
    lookup = {normalize_name(place.name): place for place in places}
    if itinerary.city.strip().casefold() != trip.city.strip().casefold():
        findings.append(Finding(issue="itinerary city does not match request", owner="planner"))
    if len(itinerary.days) != trip.days:
        findings.append(Finding(issue="wrong number of days", owner="planner"))
    if [day.day_number for day in itinerary.days] != list(range(1, trip.days + 1)):
        findings.append(Finding(issue="day numbering does not match request", owner="planner"))
    seen = set()
    for day in itinerary.days:
        if len(day.stops) != stops_per_day(trip.pace):
            findings.append(Finding(
                issue="day outside its pace limit", owner="planner", day=day.day_number,
            ))
        food = 0
        for stop in day.stops:
            name = normalize_name(stop.place_name)
            place = lookup.get(name) if name else None
            if place is None:
                findings.append(Finding(
                    issue=f"stop not in places: {stop.place_name}",
                    owner="planner", day=day.day_number,
                ))
            else:
                food += place.category == "food"
                if stop.source_url != place.source_url:
                    findings.append(Finding(
                        issue=f"stop source URL does not match place: {stop.place_name}",
                        owner="planner", day=day.day_number,
                    ))
            if name in seen:
                findings.append(Finding(
                    issue=f"duplicate stop: {stop.place_name}", owner="planner", day=day.day_number,
                ))
            seen.add(name)
        if food > food_per_day(trip.interests):
            findings.append(Finding(
                issue="too many food stops in a day", owner="planner", day=day.day_number,
            ))
    return findings


def doubtful_stops(state: TripState) -> list[dict[str, Any]]:
    """Select at most three named stops with thin or city-unclear evidence."""
    itinerary = Itinerary.model_validate(state["itinerary"])
    trip = TripRequest.model_validate(state["request"])
    places = {
        normalize_name(place.name): place
        for place in (Place.model_validate(raw) for raw in state.get("places", []))
    }
    candidates = []
    seen = set()
    for day in itinerary.days:
        for stop in day.stops:
            name = normalize_name(stop.place_name)
            if name in seen:
                continue
            seen.add(name)
            place = places.get(name)
            doubtful = (
                place is None
                or not validate_places([place], state.get("source_registry", {}))[0]
                or trip.city.casefold() not in place.evidence_quote.casefold()
            )
            if doubtful:
                candidates.append({
                    "place_name": stop.place_name, "day": day.day_number,
                    "source_url": stop.source_url,
                })
    return candidates[:3]
