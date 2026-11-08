"""Pure research coverage requirements and bounded tool-call budgets."""

from collections.abc import Iterable, Mapping
from math import ceil
from typing import Any

from src.evidence import normalize_name
from src.state import Pace, Place, TripRequest


def stops_per_day(pace: Pace) -> int:
    """Return the exact daily stop count for the requested pace."""
    return {"relaxed": 3, "normal": 4, "packed": 5}[pace]


def food_per_day(interests: Iterable[str]) -> int:
    """Require two food stops for a food interest, otherwise one."""
    return 2 if any(interest.strip().casefold() == "food" for interest in interests) else 1


def _request(request: TripRequest | Mapping[str, Any]) -> TripRequest:
    """Accept validated models or their checkpoint/JSON dictionaries."""
    return TripRequest.model_validate(request)


def needed(request: TripRequest | Mapping[str, Any]) -> dict[str, int]:
    """Calculate the minimum sight and food candidates for the full trip."""
    trip = _request(request)
    food = food_per_day(trip.interests)
    return {
        "sights": (stops_per_day(trip.pace) - food) * trip.days,
        "food": food * trip.days,
    }


def targets(request: TripRequest | Mapping[str, Any]) -> dict[str, int]:
    """Provide a 30-percent candidate margin, rounded up."""
    # Integer arithmetic computes ceil(n * 1.3) without floating-point drift.
    return {category: (count * 13 + 9) // 10 for category, count in needed(request).items()}


def tool_budget(target: int) -> int:
    """Bound a researcher's tool calls between eight and fourteen."""
    return min(14, max(8, ceil(target / 3)))


def has_enough(
    places: Iterable[Place | Mapping[str, Any]],
    request: TripRequest | Mapping[str, Any],
) -> bool:
    """Require distinct food and non-food candidates; activities count as sights.

    Sight, activity and other are the non-food pool. Categories, rather than the
    researcher identity, determine coverage. Normalized duplicate names count once.
    """
    minimum = needed(request)
    candidates: dict[str, set[str]] = {"sights": set(), "food": set()}
    seen: set[str] = set()
    for place in places:
        category = place.category if isinstance(place, Place) else place.get("category")
        name = place.name if isinstance(place, Place) else place.get("name", "")
        if not isinstance(name, str) or category not in {"sight", "activity", "food", "other"}:
            continue
        normalized = normalize_name(name)
        if normalized and normalized not in seen:
            candidates["food" if category == "food" else "sights"].add(normalized)
            seen.add(normalized)
    return all(len(candidates[category]) >= count for category, count in minimum.items())
