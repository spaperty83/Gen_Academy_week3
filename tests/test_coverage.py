"""MOCKED: exact coverage formulas and category-specific prerequisites."""

import pytest

from src.coverage import (
    food_per_day,
    has_enough,
    needed,
    stops_per_day,
    targets,
    tool_budget,
)
from src.state import Place, TripRequest


def request(days=7, pace="packed", interests=None):
    return TripRequest(
        city="Lisbon", days=days, pace=pace, budget="budget",
        interests=["food"] if interests is None else interests,
    )


def places(sights, food):
    return (
        [{"name": f"Sight {index}", "category": "sight"} for index in range(sights)]
        + [{"name": f"Food {index}", "category": "food"} for index in range(food)]
    )


def test_seven_days_packed_with_food():
    trip = request()
    assert needed(trip) == {"sights": 21, "food": 14}
    assert targets(trip) == {"sights": 28, "food": 19}
    assert {category: tool_budget(count) for category, count in targets(trip).items()} == {
        "sights": 10, "food": 8,
    }


def test_one_day_relaxed_small_targets():
    trip = request(days=1, pace="relaxed", interests=["history"])
    assert needed(trip) == {"sights": 2, "food": 1}
    assert targets(trip) == {"sights": 3, "food": 2}
    assert all(tool_budget(count) == 8 for count in targets(trip).values())


@pytest.mark.parametrize(("pace", "count"), [("relaxed", 3), ("normal", 4), ("packed", 5)])
def test_stops_per_day(pace, count):
    assert stops_per_day(pace) == count


def test_food_interest_matching():
    assert food_per_day([]) == 1
    assert food_per_day(["history"]) == 1
    assert food_per_day(["food"]) == 2
    assert food_per_day([" Food ", "history", "food"]) == 2


@pytest.mark.parametrize(("target", "budget"), [(0, 8), (1, 8), (25, 9), (28, 10), (100, 14)])
def test_tool_budget_bounds_and_rounding(target, budget):
    assert tool_budget(target) == budget


@pytest.mark.parametrize(("sights", "food", "enough"), [
    (21, 14, True), (20, 14, False), (21, 13, False), (100, 0, False), (0, 100, False),
])
def test_coverage_requires_each_category(sights, food, enough):
    assert has_enough(places(sights, food), request()) is enough


def test_checkpoint_dictionary_request_is_supported():
    trip = request(days=1, pace="relaxed", interests=[])
    assert needed(trip.model_dump()) == needed(trip)


def test_duplicate_normalized_names_do_not_inflate_coverage():
    trip = request(days=1, pace="relaxed", interests=[])
    candidates = [
        {"name": "Museum Alpha", "category": "sight"},
        {"name": "The Alpha Museum", "category": "sight"},
        {"name": "Lunch", "category": "food"},
    ]
    assert not has_enough(candidates, trip)


def test_model_places_and_non_food_categories_are_supported():
    trip = request(days=1, pace="relaxed", interests=[])
    candidates = [
        Place(
            name=name, category=category, description="", found_by=found_by,
            source_id=name, source_url="https://example.org", source_title="",
            retrieved_at="2026-11-04T12:00:00Z", evidence_quote=f"Visit {name}.",
        )
        for name, category, found_by in [
            ("Walking tour", "activity", "sights"),
            ("Riverfront", "other", "sights"),
            ("Lunch", "food", "food"),
        ]
    ]
    assert has_enough(candidates, trip)


def test_one_place_cannot_satisfy_both_category_pools():
    trip = request(days=1, pace="relaxed", interests=[])
    assert not has_enough([
        {"name": "Alpha", "category": "sight"},
        {"name": "Beta", "category": "sight"},
        {"name": "Alpha", "category": "food"},
    ], trip)
