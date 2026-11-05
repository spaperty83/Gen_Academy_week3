"""Live tests are opt-in."""

import os

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip external service tests unless explicitly enabled."""
    if os.getenv("RUN_LIVE") == "1":
        return
    for item in items:
        if item.get_closest_marker("live"):
            item.add_marker(pytest.mark.skip(reason="Set RUN_LIVE=1 for live tests"))
