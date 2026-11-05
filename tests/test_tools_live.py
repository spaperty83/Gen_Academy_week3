"""LIVE: one invocation of each read tool; skipped unless RUN_LIVE=1."""

from datetime import UTC, datetime

import pytest

from src.tools import get_weather, read_page, web_search

pytestmark = pytest.mark.live


def response(tool, args):
    """Exercise the full LangChain ToolCall/ToolMessage contract."""
    message = tool.invoke({
        "name": tool.name, "args": args, "type": "tool_call", "id": f"live-{tool.name}",
    })
    assert message.artifact["ok"], message.artifact["error"]
    return message.artifact["data"]


def test_live_web_search():
    assert response(web_search, {"query": "Lisbon museums", "max_results": 1})


def test_live_read_page():
    assert response(read_page, {"url": "https://www.maat.pt/en"})["text"]


def test_live_get_weather():
    data = response(get_weather, {
        "city": "Lisbon", "start_date": datetime.now(UTC).date().isoformat(), "days": 1,
    })
    assert len(data["daily"]["time"]) == 1
