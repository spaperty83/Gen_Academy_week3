"""MOCKED optional photo lookup; no live APIs or image downloads."""

from unittest.mock import Mock

import pytest
import requests
from tenacity import wait_none

from src import graph, tools
from src.state import Itinerary


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(requests.sessions.Session, "request", Mock(side_effect=AssertionError("No network")))
    fake = Mock()
    monkeypatch.setattr(tools, "get_tavily_client", lambda: fake)
    monkeypatch.setattr(tools, "wait_exponential", lambda **kwargs: wait_none())
    return fake


def test_city_photos_limit_dedupe_source_and_api_options(client):
    client.search.return_value = {"images": [
        {"url": "https://images.test/one.jpg", "description": "City skyline"},
        {"url": "https://images.test/one.jpg"},
        {"url": "javascript:alert(1)"},
        {"url": "https://user:password@images.test/private.jpg"},
        "https://images.test/two.jpg", "https://images.test/three.jpg",
    ]}
    photos = graph.get_city_images("Lisbon")
    assert len(photos) == 2
    assert photos[0]["caption"] == "City skyline"
    assert photos[0]["source_url"] == photos[0]["url"]
    assert photos[1]["url"] == "https://images.test/two.jpg"
    client.search.assert_called_once_with(
        query="Lisbon city landmarks photography", search_depth="basic", max_results=3,
        include_images=True, include_image_descriptions=True, timeout=15,
    )


def test_result_linked_photo_uses_page_for_attribution(client):
    client.search.return_value = {"results": [{"url": "https://tourism.test/porto",
        "images": [{"url": "https://images.test/porto.jpg", "description": "Porto waterfront"}]}]}
    assert graph.get_city_images("Porto")[0]["source_url"] == "https://tourism.test/porto"


@pytest.mark.parametrize("response", [{}, {"images": []}, {"images": [None, {}, 4]}])
def test_missing_or_malformed_photos_are_optional(client, response):
    client.search.return_value = response
    assert graph.get_city_images("Porto") == []


def test_photo_failure_is_optional_and_does_not_leak_provider_error(client, caplog):
    client.search.side_effect = RuntimeError("private-provider-details")
    assert graph.get_city_images("Paris") == []
    assert "private-provider-details" not in caplog.text


def test_photo_lookup_uses_existing_transient_retry_policy(client):
    response = requests.Response()
    response.status_code = 500
    client.search.side_effect = [requests.HTTPError(response=response), {"images": ["https://images.test/a.jpg"]}]
    assert graph.get_city_images("Lisbon")
    assert client.search.call_count == 2


def test_photos_are_not_agent_tools_or_itinerary_fields():
    assert {tool.name for tool in tools.SEARCH_TOOLS} == {"web_search", "read_page"}
    assert {tool.name for tool in tools.WEATHER_TOOLS} == {"get_weather"}
    assert not {"images", "photos", "city_photos"} & Itinerary.model_fields.keys()
