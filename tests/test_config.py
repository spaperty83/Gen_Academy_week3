"""Mocked tests; no credentials or network."""

from unittest.mock import patch

import pytest

from src.config import Settings, get_llm, get_tavily_client


@pytest.fixture(autouse=True)
def isolated_environment():
    """Ignore real credentials and dotenv files."""
    with patch.dict("os.environ", {}, clear=True), patch("src.config.load_dotenv"):
        yield


def test_settings_load_without_keys():
    settings = Settings.load()
    assert settings.nebius_api_key is None
    assert settings.tavily_api_key is None


def test_llm_requires_key():
    with pytest.raises(ValueError, match="NEBIUS_API_KEY"):
        get_llm()


def test_native_llm_options():
    with (
        patch.dict("os.environ", {
            "NEBIUS_API_KEY": "fake",
            "NEBIUS_BASE_URL": "https://example.invalid/v1",
            "NEBIUS_MODEL": "configured",
        }),
        patch("src.config.ChatOpenAI") as client,
    ):
        assert get_llm(model="override") is client.return_value
        client.assert_called_once_with(
            api_key="fake", base_url="https://example.invalid/v1",
            model="override", temperature=0.3, max_retries=2, timeout=60,
        )


def test_tavily_requires_key():
    with pytest.raises(ValueError, match="TAVILY_API_KEY"):
        get_tavily_client()


def test_tavily_client():
    with (
        patch.dict("os.environ", {"TAVILY_API_KEY": "fake"}),
        patch("src.config.TavilyClient") as client,
    ):
        assert get_tavily_client() is client.return_value
        client.assert_called_once_with(api_key="fake")
