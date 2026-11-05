"""Lazy service configuration without credential validation at import time."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tavily import TavilyClient


@dataclass(frozen=True)
class Settings:
    """Optional settings loaded from the environment and .env."""

    nebius_api_key: str | None = field(default=None, repr=False)
    nebius_base_url: str | None = None
    nebius_model: str | None = None
    tavily_api_key: str | None = field(default=None, repr=False)
    langsmith_api_key: str | None = field(default=None, repr=False)
    langsmith_tracing: bool = False
    langsmith_project: str = "travel-planner"

    @classmethod
    def load(cls) -> "Settings":
        """Read .env without overriding existing environment values."""
        load_dotenv()
        return cls(
            nebius_api_key=os.getenv("NEBIUS_API_KEY"),
            nebius_base_url=os.getenv("NEBIUS_BASE_URL"),
            nebius_model=os.getenv("NEBIUS_MODEL"),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            langsmith_api_key=os.getenv("LANGSMITH_API_KEY"),
            langsmith_tracing=os.getenv("LANGSMITH_TRACING", "").lower() == "true",
            langsmith_project=os.getenv("LANGSMITH_PROJECT", "travel-planner"),
        )


def _required(value: str | None, name: str) -> str:
    """Validate a setting without disclosing its value."""
    if not value or not value.strip():
        raise ValueError(f"Missing {name}; set it in .env before using this service.")
    return value


def get_llm(temperature: float = 0.3, model: str | None = None) -> ChatOpenAI:
    """Return the native Nebius client, preserving its tool interfaces."""
    settings = Settings.load()
    return ChatOpenAI(
        api_key=_required(settings.nebius_api_key, "NEBIUS_API_KEY"),
        base_url=_required(settings.nebius_base_url, "NEBIUS_BASE_URL"),
        model=_required(model or settings.nebius_model, "NEBIUS_MODEL"),
        temperature=temperature,
        max_retries=2,
        timeout=60,
    )


def get_tavily_client() -> TavilyClient:
    """Validate Tavily credentials only when requested."""
    return TavilyClient(
        api_key=_required(Settings.load().tavily_api_key, "TAVILY_API_KEY")
    )
