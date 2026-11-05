"""Build evidence only from tool messages and validate exact source quotes."""

import json
import unicodedata
from collections.abc import Iterable, Mapping
from html import unescape
from typing import Any

from langchain_core.messages import ToolMessage

from src.state import SourceRecord

GENERIC_WORDS = {"restaurant", "cafe", "bar", "museum", "the", "of"}


def normalize_name(name: str) -> str:
    """Normalize accents, punctuation, case, and generic place words."""
    decomposed = unicodedata.normalize("NFKD", name.casefold())
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    words = "".join(char if char.isalnum() else " " for char in plain).split()
    return " ".join(word for word in words if word not in GENERIC_WORDS)


def _payload(message: ToolMessage) -> dict[str, Any] | None:
    """Accept successful read-tool artifacts, or serialized read-tool responses."""
    if message.name not in {"web_search", "read_page"} or message.status != "success":
        return None
    value = message.artifact
    if value is None and isinstance(message.content, str):
        content = message.content
        if content.startswith("<tool_data>") and content.endswith("</tool_data>"):
            content = unescape(content[len("<tool_data>"):-len("</tool_data>")])
        try:
            value = json.loads(content)
        except (ValueError, TypeError):
            return None
    if isinstance(value, dict) and value.get("ok") is True:
        return value
    return None


def registry_entries(tool_messages: Iterable[Any]) -> dict[str, dict[str, str]]:
    """Extract evidence from successful real ToolMessages, never AI messages."""
    registry: dict[str, dict[str, str]] = {}
    for message in tool_messages:
        if not isinstance(message, ToolMessage):
            continue
        payload = _payload(message)
        if payload is None:
            continue
        data = payload.get("data")
        entries = data if isinstance(data, list) else [data]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            text = entry.get("text", entry.get("content"))
            fields = [entry.get(key) for key in ("source_id", "url", "retrieved_at")]
            if not all(isinstance(value, str) and value for value in fields):
                continue
            if not isinstance(text, str) or not text.strip():
                continue
            source_id, url, retrieved_at = fields
            registry[source_id] = {
                "url": url, "retrieved_at": retrieved_at, "text": text[:4000],
            }
    return registry


def _field(place: Any, name: str) -> Any:
    """Read mapping or Pydantic place fields without changing the original."""
    return place.get(name) if isinstance(place, Mapping) else getattr(place, name, None)


def validate_places(
    places: Iterable[Any], registry: Mapping[str, SourceRecord | Mapping[str, str]]
) -> tuple[list[Any], list[str]]:
    """Keep only places with matching URLs and verbatim, name-bearing quotes."""
    kept: list[Any] = []
    dropped: list[str] = []
    for index, place in enumerate(places):
        source_id = _field(place, "source_id")
        source = registry.get(source_id) if isinstance(source_id, str) else None
        quote = _field(place, "evidence_quote")
        name = _field(place, "name")
        reason = None
        if source is None:
            reason = "unknown_source_id"
        elif _field(place, "source_url") != _field(source, "url"):
            reason = "source_url_mismatch"
        elif (
            not isinstance(name, str) or not normalize_name(name)
            or not isinstance(quote, str)
            or f" {normalize_name(name)} " not in f" {normalize_name(quote)} "
        ):
            reason = "quote_missing_name"
        elif " ".join(quote.split()) not in " ".join((_field(source, "text") or "").split()):
            reason = "quote_not_in_source"
        if reason:
            dropped.append(f"place[{index}]: {reason}")
        else:
            kept.append(place)
    return kept, dropped
