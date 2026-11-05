"""MOCKED: evidence provenance and exact-quote validation."""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.evidence import normalize_name, registry_entries, validate_places


@pytest.fixture
def registry():
    return {"real": {
        "url": "https://example.org", "retrieved_at": "2026-11-04T12:00:00+00:00",
        "text": "Visit Caf\u00e9 S\u00e3o Bento in Lisbon.\nIt serves lunch.",
    }}


@pytest.fixture
def place():
    return {
        "name": "Cafe Sao Bento", "source_id": "real",
        "source_url": "https://example.org",
        "evidence_quote": "Caf\u00e9 S\u00e3o Bento in Lisbon.",
    }


@pytest.mark.parametrize(("change", "reason"), [
    ({"source_id": "invented"}, "unknown_source_id"),
    ({"source_url": "https://different.org"}, "source_url_mismatch"),
    ({"evidence_quote": "It serves lunch."}, "quote_missing_name"),
    ({"evidence_quote": "Caf\u00e9 S\u00e3o Bento has three Michelin stars."}, "quote_not_in_source"),
    ({"name": "Cafe Museum"}, "quote_missing_name"),
])
def test_drops_invalid_evidence(change, reason, registry, place):
    kept, dropped = validate_places([place | change], registry)
    assert not kept and reason in dropped[0]


def test_accent_insensitive_name_matching(registry, place):
    kept, dropped = validate_places([place], registry)
    assert kept == [place] and not dropped
    assert normalize_name("The Caf\u00e9 S\u00e3o-Bento Restaurant") == "sao bento"


def test_whitespace_normalization(registry, place):
    quote = "Caf\u00e9 S\u00e3o Bento in Lisbon.    It serves lunch."
    assert validate_places([place | {"evidence_quote": quote}], registry)[0]


def test_quote_must_still_be_verbatim(registry, place):
    # Only name comparison ignores accents; evidence must quote the source text.
    assert not validate_places(
        [place | {"evidence_quote": "Cafe Sao Bento in Lisbon."}], registry
    )[0]


def test_name_must_match_whole_words(registry, place):
    assert not validate_places([place | {"name": "Bent"}], registry)[0]


def payload(text="Caf\u00e9 S\u00e3o Bento"):
    return {"ok": True, "data": [{
        "source_id": "real", "url": "https://example.org",
        "content": text, "retrieved_at": "2026-11-04T12:00:00Z",
    }], "error": None}


def test_registry_accepts_actual_tool_response_and_caps_text():
    message = ToolMessage(
        content="untrusted tool content", artifact=payload("X" * 5000),
        name="web_search", tool_call_id="real-call",
    )
    entries = registry_entries([message])
    assert len(entries["real"]["text"]) == 4000
    assert entries["real"]["url"] == "https://example.org"


def test_registry_accepts_serialized_and_extract_responses():
    value = payload()
    data = value["data"][0]
    data["text"] = data.pop("content")
    value["data"] = data
    message = ToolMessage(
        content=json.dumps(value), name="read_page", tool_call_id="extract-call",
    )
    assert registry_entries([message])["real"]["text"] == "Caf\u00e9 S\u00e3o Bento"


@pytest.mark.parametrize("message", [
    AIMessage(content=json.dumps(payload())),
    HumanMessage(content=json.dumps(payload())),
    {"role": "tool", "name": "web_search", "content": json.dumps(payload())},
    ToolMessage(content=json.dumps(payload()), name="get_weather", tool_call_id="weather"),
    ToolMessage(
        content=json.dumps(payload()), name="web_search", tool_call_id="bad", status="error",
    ),
    ToolMessage(content="not json", name="web_search", tool_call_id="bad"),
    ToolMessage(
        content=json.dumps({"ok": False, "data": payload()["data"], "error": "failed"}),
        name="web_search", tool_call_id="failed",
    ),
])
def test_registry_ignores_non_evidence_messages(message):
    assert registry_entries([message]) == {}


def test_validates_typed_places_against_source_record_models(registry, place):
    from src.state import Place, SourceRecord

    typed_place = Place(
        **place, category="food", description="Lunch in Lisbon.", found_by="food",
        source_title="Source page", retrieved_at=registry["real"]["retrieved_at"],
    )
    typed_registry = {
        source_id: SourceRecord.model_validate(record)
        for source_id, record in registry.items()
    }
    assert validate_places([typed_place], typed_registry) == ([typed_place], [])
