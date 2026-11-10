"""MOCKED: real create_agent loops with fake chat models and fake read tools."""

import json
from typing import Any
from unittest.mock import Mock

import pytest
import requests
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from src.agents import critic as critic_module
from src.agents import planner as planner_module
from src.agents import researchers
from src.agents._checks import doubtful_stops, hard_checks
from src.agents.critic import CriticReview, critic
from src.agents.planner import planner
from src.agents.researchers import ResearchCandidates, make_researcher
from src.coverage import targets, tool_budget
from src.state import (
    Critique,
    DayPlan,
    Finding,
    Itinerary,
    Place,
    SourceRecord,
    Stop,
    TripRequest,
)
from src.tools import _result


class ScriptedModel(FakeMessagesListChatModel):
    """A bindable fake that exercises LangChain's real agent graph and parsing."""

    bound_names: list[list[str]] = Field(default_factory=list)
    bound_options: list[dict[str, Any]] = Field(default_factory=list)
    bound_tools: list[list[dict[str, Any]]] = Field(default_factory=list)
    seen_messages: list[Any] = Field(default_factory=list)
    calls: int = 0
    fail_provider: bool = False

    def bind_tools(self, tools, **kwargs):
        self.bound_options.append(kwargs)
        self.bound_tools.append([convert_to_openai_tool(tool) for tool in tools])
        self.bound_names.append([
            convert_to_openai_tool(tool)["function"]["name"] for tool in tools
        ])
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen_messages.append(list(messages))
        self.calls += 1
        if self.fail_provider:
            raise RuntimeError("private-provider-secret")
        response = self.responses[min(self.calls - 1, len(self.responses) - 1)]
        response = response.model_copy(deep=True, update={"id": None})
        return ChatResult(generations=[ChatGeneration(message=response)])


def call(name, args=None):
    return AIMessage(content="", tool_calls=[{
        "name": name, "args": args or {}, "id": f"call-{name}", "type": "tool_call",
    }])


def structured(schema, value):
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return call(schema.__name__, value)


@pytest.fixture
def state():
    places = []
    registry = {}
    for index, (name, category, found_by) in enumerate([
        ("Museum Alpha", "sight", "sights"),
        ("Gallery Beta", "activity", "sights"),
        ("Restaurant Gamma", "food", "food"),
    ]):
        quote = f"{name} is an indoor venue in Lisbon."
        place = Place(
            name=name, category=category, found_by=found_by, description=quote,
            source_id=f"source-{index}", source_url=f"https://example.org/{index}",
            source_title=name, retrieved_at="2026-11-04T12:00:00Z", evidence_quote=quote,
        )
        places.append(place)
        registry[place.source_id] = SourceRecord(
            url=place.source_url, retrieved_at=place.retrieved_at, text=quote,
        )
    itinerary = Itinerary(
        city="Lisbon", days=[DayPlan(
            day_number=1, theme="Art and food",
            stops=[Stop(
                time_slot=slot, place_name=place.name, note="Visit this indoor venue.",
                source_url=place.source_url,
            ) for slot, place in zip(("morning", "afternoon", "evening"), places, strict=True)],
        )],
        budget_tips=["Use transit.", "Consider free hours."],
    )
    return {
        "request": TripRequest(
            city="Lisbon", days=1, budget="budget", interests=[], pace="relaxed",
        ),
        "places": places, "source_registry": registry, "itinerary": itinerary,
        "draft_version": 5, "approved_version": 5, "critique_version": 5,
        "critique": Critique(), "research_round": 7, "instructions": {},
        "weather": None, "agent_error": None,
    }


@pytest.fixture(autouse=True)
def no_external_services(monkeypatch):
    """Fail accidental networking and isolate tracing for all specialist tests."""
    for key in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "NEBIUS_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        requests.sessions.Session, "request",
        Mock(side_effect=AssertionError("Mocked agents attempted network access")),
    )


@pytest.fixture
def fake_tools(state, monkeypatch):
    """Fake receipts retain the real tool names, artifacts, and config injection."""
    calls = []

    @tool("web_search", response_format="content_and_artifact")
    def search(query: str, config: RunnableConfig, max_results: int = 5):
        """READ: Search the fake evidence source."""
        calls.append(("web_search", config))
        return _result([{
            "source_id": place.source_id, "title": place.source_title,
            "url": place.source_url, "content": place.evidence_quote,
            "retrieved_at": place.retrieved_at,
        } for place in state["places"]])

    @tool("read_page", response_format="content_and_artifact")
    def read(url: str, config: RunnableConfig, max_chars: int = 4000):
        """READ: Extract the fake source."""
        calls.append(("read_page", config))
        place = state["places"][0]
        return _result({
            "source_id": place.source_id, "url": place.source_url,
            "retrieved_at": place.retrieved_at, "text": place.evidence_quote,
        })

    @tool("get_weather", response_format="content_and_artifact")
    def weather(city: str, start_date: str, days: int, config: RunnableConfig):
        """READ: Return a fake dated forecast."""
        calls.append(("get_weather", config))
        return _result({
            "city": city, "daily": {
                "time": [start_date], "precipitation_probability_max": [80],
            },
        })

    search_tools = [search, read]
    monkeypatch.setattr(researchers, "SEARCH_TOOLS", search_tools)
    monkeypatch.setattr(critic_module, "SEARCH_TOOLS", search_tools)
    monkeypatch.setattr(planner_module, "WEATHER_TOOLS", [weather])
    return calls, search_tools, [weather]


def use_model(monkeypatch, module, responses, **kwargs):
    model = ScriptedModel(responses=responses, **kwargs)
    monkeypatch.setattr(module, "get_llm", lambda: model)
    return model


@pytest.mark.parametrize("focus", ["sights", "food"])
def test_researchers_own_tools_config_round_and_evidence(state, fake_tools, monkeypatch, focus):
    desired = [place for place in state["places"] if place.found_by == focus]
    model = use_model(monkeypatch, researchers, [
        call("web_search", {"query": "Lisbon museums"}),
        structured(ResearchCandidates, {"places": [place.model_dump() for place in desired]}),
    ])
    config = {"configurable": {"marker": "forwarded"}, "tags": ["mocked"]}
    result = make_researcher(focus)(state, config)
    assert set(result) == {f"{focus}_result"}
    assert result[f"{focus}_result"].status == "ok"
    assert result[f"{focus}_result"].round_id == 7
    assert all(place.found_by == focus for place in result[f"{focus}_result"].places)
    assert set(model.bound_names[0]) == {"web_search", "read_page", "ResearchCandidates"}
    assert fake_tools[0][0][1]["configurable"]["marker"] == "forwarded"
    assert result[f"{focus}_result"].sources
    tool_messages = [
        message for message in model.seen_messages[-1] if isinstance(message, ToolMessage)
    ]
    assert tool_messages and all(
        message.content.startswith("<tool_data>") for message in tool_messages
    )


def test_planner_has_only_weather_and_increments_version(state, fake_tools, monkeypatch):
    state["request"] = state["request"].model_copy(update={"start_date": "2026-11-04"})
    model = use_model(monkeypatch, planner_module, [
        call("get_weather", {"city": "Lisbon", "start_date": "2026-11-04", "days": 1}),
        structured(Itinerary, state["itinerary"]),
    ])
    result = planner(state, {"configurable": {"marker": "planner-config"}})
    assert result["agent_error"] is None
    assert result["draft_version"] == 6 and result["approved_version"] is None
    assert result["weather"]["daily"]["precipitation_probability_max"] == [80]
    assert "80%" in result["itinerary"].days[0].weather_note
    assert set(model.bound_names[0]) == {"get_weather", "Itinerary"}
    assert fake_tools[0][0][1]["configurable"]["marker"] == "planner-config"


def test_planner_without_date_never_has_weather_tool(state, fake_tools, monkeypatch):
    model = use_model(monkeypatch, planner_module, [structured(Itinerary, state["itinerary"])])
    result = planner(state, {})
    assert result["agent_error"] is None and result["weather"] is None
    assert set(model.bound_names[0]) == {"Itinerary"}
    assert not fake_tools[0]


def test_critic_has_only_search_tools_and_passes_config(state, fake_tools, monkeypatch):
    model = use_model(monkeypatch, critic_module, [
        call("web_search", {"query": "Museum Alpha Lisbon"}),
        structured(CriticReview, {"soft_findings": []}),
    ])
    result = critic(state, {"configurable": {"marker": "critic-config"}})
    assert result["agent_error"] is None
    assert result["critique_version"] == state["draft_version"]
    assert set(model.bound_names[0]) == {"web_search", "read_page", "CriticReview"}
    assert fake_tools[0][0][1]["configurable"]["marker"] == "critic-config"


@pytest.mark.parametrize("focus", ["sights", "food"])
def test_looping_researcher_stops_at_fallback_budget(state, fake_tools, monkeypatch, focus):
    model = use_model(monkeypatch, researchers, [call("web_search", {"query": "Lisbon"})])
    result = make_researcher(focus)(state, {})
    assert result[f"{focus}_result"].status == "failed"
    assert result[f"{focus}_result"].error == "specialist_step_limit_reached"
    assert len(fake_tools[0]) == 8
    assert model.calls == 9
    assert result[f"{focus}_result"].sources  # successful evidence survives a later limit


@pytest.mark.parametrize("focus", ["sights", "food"])
def test_seven_day_packed_budget_comes_from_coverage(state, fake_tools, monkeypatch, focus):
    state["request"] = state["request"].model_copy(update={
        "days": 7, "pace": "packed", "interests": ["food"],
    })
    desired = targets(state["request"])[focus]
    expected_budget = tool_budget(desired)
    state["instructions"] = {focus: json.dumps({
        "target": desired, "tool_budget": expected_budget,
    })}
    model = use_model(monkeypatch, researchers, [call("web_search", {"query": "Lisbon"})])
    result = make_researcher(focus)(state, {})
    assert result[f"{focus}_result"].status == "failed"
    assert len(fake_tools[0]) == expected_budget
    assert model.calls == expected_budget + 1
    assert expected_budget == (10 if focus == "sights" else 8)


def test_raising_researcher_tool_returns_failed_without_secret(state, monkeypatch, caplog):
    @tool("web_search")
    def raising_search(query: str):
        """READ: Raise an unexpected error."""
        raise RuntimeError("private-tool-secret")

    monkeypatch.setattr(researchers, "SEARCH_TOOLS", [raising_search])
    use_model(monkeypatch, researchers, [call("web_search", {"query": "Lisbon"})])
    result = make_researcher("sights")(state, {})
    assert result["sights_result"].status == "failed"
    assert "private-tool-secret" not in caplog.text
    assert "private-tool-secret" not in result["sights_result"].model_dump_json()


@pytest.mark.parametrize("bad_field", ["source_url", "evidence_quote"])
def test_researcher_drops_bad_quote_or_url(state, fake_tools, monkeypatch, bad_field):
    bad = state["places"][0].model_copy(update={
        bad_field: "https://invented.org" if bad_field == "source_url"
        else "Museum Alpha is a fictional invented claim.",
    })
    use_model(monkeypatch, researchers, [
        call("web_search", {"query": "Lisbon"}),
        structured(ResearchCandidates, {"places": [bad.model_dump()]}),
    ])
    result = make_researcher("sights")(state, {})
    assert result["sights_result"].status == "empty"
    assert result["sights_result"].places == []


def test_researcher_cannot_reuse_registry_without_own_tool_responses(state, fake_tools, monkeypatch):
    use_model(monkeypatch, researchers, [
        structured(ResearchCandidates, {"places": [state["places"][0].model_dump()]}),
    ])
    result = make_researcher("sights")(state, {})
    assert result["sights_result"].status == "empty"
    assert result["sights_result"].sources == {}


@pytest.mark.parametrize(("module", "node", "schema", "invalid"), [
    (planner_module, planner, Itinerary, {"days": "not days"}),
    (critic_module, critic, CriticReview, {"soft_findings": "not findings"}),
])
def test_two_invalid_outputs_return_agent_error(
    state, fake_tools, monkeypatch, module, node, schema, invalid,
):
    model = use_model(monkeypatch, module, [structured(schema, invalid)])
    result = node(state, {})
    assert result["agent_error"].message == "structured_output_invalid_after_one_reask"
    assert model.calls == 2
    if node is critic:
        assert result["critique_version"] == state["draft_version"]


def test_researcher_two_invalid_outputs_fail_safely(state, fake_tools, monkeypatch):
    model = use_model(monkeypatch, researchers, [
        structured(ResearchCandidates, {"places": "not places"}),
    ])
    result = make_researcher("sights")(state, {})
    assert result["sights_result"].status == "failed"
    assert model.calls == 2


@pytest.mark.parametrize(("module", "node", "schema", "valid"), [
    (planner_module, planner, Itinerary, "itinerary"),
    (critic_module, critic, CriticReview, "review"),
])
def test_one_validation_reask_can_recover(state, fake_tools, monkeypatch, module, node, schema, valid):
    data = state["itinerary"] if valid == "itinerary" else {"soft_findings": []}
    invalid = {"days": "bad"} if valid == "itinerary" else {"soft_findings": "bad"}
    model = use_model(monkeypatch, module, [structured(schema, invalid), structured(schema, data)])
    assert node(state, {})["agent_error"] is None
    assert model.calls == 2


def test_planner_llm_json_failure_mode_forces_exactly_one_reask(state, fake_tools, monkeypatch):
    model = use_model(monkeypatch, planner_module, [structured(Itinerary, state["itinerary"])])
    result = planner(state, {"configurable": {"failure_mode": "llm_json"}})
    assert result["agent_error"] is None and model.calls == 2


def test_planner_semantic_errors_get_only_one_reask(state, fake_tools, monkeypatch):
    invalid = state["itinerary"].model_copy(update={"days": []})
    model = use_model(monkeypatch, planner_module, [structured(Itinerary, invalid)])
    result = planner(state, {})
    assert result["agent_error"].message == "structured_output_invalid_after_one_reask"
    assert model.calls == 2


def test_plain_unstructured_responses_get_only_one_reask(state, fake_tools, monkeypatch):
    model = use_model(monkeypatch, planner_module, [AIMessage(content="not JSON")])
    result = planner(state, {})
    assert result["agent_error"].message == "structured_output_invalid_after_one_reask"
    assert model.calls == 2


def test_reask_does_not_reset_tool_budget(state, fake_tools, monkeypatch):
    responses = [
        *[call("web_search", {"query": "Lisbon"}) for _ in range(8)],
        structured(ResearchCandidates, {"places": "invalid"}),
        call("web_search", {"query": "Lisbon again"}),
    ]
    model = use_model(monkeypatch, researchers, responses)
    result = make_researcher("sights")(state, {})
    assert result["sights_result"].error == "specialist_step_limit_reached"
    assert len(fake_tools[0]) == 8 and model.calls == 10


@pytest.mark.parametrize(("module", "node", "tool_name"), [
    (planner_module, planner, "get_weather"), (critic_module, critic, "web_search"),
])
def test_planner_and_critic_loop_limits_are_four(state, fake_tools, monkeypatch, module, node, tool_name):
    state["request"] = state["request"].model_copy(update={"start_date": "2026-11-04"})
    args = (
        {"city": "Lisbon", "start_date": "2026-11-04", "days": 1}
        if tool_name == "get_weather" else {"query": "Lisbon Museum Alpha"}
    )
    model = use_model(monkeypatch, module, [call(tool_name, args)])
    result = node(state, {})
    assert result["agent_error"].message == "specialist_step_limit_reached"
    assert len(fake_tools[0]) == 4 and model.calls == 5


def test_critic_rechecks_checkpoint_registry_without_new_tools(state, fake_tools, monkeypatch):
    # Serialize all evidence to simulate restored state without tool-message history.
    state["source_registry"] = {
        key: record.model_dump() for key, record in state["source_registry"].items()
    }
    state["places"] = [place.model_dump() for place in state["places"]]
    model = use_model(monkeypatch, critic_module, [structured(CriticReview, {"soft_findings": []})])
    result = critic(state, {})
    assert result["critique"].hard_failures == []
    assert result["critique_version"] == 5 and not fake_tools[0]
    assert model.calls == 1


def test_critic_registry_loss_is_hard_failure_even_if_agent_says_clean(state, fake_tools, monkeypatch):
    state["source_registry"] = {}
    use_model(monkeypatch, critic_module, [structured(CriticReview, {"soft_findings": []})])
    result = critic(state, {})
    assert len(result["critique"].hard_failures) == 3
    assert all("without valid evidence" in finding.issue for finding in result["critique"].hard_failures)


def test_critic_keeps_hard_checks_after_two_invalid_outputs(state, fake_tools, monkeypatch):
    state["source_registry"] = {}
    model = use_model(monkeypatch, critic_module, [
        structured(CriticReview, {"soft_findings": "invalid"}),
    ])
    result = critic(state, {})
    assert result["agent_error"] and result["critique"].hard_failures
    assert result["critique_version"] == state["draft_version"] and model.calls == 2


@pytest.mark.parametrize("version", [0, 3, 9])
def test_critique_version_is_exact_on_provider_failure(state, fake_tools, monkeypatch, version, caplog):
    state["draft_version"] = version
    use_model(monkeypatch, critic_module, [AIMessage(content="")], fail_provider=True)
    result = critic(state, {})
    assert result["critique_version"] == version and result["agent_error"]
    assert "private-provider-secret" not in caplog.text
    assert "private-provider-secret" not in result["agent_error"].model_dump_json()


def test_critic_multiple_failed_tools_add_one_soft_finding(state, monkeypatch):
    @tool("web_search")
    def raising_search(query: str):
        """READ: Simulate a tool failure."""
        raise RuntimeError("private-tool-secret")

    monkeypatch.setattr(critic_module, "SEARCH_TOOLS", [raising_search])
    use_model(monkeypatch, critic_module, [
        call("web_search", {"query": "Lisbon Museum Alpha"}),
        call("web_search", {"query": "Lisbon Museum Alpha again"}),
        structured(CriticReview, {"soft_findings": []}),
    ])
    result = critic(state, {})
    assert result["agent_error"] is None
    assert sum("uncertain" in finding.issue for finding in result["critique"].soft_findings) == 1
    assert any("Weather unavailable" in finding.issue for finding in result["critique"].soft_findings)


def test_planner_applies_findings_and_feedback_in_task_prompt(state, fake_tools, monkeypatch):
    state["critique"] = Critique(soft_findings=[
        Finding(issue="Schedule museums earlier.", owner="planner"),
        Finding(issue="Find more sights.", owner="researchers"),
    ])
    state["human_feedback"] = "Prefer the river in the afternoon."
    model = use_model(monkeypatch, planner_module, [structured(Itinerary, state["itinerary"])])
    assert planner(state, {})["agent_error"] is None
    prompt = model.seen_messages[0][-1].content
    assert "Schedule museums earlier." in prompt and "Prefer the river" in prompt
    assert "Find more sights." not in prompt


@pytest.mark.parametrize("mutation", ["unknown_stop", "evidence", "days", "pace", "duplicate", "schema"])
def test_python_hard_check_categories(state, mutation):
    if mutation == "unknown_stop":
        state["itinerary"].days[0].stops[0].place_name = "Invented place"
    elif mutation == "evidence":
        state["source_registry"] = {}
    elif mutation == "days":
        state["request"] = state["request"].model_copy(update={"days": 2})
    elif mutation == "pace":
        state["itinerary"].days[0].stops.pop()
    elif mutation == "duplicate":
        state["itinerary"].days[0].stops[1] = state["itinerary"].days[0].stops[0]
    else:
        state["itinerary"] = {"city": "Lisbon", "days": "invalid"}
    assert hard_checks(state)


def test_doubtful_stop_selection_is_bounded_to_three(state):
    place = state["places"][0]
    state["source_registry"] = {}
    state["itinerary"].days[0].stops.extend([
        Stop(time_slot="evening", place_name=f"Doubtful {index}", note="", source_url=place.source_url)
        for index in range(5)
    ])
    assert len(doubtful_stops(state)) == 3


def test_researcher_must_search_before_reading(state, fake_tools, monkeypatch):
    use_model(monkeypatch, researchers, [
        call("read_page", {"url": "https://example.org/0"}),
    ])
    result = make_researcher("sights")(state, {})
    assert result["sights_result"].status == "failed" and not fake_tools[0]


def test_planner_precondition_prevents_llm_call_when_coverage_short(state, fake_tools, monkeypatch):
    state["places"] = []
    factory = Mock(side_effect=AssertionError("LLM must not be constructed"))
    monkeypatch.setattr(planner_module, "get_llm", factory)
    assert planner(state, {})["agent_error"]
    factory.assert_not_called()


def test_researcher_read_page_limit_is_two(state, fake_tools, monkeypatch):
    use_model(monkeypatch, researchers, [
        call("web_search", {"query": "Lisbon"}),
        *[call("read_page", {"url": "https://example.org/0"}) for _ in range(3)],
    ])
    result = make_researcher("sights")(state, {})
    assert result["sights_result"].status == "failed"
    assert [name for name, _ in fake_tools[0]] == ["web_search", "read_page", "read_page"]


@pytest.mark.parametrize(("probability", "should_fail"), [(60, False), (61, True)])
def test_planner_rain_threshold_requires_grounded_indoor_stops(
    state, monkeypatch, probability, should_fail,
):
    state["request"] = state["request"].model_copy(update={"start_date": "2026-11-04"})
    state["places"][0].description = "This outdoor venue has no covered areas."

    @tool("get_weather", response_format="content_and_artifact")
    def weather(city: str, start_date: str, days: int):
        """READ: Return a fixed threshold forecast."""
        return _result({"daily": {
            "time": [start_date], "precipitation_probability_max": [probability],
        }})

    monkeypatch.setattr(planner_module, "WEATHER_TOOLS", [weather])
    model = use_model(monkeypatch, planner_module, [
        call("get_weather", {"city": "Lisbon", "start_date": "2026-11-04", "days": 1}),
        structured(Itinerary, state["itinerary"]),
    ])
    result = planner(state, {})
    if should_fail:
        assert result["agent_error"].message == "structured_output_invalid_after_one_reask"
        assert model.calls == 3
    else:
        assert result["agent_error"] is None


def test_planner_blocks_weather_without_date_even_if_model_requests_it(state, fake_tools, monkeypatch):
    use_model(monkeypatch, planner_module, [
        call("get_weather", {"city": "Lisbon", "start_date": "2026-11-04", "days": 1}),
    ])
    result = planner(state, {})
    assert result["agent_error"] and not fake_tools[0]


def test_critic_schema_failure_still_returns_hard_finding_and_version(state, fake_tools, monkeypatch):
    state["itinerary"] = {"days": "bad"}
    use_model(monkeypatch, critic_module, [structured(CriticReview, {"soft_findings": []})])
    result = critic(state, {})
    assert result["agent_error"] is not None
    assert result["critique_version"] == state["draft_version"]
    assert result["critique"].hard_failures[0].issue == "schema invalid"


@pytest.mark.parametrize("focus", ["sights", "food"])
def test_research_provider_failure_is_failed_result(state, fake_tools, monkeypatch, focus):
    use_model(monkeypatch, researchers, [AIMessage(content="")], fail_provider=True)
    result = make_researcher(focus)(state, {})
    assert result[f"{focus}_result"].status == "failed"
    assert result[f"{focus}_result"].round_id == state["research_round"]


def test_planner_provider_failure_keeps_previous_draft(state, fake_tools, monkeypatch):
    use_model(monkeypatch, planner_module, [AIMessage(content="")], fail_provider=True)
    result = planner(state, {})
    assert result["agent_error"].agent == "planner"
    assert "itinerary" not in result and "draft_version" not in result
    assert state["draft_version"] == 5


def test_failed_tool_envelopes_fail_researcher_without_raising(state, monkeypatch):
    @tool("web_search", response_format="content_and_artifact")
    def failed_search(query: str):
        """READ: Return a service failure."""
        return _result(error="simulated_failure")

    monkeypatch.setattr(researchers, "SEARCH_TOOLS", [failed_search])
    use_model(monkeypatch, researchers, [call("web_search", {"query": "Lisbon"})])
    assert make_researcher("sights")(state, {})["sights_result"].status == "failed"


def test_critic_cannot_remove_python_checks_with_model_output(state, fake_tools, monkeypatch):
    state["itinerary"].days[0].stops[0].place_name = "Invented place"
    use_model(monkeypatch, critic_module, [structured(CriticReview, {"soft_findings": []})])
    result = critic(state, {})
    assert any("stop not in places" in finding.issue for finding in result["critique"].hard_failures)
    assert result["critique_version"] == 5


def test_invalid_source_record_is_dropped_without_breaking_research_boundary(state, monkeypatch):
    @tool("web_search", response_format="content_and_artifact")
    def malformed_search(query: str):
        """READ: Return malformed evidence metadata."""
        return _result([{
            "source_id": "bad", "url": " ", "retrieved_at": " ",
            "content": "Museum Alpha is an indoor venue in Lisbon.",
        }])

    monkeypatch.setattr(researchers, "SEARCH_TOOLS", [malformed_search])
    use_model(monkeypatch, researchers, [
        call("web_search", {"query": "Lisbon"}),
        structured(ResearchCandidates, {"places": [state["places"][0].model_dump()]}),
    ])
    result = make_researcher("sights")(state, {})
    assert result["sights_result"].status == "empty" and result["sights_result"].sources == {}


def test_critic_does_not_return_malformed_records_to_parent_reducer(state, monkeypatch):
    @tool("web_search", response_format="content_and_artifact")
    def malformed_search(query: str):
        """READ: Return unusable checkpoint metadata."""
        return _result([{
            "source_id": "bad", "url": " ", "retrieved_at": " ",
            "content": "Museum Alpha in Lisbon.",
        }])

    monkeypatch.setattr(critic_module, "SEARCH_TOOLS", [malformed_search])
    use_model(monkeypatch, critic_module, [
        call("web_search", {"query": "Museum Alpha Lisbon"}),
        structured(CriticReview, {"soft_findings": []}),
    ])
    result = critic(state, {})
    assert result["source_registry"] == {}
    assert result["agent_error"] is None


def test_critic_cannot_investigate_an_unselected_fourth_stop(state, fake_tools, monkeypatch):
    use_model(monkeypatch, critic_module, [
        call("web_search", {"query": "Fourth venue outside the itinerary"}),
        structured(CriticReview, {"soft_findings": []}),
    ])
    result = critic(state, {})
    assert result["agent_error"] is None
    assert not fake_tools[0]  # request was blocked before any service/tool execution
    assert sum("uncertain" in finding.issue for finding in result["critique"].soft_findings) == 1


@pytest.mark.parametrize("bad_field", ["source_id", "source_url", "evidence_quote"])
def test_researcher_repairs_evidence_with_one_reask(state, fake_tools, monkeypatch, bad_field):
    good = state["places"][0]
    bad = good.model_copy(update={bad_field: "invented unsupported value"})
    model = use_model(monkeypatch, researchers, [
        call("web_search", {"query": "Lisbon"}),
        structured(ResearchCandidates, {"places": [bad.model_dump()]}),
        structured(ResearchCandidates, {"places": [good.model_dump()]}),
    ])
    result = make_researcher("sights")(state, {})["sights_result"]
    assert result.status == "ok" and result.places == [good]
    assert model.calls == 3 and len(fake_tools[0]) == 1
    correction = str(model.seen_messages[-1])
    assert "rejections" in correction and "sources" in correction
    assert "Evidence validation details" in correction


def test_researcher_persistent_bad_quote_is_dropped_after_one_correction(state, fake_tools, monkeypatch):
    bad = state["places"][0].model_copy(update={"evidence_quote": "Museum Alpha is an invented claim."})
    model = use_model(monkeypatch, researchers, [
        call("web_search", {"query": "Lisbon"}),
        structured(ResearchCandidates, {"places": [bad.model_dump()]}),
    ])
    result = make_researcher("sights")(state, {})["sights_result"]
    assert result.status == "empty" and not result.places
    assert model.calls == 3 and len(fake_tools[0]) == 1
    assert "quote_not_in_source: 1" in result.warnings[0]


def test_critic_tool_budget_leaves_only_structured_review(state, fake_tools, monkeypatch):
    model = use_model(monkeypatch, critic_module, [
        *[call("web_search", {"query": "Lisbon Museum Alpha"}) for _ in range(4)],
        structured(CriticReview, {"soft_findings": []}),
    ])
    result = critic(state, {})
    assert result["agent_error"] is None
    assert result["critique_version"] == state["draft_version"]
    assert len(fake_tools[0]) == 4
    assert model.bound_names[-1] == ["CriticReview"]
    assert "budget is exhausted" in str(model.seen_messages[-1])


def test_critic_extract_limit_removes_extract_without_removing_search(state, fake_tools, monkeypatch):
    model = use_model(monkeypatch, critic_module, [
        call("read_page", {"url": state["places"][0].source_url}),
        call("read_page", {"url": state["places"][0].source_url}),
        structured(CriticReview, {"soft_findings": []}),
    ])
    result = critic(state, {})
    assert result["agent_error"] is None
    assert "read_page" not in model.bound_names[-1]
    assert set(model.bound_names[-1]) == {"web_search", "CriticReview"}


def test_critic_hard_failures_survive_tool_budget_finalization(state, fake_tools, monkeypatch):
    state["source_registry"] = {}
    use_model(monkeypatch, critic_module, [
        *[call("web_search", {"query": "Lisbon Museum Alpha"}) for _ in range(4)],
        structured(CriticReview, {"soft_findings": []}),
    ])
    result = critic(state, {})
    assert result["agent_error"] is None
    assert result["critique"].hard_failures
    assert result["critique_version"] == state["draft_version"]


def test_critic_batch_crossing_budget_declines_fifth_call_then_finishes(state, fake_tools, monkeypatch):
    def batch(names):
        return AIMessage(content="", tool_calls=[{
            "name": name, "args": ({"query": "Lisbon Museum Alpha"} if name == "web_search"
                                     else {"url": state["places"][0].source_url}),
            "id": f"batch-{index}-{name}", "type": "tool_call",
        } for index, name in enumerate(names)])

    model = use_model(monkeypatch, critic_module, [
        batch(["web_search", "web_search", "read_page"]),
        batch(["web_search", "read_page"]),
        structured(CriticReview, {"soft_findings": []}),
    ])
    result = critic(state, {})
    assert result["agent_error"] is None
    assert len(fake_tools[0]) == 4  # fifth requested call never executes
    assert model.bound_names[-1] == ["CriticReview"]
    assert any("tool failed" in finding.issue.casefold() for finding in result["critique"].soft_findings)
    assert result["critique_version"] == state["draft_version"]
    assert not result["critique"].hard_failures


def test_weather_notes_include_only_received_dates_temperatures_and_units(state):
    from src.agents._runtime import SpecialistRun
    from src.agents.planner import _weather_notes

    forecast = {"daily": {"time": ["2026-11-15"], "precipitation_probability_max": [2],
                           "temperature_2m_min": [20], "temperature_2m_max": [29]},
                "units": {"temperature_2m_min": " C", "temperature_2m_max": " C"}}
    draft = _weather_notes(state["itinerary"], forecast, SpecialistRun([], 4))
    note = draft.days[0].weather_note
    assert "2026-11-15" in note and "2%" in note
    assert "Low: 20 C" in note and "High: 29 C" in note
    forecast["daily"].pop("temperature_2m_min")
    forecast["daily"].pop("temperature_2m_max")
    note = _weather_notes(state["itinerary"], forecast, SpecialistRun([], 4)).days[0].weather_note
    assert "Low:" not in note and "High:" not in note


def test_researcher_short_valid_output_gets_bounded_coverage_correction(state, fake_tools, monkeypatch):
    sights = [place.model_dump() for place in state["places"] if place.category != "food"]
    model = use_model(monkeypatch, researchers, [
        call("web_search", {"query": "Lisbon"}),
        structured(ResearchCandidates, {"places": sights[:1]}),
        structured(ResearchCandidates, {"places": sights}),
    ])
    result = make_researcher("sights")(state, {})["sights_result"]
    assert result.status == "ok" and len(result.places) == 2
    assert model.calls == 3 and len(fake_tools[0]) == 1
    assert "minimum_required" in str(model.seen_messages[-1])


def test_planner_multiple_itineraries_gets_specific_single_response_correction(state, fake_tools, monkeypatch):
    data = state["itinerary"].model_dump()
    multiple = AIMessage(content="", tool_calls=[{
        "name": "Itinerary", "args": data, "id": f"alternative-{i}", "type": "tool_call",
    } for i in range(5)])
    model = use_model(monkeypatch, planner_module, [multiple, structured(Itinerary, data)])
    result = planner(state, {})
    assert result["agent_error"] is None
    assert model.calls == 2
    assert all(options["parallel_tool_calls"] is False for options in model.bound_options)
    assert "EXACTLY ONE" in str(model.seen_messages[-1])


def test_planner_semantic_correction_identifies_day_and_stop_requirements(state, fake_tools, monkeypatch):
    bad = state["itinerary"].model_dump()
    bad["days"][0]["stops"] = bad["days"][0]["stops"][:2]
    model = use_model(monkeypatch, planner_module, [structured(Itinerary, bad),
                                                           structured(Itinerary, state["itinerary"])])
    result = planner(state, {})
    assert result["agent_error"] is None and model.calls == 2
    feedback = str(model.seen_messages[-1])
    assert "required_stops_per_day" in feedback and "day outside its pace limit" in feedback


def test_researcher_crossing_budget_finalizes_successful_receipts(state, fake_tools, monkeypatch):
    def searches(count, prefix):
        return AIMessage(content="", tool_calls=[{
            "name": "web_search", "args": {"query": "Lisbon food"},
            "id": f"{prefix}-{index}", "type": "tool_call",
        } for index in range(count)])

    foods = [place.model_dump() for place in state["places"] if place.category == "food"]
    model = use_model(monkeypatch, researchers, [
        searches(1, "a"), searches(3, "b"), searches(3, "c"), searches(3, "d"),
        structured(ResearchCandidates, {"places": foods}),
    ])
    result = make_researcher("food")(state, {})["food_result"]
    assert result.status == "ok" and result.places
    assert len(fake_tools[0]) == 8
    assert model.bound_names[-1] == ["ResearchCandidates"]
    assert any("excess calls skipped" in warning for warning in result.warnings)


def test_planner_schema_failure_reports_field_not_rejected_values(state, fake_tools, monkeypatch):
    bad = state["itinerary"].model_dump()
    bad["days"][0]["stops"][0]["time_slot"] = "private-rejected-value"
    model = use_model(monkeypatch, planner_module, [structured(Itinerary, bad)])
    result = planner(state, {})
    assert result["agent_error"].message == "structured_output_invalid_after_one_reask"
    assert model.calls == 2
    assert "time_slot" in result["warnings"][0]
    assert "private-rejected-value" not in str(result["warnings"])


def test_planner_semantic_failure_reports_day_and_constraint(state, fake_tools, monkeypatch):
    bad = state["itinerary"].model_dump()
    bad["days"][0]["stops"] = bad["days"][0]["stops"][:2]
    use_model(monkeypatch, planner_module, [structured(Itinerary, bad)])
    result = planner(state, {})
    assert result["agent_error"]
    assert "Day 1: day outside its pace limit" in result["warnings"][0]


@pytest.mark.parametrize(("days", "pace", "stops"), [(2, "normal", 4), (7, "packed", 5), (1, "relaxed", 3)])
def test_planner_response_schema_exposes_exact_requested_counts(days, pace, stops):
    from src.agents.planner import _itinerary_schema

    trip = TripRequest(city="Agra", days=days, budget="mid", interests=[], pace=pace)
    schema = _itinerary_schema(trip)
    assert schema["title"] == "Itinerary"
    assert schema["properties"]["days"]["minItems"] == schema["properties"]["days"]["maxItems"] == days
    day = schema["$defs"]["DayPlan"]["properties"]
    assert day["stops"]["minItems"] == day["stops"]["maxItems"] == stops
    assert day["day_number"]["maximum"] == days


def test_planner_model_receives_count_bounds_and_returns_static_model(state, fake_tools, monkeypatch):
    model = use_model(monkeypatch, planner_module, [structured(Itinerary, state["itinerary"])])
    result = planner(state, {})
    assert result["agent_error"] is None
    assert type(result["itinerary"]) is Itinerary
    schema = next(tool["function"]["parameters"] for tool in model.bound_tools[0]
                  if tool["function"]["name"] == "Itinerary")
    assert schema["properties"]["days"]["minItems"] == schema["properties"]["days"]["maxItems"] == 1
    day = schema.get("$defs", {}).get("DayPlan") or schema["properties"]["days"]["items"]
    assert day["properties"]["stops"]["minItems"] == day["properties"]["stops"]["maxItems"] == 3
