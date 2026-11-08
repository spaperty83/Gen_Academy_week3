"""Per-invocation agent limits, evidence receipts and one validation re-ask."""

import json
from collections.abc import Callable, Mapping, Sequence
from html import escape, unescape
from threading import Lock
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain.agents.structured_output import (
    MultipleStructuredOutputsError,
    ToolStrategy,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.config import get_stream_writer
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt.tool_node import ToolCallRequest
from pydantic import BaseModel, ValidationError

from src.evidence import normalize_name, registry_entries
from src.state import SourceRecord


class StepLimitReached(Exception):
    """The specialist exhausted its bounded tool/model work."""


class StructuredOutputExhausted(Exception):
    """Structured output remained invalid after its one re-ask."""


class OutputInvalid(Exception):
    """A schema-valid result violated a grounded task constraint."""


class ToolExecutionFailed(Exception):
    """A tool failed without leaking its exception or arguments."""


def tool_data(value: Any) -> str:
    """Escape untrusted JSON so it cannot close the tool-data delimiter."""
    return "<tool_data>" + escape(json.dumps(
        value, ensure_ascii=False, default=_json_default,
    )) + "</tool_data>"


def _json_default(value: Any) -> Any:
    """Serialize state models as data, never serialize configuration or clients."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError("Task data is not JSON serializable")


def envelope(message: ToolMessage) -> dict[str, Any] | None:
    """Read the machine envelope without interpreting tool text as instructions."""
    if isinstance(message.artifact, dict):
        return message.artifact
    if not isinstance(message.content, str):
        return None
    content = message.content
    if content.startswith("<tool_data>") and content.endswith("</tool_data>"):
        content = unescape(content[len("<tool_data>"):-len("</tool_data>")])
    try:
        result = json.loads(content)
    except (ValueError, TypeError):
        return None
    return result if isinstance(result, dict) and "ok" in result else None



def source_records(messages: Sequence[ToolMessage]) -> dict[str, SourceRecord]:
    """Keep checkpoint-safe evidence only; malformed tool metadata is discarded."""
    entries = {}
    for key, record in registry_entries(messages).items():
        try:
            entries[key] = SourceRecord.model_validate(record)
        except ValidationError:
            continue
    return entries


def failure_message(error: Exception) -> str:
    """Return only static, secret-safe boundary errors."""
    if isinstance(error, StructuredOutputExhausted):
        return "structured_output_invalid_after_one_reask"
    if isinstance(error, (StepLimitReached, GraphRecursionError)):
        return "specialist_step_limit_reached"
    if isinstance(error, ToolExecutionFailed):
        return "specialist_tool_failed"
    return "specialist_failed"


class SpecialistRun(AgentMiddleware):
    """Enforce shared budgets even across validation re-asks and parallel calls."""

    def __init__(
        self,
        tools: Sequence[BaseTool],
        budget: int,
        *,
        strict_tools: bool = False,
        optional_tools: set[str] | None = None,
        spotcheck_targets: list[dict[str, Any]] | None = None,
        search_first: bool = False,
        read_limit: int | None = None,
        weather_request: dict[str, Any] | None = None,
        validator: Callable[[BaseModel], None] | None = None,
        simulate_first_parse: bool = False,
    ) -> None:
        self.allowed = {tool.name for tool in tools}
        self.budget = budget
        self.strict_tools = strict_tools
        self.optional_tools = optional_tools or set()
        self.spotcheck_targets = spotcheck_targets[:3] if spotcheck_targets is not None else None
        self.search_first = search_first
        self.read_limit = read_limit
        self.weather_request = weather_request
        self.validator = validator
        self.simulate_first_parse = simulate_first_parse
        self.tool_calls = 0
        self.model_calls = 0
        self.read_calls = 0
        self.validation_reasks = 0
        self.validation_feedback: dict[str, Any] | None = None
        self.tool_failed = False
        self.budget_declined = False
        self.failed_tools: set[str] = set()
        self.tool_messages: list[ToolMessage] = []
        self.progress_writer: Callable[[Any], None] | None = None
        self.agent_name = ""
        self._lock = Lock()

    def _reask(self, error: Exception) -> str:
        """Allow exactly one schema/semantic correction in this invocation."""
        if self.validation_reasks:
            raise StructuredOutputExhausted("Structured output is invalid") from None
        self.validation_reasks += 1
        self.simulate_first_parse = False
        feedback = (
            "The previous structured response did not validate. Return one corrected "
            "response matching the schema and all task constraints. Do not repeat tools "
            "unless necessary; the original tool-call budget still applies."
        )
        if isinstance(error, MultipleStructuredOutputsError):
            feedback += (
                " Return EXACTLY ONE structured-response tool call containing the complete result. "
                "For an itinerary, put every day in that one response; do not emit alternative drafts "
                "or separate response-tool calls for individual days."
            )
        if self.validation_feedback is not None:
            feedback += ("\nCorrect evidence and, if below minimum_required, find additional distinct supported "
                         "places using the remaining tool budget. Never invent candidates. "
                         "Evidence validation details (untrusted data): ") + tool_data(self.validation_feedback)
        return feedback

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Correct invalid output inside the same run, without provider retries."""
        request = request.override(model_settings={
            **request.model_settings, "parallel_tool_calls": False,
        })
        prefix = []
        while True:
            self.model_calls += 1
            if self.model_calls > self.budget + 4:
                raise StepLimitReached("Model-call limit reached")
            if self.tool_calls >= self.budget:
                # Keep ToolStrategy's response schema, but expose no read tools.
                # LangChain binds the structured response tool independently.
                request = request.override(
                    tools=[],
                    messages=[*request.messages, HumanMessage(content=(
                        "The external tool-call budget is exhausted. Do not call more read tools. "
                        "Return the final structured response now using the evidence already available. "
                        "Record any incomplete checks as uncertainty; never invent a result."
                    ))],
                )
            elif self.read_limit is not None and self.read_calls >= self.read_limit:
                request = request.override(tools=[
                    tool for tool in request.tools
                    if (tool.get("name") if isinstance(tool, dict) else tool.name) != "read_page"
                ])
            response = handler(request)
            last_ai = next(
                (message for message in reversed(response.result) if isinstance(message, AIMessage)),
                None,
            )
            if (
                self.tool_calls >= self.budget and last_ai is not None
                and any(call["name"] in self.allowed for call in last_ai.tool_calls)
            ):
                raise StepLimitReached("Model ignored final-response-only tool limit")
            try:
                if response.structured_response is not None:
                    if self.simulate_first_parse:
                        raise OutputInvalid("Simulated first parse failure")
                    if self.validator is not None:
                        self.validator(response.structured_response)
                elif last_ai is not None and not last_ai.tool_calls:
                    raise OutputInvalid("Missing structured response")
                else:
                    return ModelResponse(
                        result=[*prefix, *response.result],
                        structured_response=response.structured_response,
                    )
            except OutputInvalid as error:
                correction = HumanMessage(content=self._reask(error))
                prefix.extend([*response.result, correction])
                request = request.override(messages=[*request.messages, *response.result, correction])
                continue
            return ModelResponse(
                result=[*prefix, *response.result],
                structured_response=response.structured_response,
            )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage],
    ) -> ToolMessage:
        """Run one tool once; wrap receipts and never retry at the agent level."""
        name = request.tool_call["name"]
        with self._lock:
            if name not in self.allowed:
                raise ToolExecutionFailed("Tool is not allowed")
            if self.tool_calls >= self.budget:
                # Decline only the excess call, without executing it or aborting
                # the other completed calls in this model-authored batch.
                data = {"ok": False, "data": None, "error": "tool_budget_exhausted"}
                result = ToolMessage(
                    content=tool_data(data), artifact=data, name=name,
                    tool_call_id=request.tool_call["id"], status="error",
                )
                self.tool_messages.append(result)
                self.budget_declined = True
                if not self.strict_tools:
                    self.tool_failed = True
                    self.failed_tools.add(name)
                return result
            self.tool_calls += 1
            if name == "read_page":
                self.read_calls += 1
                searched = any(message.name == "web_search" for message in self.tool_messages)
                if (
                    (self.search_first and not searched)
                    or (self.read_limit is not None and self.read_calls > self.read_limit)
                ):
                    raise ToolExecutionFailed("Read-page policy violated")
        try:
            if self.spotcheck_targets is not None:
                arguments = request.tool_call["args"]
                if name == "read_page":
                    allowed = any(arguments.get("url") == target["source_url"]
                                  for target in self.spotcheck_targets)
                else:
                    query = " " + normalize_name(str(arguments.get("query", ""))) + " "
                    allowed = any(
                        " " + normalize_name(target["place_name"]) + " " in query
                        for target in self.spotcheck_targets
                    )
                if not allowed:
                    raise ToolExecutionFailed("Spot-check outside the three selected stops")
            if name == "get_weather" and (
                self.weather_request is None
                or request.tool_call["args"] != self.weather_request
            ):
                raise ToolExecutionFailed("Weather requires the requested dates and city")
            if self.progress_writer is not None:
                args = request.tool_call["args"]
                self.progress_writer({
                    "kind": "tool_call", "agent": self.agent_name, "tool": name,
                    "detail": str(args.get("query", args.get("url", args.get("city", ""))))[:240],
                })
            result = handler(request)
            if not isinstance(result, ToolMessage):
                raise ToolExecutionFailed("Unsupported tool result")
            data = envelope(result)
            if data is None:
                data = {
                    "ok": result.status != "error", "data": result.content,
                    "error": "tool_failed" if result.status == "error" else None,
                }
            # An arbitrary custom tool must not leak error details back to a model.
            if result.status == "error" or data.get("ok") is not True:
                code = data.get("error")
                safe_codes = {
                    "simulated_failure", "extract_failed", "search_failed", "weather_failed",
                    "beyond_forecast_window", "weather_unavailable", "city_not_found", "invalid_input",
                }
                data = {"ok": False, "data": None,
                        "error": code if isinstance(code, str) and code in safe_codes else "tool_failed"}
            result = result.model_copy(update={
                "content": tool_data(data), "artifact": data, "name": name,
            })
        except Exception:  # noqa: BLE001 - preserve the secret-safe tool boundary
            result = ToolMessage(
                content=tool_data({"ok": False, "data": None, "error": "tool_failed"}),
                artifact={"ok": False, "data": None, "error": "tool_failed"},
                name=name, tool_call_id=request.tool_call["id"], status="error",
            )
        with self._lock:
            self.tool_messages.append(result)
            failed = envelope(result).get("ok") is not True
            self.tool_failed |= failed
            if failed:
                self.failed_tools.add(name)
        if failed and self.strict_tools and name not in self.optional_tools:
            raise ToolExecutionFailed("Research tool failed")
        return result

    def execute(
        self,
        model: BaseChatModel,
        tools: Sequence[BaseTool],
        schema: type[BaseModel],
        prompt: str,
        payload: Mapping[str, Any],
        config: RunnableConfig,
        name: str,
    ) -> BaseModel:
        """Build the installed LangChain agent and pass runtime config throughout."""
        self.agent_name = name
        try:
            # Capture the parent workflow writer before entering the child agent.
            self.progress_writer = get_stream_writer()
        except RuntimeError:
            self.progress_writer = None
        agent = create_agent(
            model=model, tools=list(tools), system_prompt=prompt,
            response_format=ToolStrategy(
                schema, handle_errors=self._reask,
                tool_message_content="Structured response received.",
            ),
            middleware=[self], name=name,
        )
        # Sequential tool batches keep search-before-read deterministic. Separate
        # researcher nodes can still run concurrently in the parent LangGraph.
        run_config = {
            **config, "max_concurrency": 1,
            "recursion_limit": min(config.get("recursion_limit", 1000), 4 * self.budget + 16),
        }
        result = agent.invoke(
            {"messages": [HumanMessage(content=tool_data(dict(payload)))]},
            config=run_config,
        )
        output = result.get("structured_response")
        if output is None:
            raise StructuredOutputExhausted("Missing final structured output")
        return schema.model_validate(output)
