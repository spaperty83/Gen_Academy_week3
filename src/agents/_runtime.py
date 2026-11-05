"""Per-invocation agent limits, evidence receipts and one validation re-ask."""

import json
from collections.abc import Callable, Mapping, Sequence
from html import escape, unescape
from threading import Lock
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt.tool_node import ToolCallRequest
from pydantic import BaseModel, ValidationError

from src.evidence import registry_entries
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
        search_first: bool = False,
        read_limit: int | None = None,
        weather_request: dict[str, Any] | None = None,
        validator: Callable[[BaseModel], None] | None = None,
        simulate_first_parse: bool = False,
    ) -> None:
        self.allowed = {tool.name for tool in tools}
        self.budget = budget
        self.strict_tools = strict_tools
        self.search_first = search_first
        self.read_limit = read_limit
        self.weather_request = weather_request
        self.validator = validator
        self.simulate_first_parse = simulate_first_parse
        self.tool_calls = 0
        self.model_calls = 0
        self.read_calls = 0
        self.validation_reasks = 0
        self.tool_failed = False
        self.tool_messages: list[ToolMessage] = []
        self._lock = Lock()

    def _reask(self, _: Exception) -> str:
        """Allow exactly one schema/semantic correction in this invocation."""
        if self.validation_reasks:
            raise StructuredOutputExhausted("Structured output is invalid") from None
        self.validation_reasks += 1
        self.simulate_first_parse = False
        return (
            "The previous structured response did not validate. Return one corrected "
            "response matching the schema and all task constraints. Do not repeat tools "
            "unless necessary; the original tool-call budget still applies."
        )

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Correct invalid output inside the same run, without provider retries."""
        prefix = []
        while True:
            self.model_calls += 1
            if self.model_calls > self.budget + 4:
                raise StepLimitReached("Model-call limit reached")
            response = handler(request)
            last_ai = next(
                (message for message in reversed(response.result) if isinstance(message, AIMessage)),
                None,
            )
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
                raise StepLimitReached("Tool-call budget reached")
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
            if name == "get_weather" and (
                self.weather_request is None
                or request.tool_call["args"] != self.weather_request
            ):
                raise ToolExecutionFailed("Weather requires the requested dates and city")
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
                data = {"ok": False, "data": None, "error": "tool_failed"}
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
        if failed and self.strict_tools:
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
