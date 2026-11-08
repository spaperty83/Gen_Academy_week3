"""Independent, non-raising research branches with evidence-backed results."""

import json
import logging
import re
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict, Field

from src.agents._runtime import (
    OutputInvalid,
    SpecialistRun,
    ToolExecutionFailed,
    failure_message,
    source_records,
)
from src.agents.prompts import RESEARCHER
from src.config import get_llm
from src.coverage import needed, targets, tool_budget
from src.evidence import normalize_name, validate_places
from src.state import Place, ResearchResult, SourceRecord, TripRequest, TripState
from src.tools import SEARCH_TOOLS

LOGGER = logging.getLogger(__name__)


class ResearchCandidates(BaseModel):
    """Only candidates are model-authored; the node owns status and round IDs."""

    model_config = ConfigDict(extra="forbid")
    places: list[Place] = Field(default_factory=list)


def research_directive(
    state: TripState, focus: Literal["sights", "food"],
) -> tuple[int, int, Any]:
    """Read target/budget from JSON or labeled prose; missing budget falls back to 8."""
    trip = TripRequest.model_validate(state["request"])
    instruction = state.get("instructions", {}).get(focus, "")
    details = instruction if isinstance(instruction, Mapping) else {}
    if isinstance(instruction, str):
        try:
            decoded = json.loads(instruction)
            if isinstance(decoded, dict):
                details = decoded
        except ValueError:
            pass

    def number(key: str, pattern: str) -> int | None:
        value = details.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        match = re.search(pattern, str(instruction), flags=re.IGNORECASE)
        return int(match.group(1)) if match else None

    target = number("target", r"target(?:\s+count)?[\s:=]+(\d+)")
    stated_budget = number("tool_budget", r"tool[\s_-]*(?:call[\s_-]*)?budget[\s:=]+(\d+)")
    desired = target if target is not None and target > 0 else targets(trip)[focus]
    # The formula is authoritative when a target is supplied. A budget-only
    # instruction is bounded to the documented range; omitted instructions use 8.
    budget = tool_budget(desired) if target is not None else (
        min(14, max(8, stated_budget)) if stated_budget is not None else 8
    )
    return desired, budget, instruction


def _sources(run: SpecialistRun | None) -> dict[str, SourceRecord]:
    """Keep only successful read receipts belonging to this invocation."""
    return source_records(run.tool_messages if run else [])


def make_researcher(
    focus: Literal["sights", "food"],
) -> Callable[[TripState, RunnableConfig], dict[str, Any]]:
    """Create a branch that writes only its own result, including evidence receipts."""
    if focus not in {"sights", "food"}:
        raise ValueError("Research focus must be sights or food")

    def researcher(state: TripState, config: RunnableConfig) -> dict[str, Any]:
        """Catch all branch failures so parallel research can still finish."""
        run = None
        round_id = 0
        entries: dict[str, SourceRecord] = {}
        try:
            round_id = state.get("research_round", 0)
            desired, budget, instruction = research_directive(state, focus)
            run = SpecialistRun(
                SEARCH_TOOLS, budget, strict_tools=True, optional_tools={"read_page"},
                search_first=True, read_limit=2,
            )
            def check_evidence(output: ResearchCandidates) -> None:
                # Evidence correction shares the existing single re-ask and budget.
                # If still invalid after that correction, retain the valid subset.
                receipts = _sources(run)
                valid, rejected = validate_places(output.places, receipts)
                minimum = needed(state["request"])[focus]
                valid_count = len({normalize_name(place.name) for place in valid
                                   if (place.category == "food") == (focus == "food")})
                if (rejected or valid_count < minimum) and not run.validation_reasks:
                    run.validation_feedback = {
                        "rejections": rejected, "minimum_required": minimum,
                        "valid_count": valid_count, "target": desired,
                        "sources": {key: record.model_dump(mode="json") for key, record in receipts.items()},
                    }
                    raise OutputInvalid("Candidate evidence did not match tool receipts")

            run.validator = check_evidence
            output = run.execute(
                get_llm(), SEARCH_TOOLS, ResearchCandidates, RESEARCHER.format(focus=focus),
                {
                    "request": state["request"], "focus": focus, "target": desired,
                    "tool_budget": budget, "instructions": instruction,
                    "human_feedback": state.get("human_feedback", ""),
                }, config, f"{focus}_researcher",
            )
            if run.failed_tools - {"read_page"}:
                raise ToolExecutionFailed("Research tool failed")
            entries = _sources(run)
            kept, dropped = validate_places(output.places, entries)
            allowed_urls = {record.url for record in entries.values()}
            places = [
                place.model_copy(update={
                    "found_by": focus,
                    "retrieved_at": entries[place.source_id].retrieved_at,
                    "extra_sources": [url for url in place.extra_sources if url in allowed_urls],
                })
                for place in kept
                if (place.category == "food") == (focus == "food")
            ]
            rejection_counts = dict(Counter(reason.split(": ", 1)[1] for reason in dropped))
            if dropped:
                LOGGER.info("Research evidence rejected for %d candidates", len(dropped))
            result = ResearchResult(
                status="ok" if places else "empty", places=places, round_id=round_id,
                sources=entries,
                warnings=(["read_page failed; using search snippets."]
                          if "read_page" in run.failed_tools else [])
                + ([(f"{focus} candidates dropped: invalid evidence or instruction-like source text "
                   f"({', '.join(f'{reason}: {count}' for reason, count in rejection_counts.items())}).")]
                   if dropped else []),
            )
        except Exception as error:  # noqa: BLE001 - research branches must never raise
            LOGGER.warning("%s researcher failed (%s)", focus, type(error).__name__)
            if run is not None:
                entries = _sources(run)
            result = ResearchResult(
                status="failed", error=failure_message(error),
                round_id=round_id if type(round_id) is int and round_id >= 0 else 0,
                sources=entries,
            )
        return {f"{focus}_result": result}

    return researcher
