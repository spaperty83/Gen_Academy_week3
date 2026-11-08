"""Python hard checks plus bounded, explicitly soft LLM spot-checks."""

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict, Field

from src.agents._checks import doubtful_stops, hard_checks
from src.agents._runtime import SpecialistRun, failure_message, source_records
from src.agents.prompts import CRITIC
from src.config import get_llm
from src.state import AgentError, Critique, Finding, TripState
from src.tools import SEARCH_TOOLS

LOGGER = logging.getLogger(__name__)
TOOL_UNCERTAINTY = "Spot-check tool failed; source confirmation remains uncertain."


class CriticReview(BaseModel):
    """The model may suggest soft findings, never remove Python hard failures."""

    model_config = ConfigDict(extra="forbid")
    soft_findings: list[Finding] = Field(default_factory=list)


def critic(state: TripState, config: RunnableConfig) -> dict[str, Any]:
    """Always label checks with the exact draft version, even if the model fails."""
    version = state.get("draft_version", 0)
    hard: list[Finding] = []
    soft: list[Finding] = []
    run = None
    error_result = None
    try:
        hard = hard_checks(state)
        targets = doubtful_stops(state)
        run = SpecialistRun(SEARCH_TOOLS, 4, read_limit=2, spotcheck_targets=targets)
        review = run.execute(
            get_llm(), SEARCH_TOOLS, CriticReview, CRITIC,
            {
                "request": state["request"], "itinerary": state["itinerary"],
                "draft_version": version, "python_hard_findings": hard,
                "doubtful_stops": targets, "weather": state.get("weather"),
                "instructions": state.get("instructions", {}).get("critic", ""),
            }, config, "critic",
        )
        soft = review.soft_findings
    except Exception as error:  # noqa: BLE001 - keep Python checks when the agent fails
        LOGGER.warning("Critic failed (%s)", type(error).__name__)
        error_result = AgentError(agent="critic", message=failure_message(error))
        if not hard:
            # A malformed draft must remain a hard failure even if preparation fails.
            try:
                hard = hard_checks(state)
            except Exception:  # noqa: BLE001 - last-resort schema failure is fail-closed
                hard = [Finding(issue="schema invalid", owner="planner")]
    if state.get("weather") is None and not any(
        "weather" in finding.issue.casefold() for finding in soft
    ):
        soft.append(Finding(issue="Weather unavailable; no forecast was invented.", owner="planner"))
    if run is not None and run.tool_failed:
        soft = [finding for finding in soft if finding.issue != TOOL_UNCERTAINTY]
        soft.append(Finding(issue=TOOL_UNCERTAINTY, owner="researchers"))
    return {
        "critique": Critique(hard_failures=hard, soft_findings=soft),
        "critique_version": version, "agent_error": error_result,
        "source_registry": source_records(run.tool_messages if run else []),
    }
