"""Per-trip state lives in checkpoints.sqlite under its thread_id until deleted.

Nothing is shared between trips; no secrets are stored in state. This module
defines the schema, not the checkpointer lifecycle. Accumulator resets use native
LangGraph Overwrite(value=...) in node updates, never empty merge/add updates.
Draft versions only increase within a trip; start a new thread for a new trip.
"""

import operator
from datetime import date
from typing import Annotated, Any, Literal, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

Budget = Literal["budget", "mid", "luxury"]
Pace = Literal["relaxed", "normal", "packed"]
ChangeType = Literal["city", "dates", "days", "interests", "budget", "pace", "order"]
NextAgent = Literal[
    "sights", "food", "both", "planner", "critic", "traveler", "recovery", "stop"
]
HumanDecision = Literal[
    "approve", "revise", "retry", "retry_export", "new_trip", "cancel"
]
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
TripDays = Annotated[int, Field(ge=1, le=7, strict=True)]
ISODate = Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]

MAX_ROUND_STEPS = 12
MAX_TRIP_STEPS = 30
MAX_CORRECTION_CYCLES = 2
MAX_RESEARCHER_ATTEMPTS = 2


class _Model(BaseModel):
    """Reject misspelled or unexpected fields at structured-output boundaries."""

    model_config = ConfigDict(extra="forbid", revalidate_instances="always")

    @model_validator(mode="before")
    @classmethod
    def restored_model_fields(cls, value: Any) -> Any:
        """Revalidate same-named checkpoint models surviving a module reload."""
        if (
            isinstance(value, BaseModel) and not isinstance(value, cls)
            and type(value).__module__ == cls.__module__
            and type(value).__name__ == cls.__name__
        ):
            # Preserve patch omission versus explicit null while validating all
            # supplied values against the current schema (never trust old types).
            return value.model_dump(exclude_unset=True)
        return value


class _DatedModel(_Model):
    """Validate optional calendar dates without changing their ISO strings."""

    @field_validator("start_date", check_fields=False)
    @classmethod
    def validate_start_date(cls, value: str | None) -> str | None:
        """Reject impossible dates such as February 30."""
        if value is not None:
            date.fromisoformat(value)
        return value


class TripRequest(_DatedModel):
    """The current or original validated traveler request."""

    city: NonEmptyStr
    days: TripDays
    start_date: ISODate | None = None
    budget: Budget
    interests: list[str]
    pace: Pace


class RequestPatch(_DatedModel):
    """Partial revision; model_fields_set distinguishes omission from null dates."""

    city: NonEmptyStr | None = None
    days: TripDays | None = None
    start_date: ISODate | None = None
    budget: Budget | None = None
    interests: list[str] | None = None
    pace: Pace | None = None
    other_notes: str | None = None
    change_types: list[ChangeType] = Field(default_factory=list)


class SourceRecord(_Model):
    """Checkpointed evidence from a real read-tool response."""

    url: NonEmptyStr
    retrieved_at: NonEmptyStr
    text: str = Field(max_length=4000)


class Place(_Model):
    """A research candidate carrying its precise source and evidence quote."""

    name: NonEmptyStr
    category: Literal["sight", "activity", "food", "other"]
    description: str
    found_by: Literal["sights", "food"]
    source_id: NonEmptyStr
    source_url: NonEmptyStr
    source_title: str
    retrieved_at: NonEmptyStr
    evidence_quote: NonEmptyStr
    extra_sources: list[str] = Field(default_factory=list)


class ResearchResult(_Model):
    """One research branch result, scoped to its dispatch round."""

    status: Literal["ok", "empty", "failed"]
    places: list[Place] = Field(default_factory=list)
    error: str | None = None
    round_id: int = Field(ge=0, strict=True)
    # Receipts travel in this branch-owned field; merge publishes shared evidence.
    sources: dict[str, SourceRecord] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class Stop(_Model):
    """A planned visit referencing a researched place by name."""

    time_slot: Literal["morning", "afternoon", "evening"]
    place_name: NonEmptyStr
    note: str
    source_url: NonEmptyStr


class DayPlan(_Model):
    """A numbered day of stops with optional weather context."""

    day_number: int = Field(ge=1, strict=True)
    theme: str
    stops: list[Stop]
    weather_note: str | None = None


class Itinerary(_Model):
    """The complete draft reviewed by the Critic and traveler."""

    city: NonEmptyStr
    days: list[DayPlan]
    budget_tips: list[str] = Field(min_length=2, max_length=3)
    warnings: list[str] = Field(default_factory=list)


class OrchestratorDecision(_Model):
    """Tool-free supervisor routing; exporting is never a routing choice."""

    next: NextAgent
    instructions: dict[str, str] = Field(default_factory=dict)
    reason: str


class Finding(_Model):
    """A hard or soft critique finding assigned to its corrective owner."""

    issue: str
    owner: Literal["researchers", "planner"]
    day: int | None = None


class Critique(_Model):
    """Findings about precisely the version recorded in critique_version."""

    hard_failures: list[Finding] = Field(default_factory=list)
    soft_findings: list[Finding] = Field(default_factory=list)


class AgentError(_Model):
    """A sanitized agent-boundary failure; never include provider secrets."""

    agent: str
    message: str


def merge_source_registry(
    current: dict[str, SourceRecord], incoming: dict[str, SourceRecord]
) -> dict[str, SourceRecord]:
    """Merge source IDs without mutation; native Overwrite(value={}) clears."""
    return {
        key: SourceRecord.model_validate(record)
        for key, record in (current | incoming).items()
    }


def preserve_draft_version(current: int, incoming: int) -> int:
    """Keep the largest draft version; a normal update can never decrease it."""
    if incoming < 0:
        raise ValueError("draft_version must be nonnegative")
    return max(current, incoming)


class DelegationEntry(TypedDict):
    """One append-only supervisor decision indexed by trip step."""

    step: int
    next: NextAgent
    reason: str


class TripState(TypedDict, total=False):
    """Partial graph updates; each checkpoint belongs to exactly one trip thread."""

    user_input: str
    original_request: TripRequest | None
    request: TripRequest | None
    next: NextAgent | None
    instructions: dict[str, str]
    # Append reducer: [] does not clear; return Overwrite(value=[]) to reset.
    delegation_log: Annotated[list[DelegationEntry], operator.add]
    dispatched: list[str]
    research_round: int
    researcher_attempts: dict[str, int]
    sights_result: ResearchResult | None
    food_result: ResearchResult | None
    places: list[Place]
    # Merge reducer: {} does not clear; return Overwrite(value={}) to reset.
    source_registry: Annotated[dict[str, SourceRecord], merge_source_registry]
    weather: dict[str, Any] | None
    itinerary: Itinerary | None
    # Max reducer: Overwrite bypasses it; never use Overwrite here within a trip.
    # Version resets belong to a new thread, not a revise/accumulator reset.
    draft_version: Annotated[int, preserve_draft_version]
    critique: Critique | None
    critique_version: int | None
    approved_version: int | None
    round_steps: int
    trip_steps: int
    correction_cycles: int
    agent_error: AgentError | None
    human_feedback: str | None
    decision: HumanDecision | None
    # Append reducer: [] does not clear; return Overwrite(value=[]) to reset.
    warnings: Annotated[list[str], operator.add]
    saved_path: str | None
    export_manifest: dict[str, Any] | None
