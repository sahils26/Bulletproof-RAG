"""Progress event protocol — the backbone for streaming, dashboard, and MCP.

Every pipeline component accepts an optional ``ProgressCallback`` and emits
``ProgressEvent`` objects at key milestones. The caller decides what to do
with them: log, stream to SSE, forward via MCP, or ignore. No tight coupling.
"""

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ProgressEventType(StrEnum):
    """Types of progress events emitted throughout the pipeline."""

    QUERY_RECEIVED = "query_received"
    RETRIEVAL_STARTED = "retrieval_started"
    RETRIEVAL_COMPLETE = "retrieval_complete"
    GRADING_STARTED = "grading_started"
    GRADING_COMPLETE = "grading_complete"
    REFORMULATING = "reformulating"
    GENERATING = "generating"
    HALLUCINATION_CHECK = "hallucination_check"
    COMPLETE = "complete"
    ABSTAINED = "abstained"
    ERROR = "error"
    BUDGET_WARNING = "budget_warning"


class ProgressEvent(BaseModel):
    """A single progress update emitted by a pipeline component."""

    event_type: ProgressEventType = Field(description="What just happened.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When this event was emitted.",
    )
    message: str = Field(default="", description="Human-readable status message.")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Event-specific structured data."
    )
    completion_pct: float | None = Field(
        default=None, description="Optional completion percentage (0.0–1.0)."
    )


# Type alias: any async function that accepts a ProgressEvent.
# Components accept this as an optional parameter for dependency injection.
ProgressCallback = Callable[[ProgressEvent], Awaitable[None]]
