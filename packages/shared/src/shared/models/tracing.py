"""Tracing and budget models — observability and resource management."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class SpanStatus(StrEnum):
    """Status of a completed trace span."""

    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    BUDGET_EXCEEDED = "budget_exceeded"


class CircuitState(StrEnum):
    """Circuit breaker state for protecting external resources.

    - CLOSED: everything is healthy, requests flow normally.
    - OPEN: failures exceeded threshold, requests are blocked.
    - HALF_OPEN: trial period — one request allowed to test recovery.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class TraceSpan(BaseModel):
    """One discrete operation in the execution trace (e.g. 'retrieve', 'grade').

    Spans form a tree via parent_id, enabling full execution tracing from
    query to final answer.
    """

    span_id: UUID = Field(default_factory=uuid4, description="Unique span identifier.")
    parent_id: UUID | None = Field(
        default=None, description="Parent span ID (None for root spans)."
    )
    operation: str = Field(
        description="Name of the operation (e.g. 'retrieve', 'generate')."
    )
    start_time: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When this span started.",
    )
    duration_ms: float | None = Field(
        default=None, description="Duration in milliseconds (set on completion)."
    )
    status: SpanStatus = Field(
        default=SpanStatus.OK, description="Outcome of the operation."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Operation-specific metadata."
    )
    token_count: int = Field(
        default=0, description="Tokens consumed by this operation."
    )
    cost_usd: float = Field(default=0.0, description="Estimated cost in USD.")


class Budget(BaseModel):
    """Execution limits for a single query — prevents runaway costs."""

    max_wall_time_seconds: float = Field(
        default=30.0, description="Maximum real-world seconds allowed."
    )
    max_total_tokens: int = Field(
        default=10_000, description="Maximum tokens that may be consumed."
    )
    max_llm_calls: int = Field(
        default=10, description="Maximum number of LLM API requests."
    )
    max_iterations: int = Field(
        default=3, description="Maximum self-correction loops permitted."
    )
