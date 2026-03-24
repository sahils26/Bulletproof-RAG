"""Generation and response models — including the abstention protocol."""

from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ResponseType(str, Enum):
    """Outcome classification for a generated response.

    This is the abstention protocol — a first-class design decision defined
    from Day 3 so every consumer (generator, MCP server, eval harness) uses
    the same vocabulary.

    - CONFIDENT: the system found strong evidence and generated a full answer.
    - PARTIAL: some evidence was found but the answer may be incomplete.
    - ABSTAINED: the system could not find sufficient evidence and chose NOT
      to hallucinate.
    """

    CONFIDENT = "confident"
    PARTIAL = "partial"
    ABSTAINED = "abstained"


class Citation(BaseModel):
    """A reference linking an answer back to a specific source chunk."""

    chunk_id: UUID = Field(description="ID of the chunk this citation points to.")
    text_span: str = Field(description="The quoted text from the chunk used in the answer.")
    relevance_score: float = Field(
        default=0.0, description="How relevant this citation was (1–5 scale)."
    )


class GenerationResult(BaseModel):
    """The final answer returned to the user, with citations and confidence."""

    answer: str = Field(description="The generated response text.")
    response_type: ResponseType = Field(
        default=ResponseType.CONFIDENT,
        description="Whether the system is confident, partial, or abstained.",
    )
    confidence: float = Field(
        default=0.0, description="Confidence score from 0.0 to 1.0."
    )
    citations: list[Citation] = Field(
        default_factory=list, description="Source citations backing the answer."
    )
    hallucination_flags: list[str] = Field(
        default_factory=list,
        description="Statements flagged as potential hallucinations by the checker.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Execution metadata (latency, tokens, cost)."
    )
