"""Retrieval-related models — queries, scored chunks, and retrieval results."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from shared.models.documents import Chunk


class QueryType(StrEnum):
    """Classification of a user query to determine retrieval strategy.

    The QueryAnalyzer (Day 11) assigns one of these types to every incoming
    question. The StrategyRouter uses it to pick dense, sparse, or hybrid
    retrieval.
    """

    FACTUAL = "factual"
    MULTI_HOP = "multi_hop"
    COMPARISON = "comparison"
    SUMMARIZATION = "summarization"
    OUT_OF_DOMAIN = "out_of_domain"
    UNKNOWN = "unknown"


class Query(BaseModel):
    """A user question enriched with classification metadata."""

    text: str = Field(description="The raw text of the user's question.")
    query_type: QueryType = Field(
        default=QueryType.UNKNOWN,
        description="Classified type of the query (set by QueryAnalyzer).",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata (e.g. user_id, session_id).",
    )


class ScoredChunk(BaseModel):
    """A Chunk paired with its retrieval scores."""

    chunk: Chunk = Field(description="The retrieved chunk.")
    relevance_score: float = Field(
        default=0.0, description="Relevance score assigned by the grader (1–5 scale)."
    )
    retrieval_score: float = Field(
        default=0.0, description="Raw similarity score from the vector store."
    )


class RetrievalResult(BaseModel):
    """The outcome of a single retrieval operation."""

    chunks: list[ScoredChunk] = Field(
        default_factory=list, description="Ranked list of scored chunks."
    )
    strategy_used: str = Field(
        default="unknown",
        description="Which retrieval strategy produced these results.",
    )
    latency_ms: float = Field(
        default=0.0, description="How long the retrieval took (ms)."
    )
    iteration: int = Field(
        default=1, description="Which self-correction iteration this result belongs to."
    )
