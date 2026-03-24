"""Shared models — re-export everything for clean imports.

Usage::

    from shared.models import Document, Chunk, Query, GenerationResult
"""

from shared.models.documents import Chunk, Document
from shared.models.events import ProgressCallback, ProgressEvent, ProgressEventType
from shared.models.generation import Citation, GenerationResult, ResponseType
from shared.models.retrieval import Query, QueryType, RetrievalResult, ScoredChunk
from shared.models.tracing import Budget, CircuitState, SpanStatus, TraceSpan

__all__ = [
    # Documents
    "Document",
    "Chunk",
    # Retrieval
    "Query",
    "QueryType",
    "ScoredChunk",
    "RetrievalResult",
    # Generation
    "ResponseType",
    "Citation",
    "GenerationResult",
    # Tracing
    "SpanStatus",
    "TraceSpan",
    "Budget",
    "CircuitState",
    # Events
    "ProgressEventType",
    "ProgressEvent",
    "ProgressCallback",
]
