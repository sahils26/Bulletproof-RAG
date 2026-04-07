"""Tests for shared models — serialization, validation, enums, and defaults."""

import json
from datetime import datetime
from uuid import UUID

import pytest
from pydantic import ValidationError
from shared.models import (
    Budget,
    Chunk,
    CircuitState,
    Citation,
    Document,
    GenerationResult,
    ProgressEvent,
    ProgressEventType,
    Query,
    QueryType,
    ResponseType,
    RetrievalResult,
    SpanStatus,
    TraceSpan,
)

# ── Document & Chunk ────────────────────────────────────────────


class TestDocument:
    def test_auto_id_and_timestamp(self):
        doc = Document(content="Hello world")
        assert isinstance(doc.id, UUID)
        assert isinstance(doc.created_at, datetime)

    def test_json_round_trip(self):
        doc = Document(content="Test", source_path="/tmp/test.txt")
        data = json.loads(doc.model_dump_json())
        doc2 = Document(**data)
        assert doc2.content == "Test"
        assert doc2.source_path == "/tmp/test.txt"

    def test_metadata_defaults_to_empty_dict(self):
        doc = Document(content="x")
        assert doc.metadata == {}


class TestChunk:
    def test_requires_document_id(self):
        doc = Document(content="parent")
        chunk = Chunk(content="slice", document_id=doc.id, chunk_index=0)
        assert chunk.document_id == doc.id

    def test_token_count_default(self):
        doc = Document(content="parent")
        chunk = Chunk(content="text", document_id=doc.id, chunk_index=0)
        assert chunk.token_count == 0  # default before counting

    def test_embedding_default_none(self):
        doc = Document(content="parent")
        chunk = Chunk(content="text", document_id=doc.id, chunk_index=0)
        assert chunk.embedding is None


# ── Retrieval ───────────────────────────────────────────────────


class TestQueryType:
    def test_enum_serialization(self):
        q = Query(text="What is X?", query_type=QueryType.FACTUAL)
        data = q.model_dump()
        assert data["query_type"] == "factual"

    def test_enum_deserialization(self):
        q = Query(text="Compare A and B", query_type="comparison")
        assert q.query_type == QueryType.COMPARISON

    def test_default_is_unknown(self):
        q = Query(text="Hello")
        assert q.query_type == QueryType.UNKNOWN


class TestRetrievalResult:
    def test_defaults(self):
        r = RetrievalResult()
        assert r.chunks == []
        assert r.strategy_used == "unknown"
        assert r.iteration == 1


# ── Generation ──────────────────────────────────────────────────


class TestResponseType:
    def test_abstention_enum(self):
        result = GenerationResult(
            query="mock question",
            answer="I don't know.",
            response_type=ResponseType.ABSTAINED,
            confidence=0.1,
        )
        data = result.model_dump()
        assert data["response_type"] == "abstained"

    def test_citation_round_trip(self):
        doc = Document(content="x")
        chunk = Chunk(content="y", document_id=doc.id, chunk_index=0)
        citation = Citation(chunk_id=chunk.id, text_span="y", relevance_score=4.5)
        data = json.loads(citation.model_dump_json())
        c2 = Citation(**data)
        assert c2.text_span == "y"


# ── Tracing ─────────────────────────────────────────────────────


class TestTracing:
    def test_span_defaults(self):
        span = TraceSpan(operation="retrieve")
        assert span.status == SpanStatus.OK
        assert span.token_count == 0

    def test_budget_defaults(self):
        b = Budget()
        assert b.max_wall_time_seconds == 30.0
        assert b.max_total_tokens == 10_000
        assert b.max_iterations == 3

    def test_circuit_state_enum(self):
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.HALF_OPEN.value == "half_open"


# ── Events ──────────────────────────────────────────────────────


class TestEvents:
    def test_event_creation(self):
        evt = ProgressEvent(
            event_type=ProgressEventType.RETRIEVAL_STARTED,
            message="Starting retrieval",
        )
        assert evt.event_type == ProgressEventType.RETRIEVAL_STARTED
        assert isinstance(evt.timestamp, datetime)

    def test_event_json_round_trip(self):
        evt = ProgressEvent(
            event_type=ProgressEventType.COMPLETE,
            message="Done",
            completion_pct=1.0,
        )
        data = json.loads(evt.model_dump_json())
        evt2 = ProgressEvent(**data)
        assert evt2.completion_pct == 1.0


# ── Validation errors ──────────────────────────────────────────


class TestValidation:
    def test_document_requires_content(self):
        with pytest.raises(ValidationError):
            Document()  # type: ignore[call-arg]

    def test_chunk_requires_document_id(self):
        with pytest.raises(ValidationError):
            Chunk(content="x", chunk_index=0)  # type: ignore[call-arg]
