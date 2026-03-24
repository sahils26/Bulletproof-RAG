"""Document and Chunk models — the core data structures for RAG ingestion."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Represents an entire loaded document before chunking.

    Every file that enters the system (PDF, Markdown, DOCX, etc.) is first
    converted into a Document. The ``content`` field holds the raw extracted
    text, while ``metadata`` carries loader-specific information such as
    headers, page counts, or file types.
    """

    id: UUID = Field(default_factory=uuid4, description="Unique document identifier.")
    content: str = Field(description="Full raw text extracted from the source file.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Loader-specific metadata (e.g. headers, page count, file_type).",
    )
    source_path: str | None = Field(
        default=None, description="Filesystem path or URI the document was loaded from."
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when the document was created.",
    )


class Chunk(BaseModel):
    """A retrieval-optimized slice of a Document.

    Chunkers split a Document into many Chunks. Each Chunk records which
    document it came from (``document_id``), its position (``chunk_index``),
    and an accurate ``token_count`` for budget tracking.
    """

    id: UUID = Field(default_factory=uuid4, description="Unique chunk identifier.")
    content: str = Field(description="The text content of this chunk.")
    document_id: UUID = Field(
        description="ID of the parent Document this chunk belongs to."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Inherited document metadata plus chunk-specific info.",
    )
    chunk_index: int = Field(
        description="Position of this chunk within the parent document."
    )
    token_count: int = Field(
        default=0, description="Accurate token count (via tiktoken)."
    )
    embedding: list[float] | None = Field(
        default=None, description="Vector embedding (populated after embedding step)."
    )
