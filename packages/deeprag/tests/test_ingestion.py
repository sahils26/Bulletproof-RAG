"""Tests for the ingestion pipeline (unit tests — no ChromaDB required).

These tests use a mock VectorStore so they run without Docker.
Integration tests with real ChromaDB are in test_chromadb_integration.py.
"""

import os
import tempfile
from unittest.mock import AsyncMock

import pytest
from deeprag.chunkers import (
    RecursiveCharacterChunker,
    SemanticParagraphChunker,
    SlidingWindowChunker,
)
from deeprag.ingestion.pipeline import (
    IngestionResult,
    _discover_files,
    _make_chunker,
    run_ingestion,
)
from shared.models.documents import Chunk


class TestMakeChunker:
    def test_recursive(self):
        c = _make_chunker("recursive", 500, 50)
        assert isinstance(c, RecursiveCharacterChunker)

    def test_sliding_window(self):
        c = _make_chunker("sliding_window", 500, 50)
        assert isinstance(c, SlidingWindowChunker)

    def test_semantic(self):
        c = _make_chunker("semantic", 500, 50)
        assert isinstance(c, SemanticParagraphChunker)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown chunk strategy"):
            _make_chunker("magic", 500, 50)


class TestDiscoverFiles:
    def test_finds_files_skips_hidden(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some files
            open(os.path.join(tmpdir, "doc.txt"), "w").close()
            open(os.path.join(tmpdir, "notes.md"), "w").close()
            open(os.path.join(tmpdir, ".hidden"), "w").close()

            files = _discover_files(tmpdir)
            basenames = [os.path.basename(f) for f in files]

            assert "doc.txt" in basenames
            assert "notes.md" in basenames
            assert ".hidden" not in basenames


@pytest.mark.asyncio
async def test_ingestion_with_mock_vectorstore():
    """Test the full pipeline with a mock vector store (no Docker needed)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a sample text file
        sample_path = os.path.join(tmpdir, "sample.txt")
        with open(sample_path, "w") as f:
            f.write("This is a test document for ingestion. " * 20)

        # Create a mock VectorStore
        mock_store = AsyncMock()
        mock_store.upsert = AsyncMock(return_value=5)

        result = await run_ingestion(
            source_dir=tmpdir,
            collection="test",
            vector_store=mock_store,
            chunk_strategy="recursive",
            chunk_size=200,
            chunk_overlap=20,
        )

        assert isinstance(result, IngestionResult)
        assert result.document_count == 1
        assert result.chunk_count == 5
        assert result.duration_ms > 0

        # Verify upsert was called with chunks that have embeddings
        mock_store.upsert.assert_called_once()
        args = mock_store.upsert.call_args
        chunks_arg = args[0][0]  # first positional arg
        assert all(isinstance(c, Chunk) for c in chunks_arg)
        assert all(c.embedding is not None for c in chunks_arg)


@pytest.mark.asyncio
async def test_ingestion_empty_directory():
    """An empty directory should return zero counts, not crash."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_store = AsyncMock()

        result = await run_ingestion(
            source_dir=tmpdir,
            collection="test",
            vector_store=mock_store,
        )

        assert result.document_count == 0
        assert result.chunk_count == 0
        mock_store.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_ingestion_skips_unsupported_files():
    """Files with unknown extensions should be skipped, not crash."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create an unsupported file type
        open(os.path.join(tmpdir, "image.png"), "w").close()
        # And a supported one
        with open(os.path.join(tmpdir, "doc.txt"), "w") as f:
            f.write("Hello world " * 50)

        mock_store = AsyncMock()
        mock_store.upsert = AsyncMock(return_value=1)

        result = await run_ingestion(
            source_dir=tmpdir,
            collection="test",
            vector_store=mock_store,
        )

        assert result.document_count == 1
        assert len(result.failed_files) == 1
        assert "image.png" in result.failed_files[0]
