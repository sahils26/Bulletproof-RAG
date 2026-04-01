"""Integration tests for ChromaDB adapter (requires docker-compose up chromadb).

These tests actually talk to a real ChromaDB instance. To run them:
    docker-compose up -d chromadb
    uv run pytest packages/deeprag/tests/test_chromadb_integration.py -v

They are marked with @pytest.mark.integration so they can be skipped
in CI if ChromaDB is not available.
"""

from uuid import uuid4

import pytest
from deeprag.vectorstore.chromadb_adapter import ChromaDBAdapter
from shared.models.documents import Chunk

# A unique collection name per test run to avoid collisions
TEST_COLLECTION = f"test_integration_{uuid4().hex[:8]}"


def _make_chunk(
    text: str,
    embedding: list[float],
    doc_id=None,
    index: int = 0,
) -> Chunk:
    """Helper to create a Chunk with an embedding."""
    return Chunk(
        content=text,
        document_id=doc_id or uuid4(),
        chunk_index=index,
        token_count=len(text.split()),
        embedding=embedding,
        metadata={"chunk_strategy": "test", "source": "test.txt"},
    )


@pytest.fixture
def adapter():
    """Create a ChromaDB adapter pointing to the local Docker instance."""
    return ChromaDBAdapter(host="localhost", port=8000)


@pytest.fixture
def fake_embedding():
    """A simple 384-dim embedding for testing (matching MiniLM dimensions)."""
    return [0.1] * 384


@pytest.mark.integration
@pytest.mark.asyncio
async def test_upsert_and_query(adapter, fake_embedding):
    """Upsert chunks and verify they can be queried back."""
    doc_id = uuid4()
    chunks = [
        _make_chunk(f"Test chunk {i}", fake_embedding, doc_id, i) for i in range(5)
    ]

    count = await adapter.upsert(chunks, TEST_COLLECTION)
    assert count == 5

    # Query with the same embedding — should return our chunks
    results = await adapter.query(
        embedding=fake_embedding,
        collection=TEST_COLLECTION,
        top_k=3,
    )
    assert len(results) == 3
    assert all(r.retrieval_score > 0 for r in results)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_metadata_filtering(adapter, fake_embedding):
    """Test that ChromaDB where-clause filtering works."""
    doc_id_1 = uuid4()
    doc_id_2 = uuid4()

    chunks = [
        _make_chunk("Doc1 chunk", fake_embedding, doc_id_1, 0),
        _make_chunk("Doc2 chunk", fake_embedding, doc_id_2, 0),
    ]

    col_name = f"filter_test_{uuid4().hex[:8]}"
    await adapter.upsert(chunks, col_name)

    # Filter by document_id
    results = await adapter.query(
        embedding=fake_embedding,
        collection=col_name,
        top_k=10,
        filters={"document_id": str(doc_id_1)},
    )
    assert len(results) == 1
    assert results[0].chunk.content == "Doc1 chunk"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete(adapter, fake_embedding):
    """Test deleting chunks from ChromaDB."""
    col_name = f"delete_test_{uuid4().hex[:8]}"
    chunk = _make_chunk("To be deleted", fake_embedding)

    await adapter.upsert([chunk], col_name)
    stats_before = await adapter.collection_stats(col_name)
    assert stats_before["chunk_count"] == 1

    deleted = await adapter.delete([chunk.id], col_name)
    assert deleted == 1

    stats_after = await adapter.collection_stats(col_name)
    assert stats_after["chunk_count"] == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_collection_stats(adapter, fake_embedding):
    """Test that collection_stats returns correct counts."""
    col_name = f"stats_test_{uuid4().hex[:8]}"
    doc_id = uuid4()

    chunks = [_make_chunk(f"Chunk {i}", fake_embedding, doc_id, i) for i in range(3)]
    await adapter.upsert(chunks, col_name)

    stats = await adapter.collection_stats(col_name)
    assert stats["chunk_count"] == 3
    assert stats["document_count"] == 1
    assert stats["collection_name"] == col_name


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_collections(adapter, fake_embedding):
    """Test that our collection shows up in list_collections."""
    col_name = f"list_test_{uuid4().hex[:8]}"
    chunk = _make_chunk("Hello", fake_embedding)
    await adapter.upsert([chunk], col_name)

    collections = await adapter.list_collections()
    assert col_name in collections
