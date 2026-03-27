"""Tests for the EmbeddingService."""

import pytest
from deeprag.embeddings.service import EmbeddingService
from shared.models.events import ProgressEventType


@pytest.mark.asyncio
async def test_embed_empty_list():
    service = EmbeddingService()
    result = await service.embed([])
    assert result == []


@pytest.mark.asyncio
async def test_embed_single_text():
    service = EmbeddingService(model_name="all-MiniLM-L6-v2")
    texts = ["This is a test document."]

    embeddings = await service.embed(texts)

    assert len(embeddings) == 1
    # all-MiniLM-L6-v2 produces 384-dimensional vectors
    assert len(embeddings[0]) == 384
    assert isinstance(embeddings[0][0], float)


@pytest.mark.asyncio
async def test_embed_batch_with_callback():
    service = EmbeddingService(batch_size=2)
    texts = ["Doc 1", "Doc 2", "Doc 3"]

    events = []

    async def mock_callback(event):
        events.append(event)

    embeddings = await service.embed(texts, callback=mock_callback)

    assert len(embeddings) == 3
    # 3 items with batch_size=2 means 2 batches total (2 items, then 1 item)
    assert len(events) == 2
    assert events[0].event_type == ProgressEventType.EMBEDDING_BATCH
    assert events[0].completion_pct == 0.5  # 1/2
    assert events[1].completion_pct == 1.0  # 2/2


@pytest.mark.asyncio
async def test_semantic_similarity():
    # A quick sanity check that the embeddings actually work semantically.
    service = EmbeddingService()

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast, dark-colored fox leaps over a resting canine.",
        "I need to buy some milk and eggs from the grocery store.",
    ]

    embeddings = await service.embed(texts)

    # Calculate simple dot product / cosine similarity (assuming normalized vectors)
    import math

    def cosine_sim(v1, v2):
        dot = sum(a * b for a, b in zip(v1, v2, strict=False))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        return dot / (norm1 * norm2)

    sim_1_2 = cosine_sim(embeddings[0], embeddings[1])
    sim_1_3 = cosine_sim(embeddings[0], embeddings[2])

    # The first two sentences are semantically similar. The third is unrelated.
    assert sim_1_2 > sim_1_3
