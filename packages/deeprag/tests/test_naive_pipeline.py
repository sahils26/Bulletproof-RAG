"""Tests for the Naive RAG stringing together components."""

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from deeprag.pipeline.naive import NaiveRAGPipeline
from deeprag.retrieval.naive import NaiveRetriever
from shared.llm.service import LLMResponse
from shared.models.documents import Chunk
from shared.models.generation import GenerationResult, ResponseType
from shared.models.retrieval import RetrievalResult, ScoredChunk


@pytest.fixture
def mock_retriever():
    retriever = AsyncMock(spec=NaiveRetriever)

    # Setup mock chunks
    chunk1 = Chunk(
        id=uuid4(),
        content="The sky is blue.",
        document_id=uuid4(),
        chunk_index=0,
        token_count=4,
        embedding=[0.1, 0.2],
        metadata={"source": "doc1.txt"},
    )

    scored_chunk1 = ScoredChunk(chunk=chunk1, retrieval_score=0.9)
    retriever.retrieve.return_value = RetrievalResult(chunks=[scored_chunk1])
    return retriever


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.complete.return_value = LLMResponse(
        content="Based on the context, the sky is blue.",
        input_tokens=50,
        output_tokens=10,
        model_name="test-model",
    )
    return llm


@pytest.mark.asyncio
async def test_naive_pipeline_flow(mock_retriever, mock_llm):
    """Test the End-to-End flow of Naive RAG without grading."""
    pipeline = NaiveRAGPipeline(retriever=mock_retriever, llm=mock_llm)

    callback = AsyncMock()

    result = await pipeline.query(
        question="What color is the sky?",
        collection="test-kb",
        top_k=3,
        callback=callback,
    )

    # 1. Check Retrieval usage
    mock_retriever.retrieve.assert_called_once_with(
        query="What color is the sky?", collection="test-kb", top_k=3
    )

    # 2. Check LLM usage
    mock_llm.complete.assert_called_once()
    args, kwargs = mock_llm.complete.call_args
    messages = kwargs["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "The sky is blue" in messages[1]["content"]  # The injected context
    assert "What color is the sky?" in messages[1]["content"]  # The question

    # 3. Check Response
    assert isinstance(result, GenerationResult)
    assert result.answer == "Based on the context, the sky is blue."
    assert result.response_type == ResponseType.CONFIDENT
    assert result.citations[0].relevance_score == 0.9
    assert result.metadata["model"] == "test-model"

    # 4. Check Callbacks
    assert callback.call_count >= 3
