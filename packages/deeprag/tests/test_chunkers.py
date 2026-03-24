"""Tests for chunkers — including token_count and chunk_strategy metadata."""

import pytest
from shared.models import Document
from deeprag.chunkers import (
    SlidingWindowChunker, RecursiveCharacterChunker, SemanticParagraphChunker,
)


@pytest.fixture
def empty_doc():
    return Document(content="", metadata={"author": "AI"})


@pytest.fixture
def single_line_doc():
    return Document(content="This is a very simple single line document.", metadata={"author": "AI"})


@pytest.fixture
def large_doc():
    text = "Paragraph 1 is here.\n\nParagraph 2 is right here.\n\nParagraph 3 is also here.\n\nParagraph 4 continues.\n\nParagraph 5 finishes up."
    return Document(content=text, metadata={"author": "AI", "source": "test.txt"})


def test_empty_doc(empty_doc):
    assert len(SlidingWindowChunker().chunk(empty_doc)) == 0
    assert len(RecursiveCharacterChunker().chunk(empty_doc)) == 0
    assert len(SemanticParagraphChunker().chunk(empty_doc)) == 0


def test_metadata_propagation(single_line_doc):
    chunks = SlidingWindowChunker(chunk_size=50, overlap=0).chunk(single_line_doc)
    assert len(chunks) == 1
    assert chunks[0].metadata["author"] == "AI"
    assert chunks[0].chunk_index == 0
    assert chunks[0].metadata["chunk_strategy"] == "sliding_window"


def test_token_count_metadata(single_line_doc):
    chunks = SlidingWindowChunker(chunk_size=100, overlap=0).chunk(single_line_doc)
    assert len(chunks) == 1
    assert chunks[0].token_count > 0


def test_sliding_window_overlap(large_doc):
    chunks = SlidingWindowChunker(chunk_size=40, overlap=10).chunk(large_doc)
    assert len(chunks) > 1
    overlap_segment = chunks[0].content[-10:]
    assert chunks[1].content.startswith(overlap_segment)


def test_recursive_character_chunker(large_doc):
    chunks = RecursiveCharacterChunker(chunk_size=30, overlap=5).chunk(large_doc)
    assert len(chunks) >= 5
    for c in chunks:
        assert len(c.content) <= 35
        assert c.metadata["chunk_strategy"] == "recursive_character"


def test_semantic_paragraph_chunker(large_doc):
    chunks = SemanticParagraphChunker().chunk(large_doc)
    assert len(chunks) == 5
    assert chunks[0].content == "Paragraph 1 is here."
    assert chunks[4].content == "Paragraph 5 finishes up."
    for c in chunks:
        assert c.metadata["chunk_strategy"] == "semantic_paragraph"
