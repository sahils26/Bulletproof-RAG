"""Chunking strategies: SlidingWindow, RecursiveCharacter, SemanticParagraph."""

import re
from abc import ABC, abstractmethod

from shared.models import Chunk, Document


def _safe_count_tokens(text: str) -> int:
    """Count tokens using tiktoken. Falls back to word-based estimate if unavailable."""
    try:
        from shared.utils.tokens import count_tokens

        return count_tokens(text)
    except ImportError:
        # Fallback: rough estimate of ~0.75 tokens per word
        return max(1, int(len(text.split()) * 1.33))


class Chunker(ABC):
    """Abstract interface for document chunking strategies."""

    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]:
        """Split a Document into a list of Chunks."""
        pass


class SlidingWindowChunker(Chunker):
    """Splits text into fixed-size overlapping blocks (windows)."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        if not doc.content.strip():
            return []

        chunks = []
        step = max(1, self.chunk_size - self.overlap)

        for index, i in enumerate(range(0, len(doc.content), step)):
            segment = doc.content[i : i + self.chunk_size].strip()
            if segment:
                metadata = doc.metadata.copy()
                metadata["start_char"] = i
                metadata["end_char"] = i + len(segment)
                metadata["chunk_strategy"] = "sliding_window"

                chunks.append(
                    Chunk(
                        content=segment,
                        document_id=doc.id,
                        chunk_index=index,
                        token_count=_safe_count_tokens(segment),
                        metadata=metadata,
                    )
                )
        return chunks


class RecursiveCharacterChunker(Chunker):
    """
    Recursively splits text using a hierarchy of separators
    (double newline → newline → space → character) to stay within chunk_size.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def chunk(self, doc: Document) -> list[Chunk]:
        if not doc.content.strip():
            return []

        text_segments = self._split_text(doc.content, self.separators)

        chunks = []
        current_chunk_text = ""
        index = 0

        for segment in text_segments:
            if (
                current_chunk_text
                and len(current_chunk_text) + len(segment) > self.chunk_size
            ):
                chunks.append(
                    self._create_chunk(doc, current_chunk_text.strip(), index)
                )
                index += 1
                overlap_text = (
                    current_chunk_text[-self.overlap :] if self.overlap > 0 else ""
                )
                current_chunk_text = overlap_text + segment
            else:
                current_chunk_text += segment

        if current_chunk_text.strip():
            chunks.append(self._create_chunk(doc, current_chunk_text.strip(), index))

        return chunks

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Split text on the first matching separator."""
        separator = separators[-1]
        for s in separators:
            if s == "":
                separator = s
                break
            if s in text:
                separator = s
                break

        if separator:
            splits = text.split(separator)
            final_chunks = [s + separator for s in splits[:-1]] + [splits[-1]]
        else:
            final_chunks = list(text)

        return [c for c in final_chunks if c]

    def _create_chunk(self, doc: Document, text: str, index: int) -> Chunk:
        metadata = doc.metadata.copy()
        metadata["chunk_strategy"] = "recursive_character"
        return Chunk(
            content=text,
            document_id=doc.id,
            chunk_index=index,
            token_count=_safe_count_tokens(text),
            metadata=metadata,
        )


class SemanticParagraphChunker(Chunker):
    """
    Splits text by natural paragraph boundaries (double newlines).
    Preserves semantic meaning over strict size limits.
    """

    def __init__(self, max_length: int = 2000):
        self.max_length = max_length

    def chunk(self, doc: Document) -> list[Chunk]:
        if not doc.content.strip():
            return []

        paragraphs = re.split(r"\n{2,}", doc.content.strip())

        chunks = []
        for index, para in enumerate(paragraphs):
            para = para.strip()
            if para:
                if len(para) > self.max_length:
                    para = para[: self.max_length]

                metadata = doc.metadata.copy()
                metadata["chunk_strategy"] = "semantic_paragraph"

                chunks.append(
                    Chunk(
                        content=para,
                        document_id=doc.id,
                        chunk_index=index,
                        token_count=_safe_count_tokens(para),
                        metadata=metadata,
                    )
                )
        return chunks
