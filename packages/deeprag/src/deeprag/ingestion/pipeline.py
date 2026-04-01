"""Ingestion pipeline — wires Loaders → Chunkers → Embeddings → VectorStore.

This is the orchestrator that takes a folder of documents and makes them
searchable. It is the first "end-to-end" feature of the RAG system.

Flow:
1. Walk the source directory and discover files
2. Load each file using the LoaderRegistry (auto-picks the right loader)
3. Chunk each document using the chosen chunking strategy
4. Embed all chunks using the EmbeddingService
5. Upsert the embedded chunks into the VectorStore (ChromaDB)
"""

import os
import time
from dataclasses import dataclass, field

from shared.models.documents import Chunk, Document
from shared.models.events import (
    ProgressCallback,
    ProgressEvent,
    ProgressEventType,
)

from deeprag.chunkers import (
    Chunker,
    RecursiveCharacterChunker,
    SemanticParagraphChunker,
    SlidingWindowChunker,
)
from deeprag.embeddings.service import EmbeddingService
from deeprag.loaders import LoaderRegistry, UnsupportedFileTypeError, default_registry
from deeprag.vectorstore.base import VectorStore


@dataclass
class IngestionResult:
    """Summary of what the ingestion pipeline accomplished."""

    document_count: int = 0
    chunk_count: int = 0
    duration_ms: float = 0.0
    failed_files: list[str] = field(default_factory=list)


def _make_chunker(
    strategy: str,
    chunk_size: int,
    overlap: int,
) -> Chunker:
    """Create the right chunker based on the strategy name.

    This is a simple factory function — it reads the strategy string
    and returns the matching Chunker instance.
    """
    if strategy == "recursive":
        return RecursiveCharacterChunker(chunk_size=chunk_size, overlap=overlap)
    elif strategy == "sliding_window":
        return SlidingWindowChunker(chunk_size=chunk_size, overlap=overlap)
    elif strategy == "semantic":
        return SemanticParagraphChunker(max_length=chunk_size)
    else:
        raise ValueError(
            f"Unknown chunk strategy: '{strategy}'. "
            f"Choose from: recursive, sliding_window, semantic"
        )


def _discover_files(source_dir: str) -> list[str]:
    """Walk a directory and return all file paths."""
    files: list[str] = []
    for root, _dirs, filenames in os.walk(source_dir):
        for fname in sorted(filenames):
            # Skip hidden files and common non-document files
            if fname.startswith("."):
                continue
            files.append(os.path.join(root, fname))
    return files


async def run_ingestion(
    source_dir: str,
    collection: str,
    vector_store: VectorStore,
    embedding_service: EmbeddingService | None = None,
    registry: LoaderRegistry | None = None,
    chunk_strategy: str = "recursive",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    callback: ProgressCallback | None = None,
) -> IngestionResult:
    """Run the full ingestion pipeline end-to-end.

    Args:
        source_dir: Path to directory containing documents.
        collection: Name of the vector store collection.
        vector_store: The VectorStore to upsert into.
        embedding_service: The embedding service (creates default if None).
        registry: The loader registry (creates default if None).
        chunk_strategy: One of "recursive", "sliding_window", "semantic".
        chunk_size: Max characters per chunk.
        chunk_overlap: Character overlap between chunks.
        callback: Optional progress callback.

    Returns:
        An IngestionResult with counts and timing.
    """
    start_time = time.time()
    result = IngestionResult()

    # ── Step 0: Setup ───────────────────────────────────────
    if embedding_service is None:
        embedding_service = EmbeddingService()
    if registry is None:
        registry = default_registry()
    chunker = _make_chunker(chunk_strategy, chunk_size, chunk_overlap)

    # ── Step 1: Discover files ──────────────────────────────
    file_paths = _discover_files(source_dir)

    if callback:
        await callback(
            ProgressEvent(
                event_type=ProgressEventType.QUERY_RECEIVED,
                message=f"Found {len(file_paths)} files in {source_dir}",
            )
        )

    # ── Step 2: Load documents ──────────────────────────────
    all_documents: list[Document] = []
    for fp in file_paths:
        try:
            docs = registry.load(fp)
            all_documents.extend(docs)
        except (UnsupportedFileTypeError, FileNotFoundError) as e:
            result.failed_files.append(f"{fp}: {e}")
            continue

    result.document_count = len(all_documents)

    if callback:
        await callback(
            ProgressEvent(
                event_type=ProgressEventType.RETRIEVAL_STARTED,
                message=f"Loaded {len(all_documents)} documents",
            )
        )

    if not all_documents:
        result.duration_ms = (time.time() - start_time) * 1000
        return result

    # ── Step 3: Chunk documents ─────────────────────────────
    all_chunks: list[Chunk] = []
    for doc in all_documents:
        chunks = chunker.chunk(doc)
        all_chunks.extend(chunks)

    if callback:
        await callback(
            ProgressEvent(
                event_type=ProgressEventType.RETRIEVAL_COMPLETE,
                message=f"Created {len(all_chunks)} chunks",
            )
        )

    if not all_chunks:
        result.duration_ms = (time.time() - start_time) * 1000
        return result

    # ── Step 4: Embed all chunks ────────────────────────────
    texts = [c.content for c in all_chunks]
    embeddings = await embedding_service.embed(texts, callback=callback)

    # Attach each embedding to its chunk
    for chunk, emb in zip(all_chunks, embeddings, strict=True):
        chunk.embedding = emb

    # ── Step 5: Upsert into vector store ────────────────────
    upserted = await vector_store.upsert(all_chunks, collection)
    result.chunk_count = upserted

    if callback:
        await callback(
            ProgressEvent(
                event_type=ProgressEventType.COMPLETE,
                message=(
                    f"Ingestion complete: {result.document_count} docs, "
                    f"{result.chunk_count} chunks"
                ),
                completion_pct=1.0,
            )
        )

    result.duration_ms = (time.time() - start_time) * 1000
    return result
