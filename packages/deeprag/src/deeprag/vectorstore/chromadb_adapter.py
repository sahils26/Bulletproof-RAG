"""ChromaDB adapter — concrete implementation of the VectorStore interface.

This is the first real database our RAG system talks to. ChromaDB is an
open-source, lightweight vector database that runs locally or via Docker.
It stores our chunk embeddings and lets us do similarity searches.
"""

import asyncio
from typing import Any
from uuid import UUID

import chromadb
from shared.models.documents import Chunk
from shared.models.retrieval import ScoredChunk

from deeprag.vectorstore.base import VectorStore


class ChromaDBAdapter(VectorStore):
    """Concrete VectorStore backed by ChromaDB.

    Why ChromaDB?
    - It's free and open-source
    - Runs locally (no cloud account needed)
    - Has a simple Python API
    - Supports metadata filtering out of the box

    This adapter translates our clean VectorStore interface into
    ChromaDB-specific API calls. If we ever want to switch to Qdrant
    or Pinecone, we just write a new adapter — nothing else changes.
    """

    def __init__(self, host: str | None = None, port: int | None = None) -> None:
        """Connect to a running ChromaDB server.

        If host/port are not provided, it looks for the VECTOR_URL environment
        variable (e.g., http://localhost:8001).
        """
        import os
        from urllib.parse import urlparse

        # 1. Start with defaults
        final_host = host or "localhost"
        final_port = port or 8001

        # 2. Override with Environment Variable if no args provided
        if host is None and port is None:
            vector_url = os.getenv("VECTOR_URL")
            if vector_url:
                try:
                    parsed = urlparse(vector_url)
                    if parsed.hostname:
                        final_host = parsed.hostname
                    if parsed.port:
                        final_port = parsed.port
                except Exception:
                    pass

        self._client = chromadb.HttpClient(host=final_host, port=final_port)

    def _get_or_create_collection(self, name: str) -> chromadb.Collection:
        """Get existing collection or create a new one."""
        return self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    async def upsert(self, chunks: list[Chunk], collection: str) -> int:
        """Insert or update chunks into ChromaDB.

        Each chunk must already have its ``embedding`` field populated
        (the ingestion pipeline handles this before calling upsert).

        We store metadata alongside each vector so we can filter later
        (e.g., "show me only chunks from document X").
        """
        if not chunks:
            return 0

        col = self._get_or_create_collection(collection)

        ids = [str(c.id) for c in chunks]
        documents = [c.content for c in chunks]
        embeddings = [c.embedding for c in chunks if c.embedding is not None]

        if len(embeddings) != len(chunks):
            raise ValueError(
                f"All chunks must have embeddings. "
                f"Got {len(embeddings)} embeddings for {len(chunks)} chunks."
            )

        metadatas = []
        for c in chunks:
            meta: dict[str, str | int | float] = {
                "document_id": str(c.document_id),
                "chunk_index": c.chunk_index,
                "token_count": c.token_count,
            }
            if "chunk_strategy" in c.metadata:
                meta["chunk_strategy"] = c.metadata["chunk_strategy"]
            if "source" in c.metadata:
                meta["source"] = c.metadata["source"]
            metadatas.append(meta)

        # ChromaDB's API is synchronous, so we offload to a thread
        await asyncio.to_thread(
            col.upsert,
            ids=ids,
            documents=documents,
            embeddings=embeddings,  # type: ignore
            metadatas=metadatas,  # type: ignore
        )

        return len(chunks)

    async def query(
        self,
        embedding: list[float],
        collection: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """Search ChromaDB for chunks most similar to the query embedding.

        ChromaDB uses cosine similarity by default (we configured this
        in _get_or_create_collection). The ``filters`` parameter maps
        directly to ChromaDB's ``where`` clause for metadata filtering.
        """
        col = self._get_or_create_collection(collection)

        query_params: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }

        if filters:
            query_params["where"] = filters

        results = await asyncio.to_thread(col.query, **query_params)

        scored_chunks: list[ScoredChunk] = []

        if not results or not results.get("ids") or not results["ids"][0]:
            return scored_chunks

        for i, chunk_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] if results.get("metadatas") else {}  # type: ignore
            doc_text = results["documents"][0][i] if results.get("documents") else ""  # type: ignore
            distance = results["distances"][0][i] if results.get("distances") else 0.0  # type: ignore

            # ChromaDB returns cosine distance (0 = identical, 2 = opposite).
            # We convert to a similarity score (1 = identical, -1 = opposite).
            similarity = 1.0 - float(distance)

            # Safely parse metadata fields that might be mixed types according to Mypy
            doc_id_val = str(
                meta.get("document_id", "00000000-0000-0000-0000-000000000000")
            )
            chunk_idx_val = int(str(meta.get("chunk_index", 0)))
            token_count_val = int(str(meta.get("token_count", 0)))

            chunk = Chunk(
                id=UUID(str(chunk_id)),
                content=str(doc_text),
                document_id=UUID(doc_id_val),
                chunk_index=chunk_idx_val,
                token_count=token_count_val,
                metadata=dict(meta) if meta else {},  # type: ignore
            )

            scored_chunks.append(
                ScoredChunk(
                    chunk=chunk,
                    retrieval_score=similarity,
                )
            )

        return scored_chunks

    async def delete(self, chunk_ids: list[UUID], collection: str) -> int:
        """Delete chunks from ChromaDB by their IDs."""
        if not chunk_ids:
            return 0

        col = self._get_or_create_collection(collection)
        ids_str = [str(cid) for cid in chunk_ids]

        await asyncio.to_thread(col.delete, ids=ids_str)
        return len(chunk_ids)

    async def list_collections(self) -> list[str]:
        """List all collection names in ChromaDB."""
        collections = await asyncio.to_thread(self._client.list_collections)
        return [c.name for c in collections]

    async def collection_stats(self, collection: str) -> dict[str, Any]:
        """Get statistics about a ChromaDB collection.

        Returns chunk_count and unique document_count.
        This feeds the dashboard and MCP status tools.
        """
        col = self._get_or_create_collection(collection)
        count = await asyncio.to_thread(col.count)

        # Get unique document IDs to count source documents
        doc_ids: set[str] = set()
        if count > 0:
            all_meta = await asyncio.to_thread(col.get, include=["metadatas"])
            if all_meta and all_meta.get("metadatas"):
                for meta in all_meta["metadatas"]:  # type: ignore
                    if meta and "document_id" in meta:
                        doc_ids.add(str(meta["document_id"]))

        return {
            "chunk_count": count,
            "document_count": len(doc_ids),
            "collection_name": collection,
        }
