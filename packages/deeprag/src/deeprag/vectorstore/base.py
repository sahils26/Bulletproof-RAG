"""Abstract interface for Vector Stores in the DeepRAG pipeline."""

from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID

from shared.models.documents import Chunk
from shared.models.retrieval import ScoredChunk


class VectorStore(ABC):
    """Abstract base class for all vector database implementations.

    This interface ensures that whether we use ChromaDB, Qdrant, Pinecone, or PGVector,
    the rest of our RAG pipeline doesn't need to change its code. It just talks to
    this interface.
    """

    @abstractmethod
    async def upsert(self, chunks: list[Chunk], collection: str) -> int:
        """Insert or update chunks in the database.

        Args:
            chunks: List of Chunk objects containing text, metadata, and embeddings.
            collection: The namespace/collection to store these chunks in.

        Returns:
            The number of chunks successfully inserted/updated.
        """
        pass

    @abstractmethod
    async def query(
        self,
        embedding: list[float],
        collection: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """Search for the most similar chunks to a given query embedding.

        Args:
            embedding: The vector representing the user's question.
            collection: The namespace/collection to search inside.
            top_k: How many results to return (default 10).
            filters: Optional metadata filters (e.g., {"author": "John"}).

        Returns:
            A list of ScoredChunk objects containing the chunk text and similarity
            score.
        """
        pass

    @abstractmethod
    async def delete(self, chunk_ids: list[UUID], collection: str) -> int:
        """Delete specific chunks from the database by their IDs.

        Args:
            chunk_ids: List of UUIDs of the chunks to delete.
            collection: The namespace/collection they reside in.

        Returns:
            The number of chunks successfully deleted.
        """
        pass

    @abstractmethod
    async def list_collections(self) -> list[str]:
        """List all available collections in this vector store.

        Returns:
            List of collection names.
        """
        pass

    @abstractmethod
    async def collection_stats(self, collection: str) -> dict[str, Any]:
        """Get statistics about a specific collection.

        This is heavily used by the Dashboard and MCP tools to understand the state
        of the database without querying raw data.

        Args:
            collection: The name of the collection.

        Returns:
            A dictionary containing stats like 'chunk_count', 'document_count', etc.
        """
        pass
