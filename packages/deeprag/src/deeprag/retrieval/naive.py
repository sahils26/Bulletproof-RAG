"""Naive Retriever — the simplest form of vector search.

This retriever simply embeds the user's query, passes it to the VectorStore,
and returns the Top-K results exactly as they are found.

No re-ranking, no hybrid search, no grading. Just purely semantics.
"""

from deeprag.embeddings.service import EmbeddingService
from deeprag.vectorstore.base import VectorStore
from shared.models.retrieval import RetrievalResult, ScoredChunk


class NaiveRetriever:
    """A direct pass-through retriever."""

    def __init__(self, vector_store: VectorStore, embedding_service: EmbeddingService):
        self._store = vector_store
        self._embedder = embedding_service

    async def retrieve(
        self, query: str, collection: str, top_k: int = 5
    ) -> RetrievalResult:
        """Embed the query and retrieve matching chunks.

        Args:
            query: The user's question.
            collection: The ChromaDB collection to search in.
            top_k: Number of chunks to retrieve.

        Returns:
            A RetrievalResult containing the chunks and scores.
        """
        # Embed the query string into a vector list[float]
        # We wrap in a list because our service expects batch inputs, then take the 0th
        embeddings = await self._embedder.embed([query])
        query_vector = embeddings[0]

        # Query the vector store
        scored_chunks: list[ScoredChunk] = await self._store.query(
            embedding=query_vector,
            collection=collection,
            top_k=top_k,
        )

        # Build and return the result object
        return RetrievalResult(
            chunks=scored_chunks,
        )
