"""Embedding service for converting text to vector representations."""

import asyncio

from sentence_transformers import SentenceTransformer
from shared.models.events import ProgressCallback, ProgressEvent, ProgressEventType


class EmbeddingService:
    """Service to generate embeddings for chunks using sentence-transformers.

    This service is designed to be efficient:
    - Lazy loading of the model (only loads into memory when first requested)
    - Batching support to avoid Out Of Memory (OOM) errors
    - Async interface to play nicely with our async RAG pipeline
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        """Lazy load the model to save memory if embeddings aren't used immediately."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    async def embed(
        self, texts: list[str], callback: ProgressCallback | None = None
    ) -> list[list[float]]:
        """Embed a list of texts into vectors asynchronously.

        Args:
            texts: List of strings to embed.
            callback: Optional callback to track batch progress.

        Returns:
            List of embedding vectors (list of floats).
        """
        if not texts:
            return []

        model = self._get_model()
        all_embeddings: list[list[float]] = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]

            # SentenceTransformer.encode is synchronous and cpu/gpu intensive.
            # We run it in a thread executor to avoid blocking the asyncio event loop.
            batch_embeddings = await asyncio.to_thread(
                model.encode, batch_texts, convert_to_numpy=False
            )

            # Ensure it's a list of list of floats
            if isinstance(batch_embeddings, list):
                # The numpy conversion returns list of tensors or floats
                emb_list = [list(float(x) for x in emb) for emb in batch_embeddings]
                all_embeddings.extend(emb_list)  # type: ignore
            else:
                # It might return a single tensor/array if batch is 1,
                # but usually a 2D array
                all_embeddings.extend(batch_embeddings.tolist())

            if callback:
                current_batch = (i // self.batch_size) + 1
                await callback(
                    ProgressEvent(
                        event_type=ProgressEventType.EMBEDDING_BATCH,
                        message=f"Embedded batch {current_batch}/{total_batches}",
                        completion_pct=current_batch / total_batches,
                    )
                )

        return all_embeddings
