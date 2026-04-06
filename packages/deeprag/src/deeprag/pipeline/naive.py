"""Naive RAG Pipeline — the simplest end-to-end workflow.

In this pipeline, a question follows exactly one path:
Question → Embed → Retrieve → Build Prompt → LLM → Answer

There is no grading, no rewriting, and no self-correction.
This establishes our baseline.
"""

from typing import Any

from shared.llm.service import LLMService
from shared.models.events import ProgressCallback, ProgressEvent, ProgressEventType
from shared.models.generation import GenerationResult, ResponseType, Citation

from deeprag.prompts.naive import (
    NAIVE_SYSTEM_PROMPT,
    NAIVE_USER_PROMPT,
    format_context_chunks,
)
from deeprag.retrieval.naive import NaiveRetriever


class NaiveRAGPipeline:
    """The Naive (baseline) RAG Pipeline."""

    def __init__(self, retriever: NaiveRetriever, llm: LLMService):
        self._retriever = retriever
        self._llm = llm

    async def query(
        self,
        question: str,
        collection: str,
        top_k: int = 5,
        callback: ProgressCallback | None = None,
    ) -> GenerationResult:
        """Execute the end-to-end naive RAG workflow.

        Args:
            question: The user's question.
            collection: ChromaDB collection to search against.
            top_k: Number of chunks to retrieve.
            callback: Optional event tracker object.

        Returns:
            A GenerationResult containing the LLM's answer and token metadata.
        """
        # Step 1: Query received
        if callback:
            await callback(
                ProgressEvent(
                    event_type=ProgressEventType.QUERY_RECEIVED,
                    message=f"Received query: '{question}'",
                )
            )

        # Step 2: Retrieve chunks
        if callback:
            await callback(
                ProgressEvent(
                    event_type=ProgressEventType.RETRIEVAL_STARTED,
                    message=(
                        f"Searching collection '{collection}' "
                        f"for top {top_k} results."
                    ),
                )
            )

        retrieval_res = await self._retriever.retrieve(
            query=question, collection=collection, top_k=top_k
        )

        if callback:
            await callback(
                ProgressEvent(
                    event_type=ProgressEventType.RETRIEVAL_COMPLETE,
                    message=f"Found {len(retrieval_res.chunks)} relevant chunks.",
                )
            )

        # Step 3: Format the prompt
        context_str = format_context_chunks(retrieval_res.chunks)
        user_message = NAIVE_USER_PROMPT.format(
            context=context_str, question=question
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": NAIVE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        # Step 4: Generation
        # (The GENERATING event is handled inside LLMService)
        llm_response = await self._llm.complete(messages=messages, callback=callback)

        citations = [
            Citation(
                chunk_id=c.chunk.id,
                text_span="[Naive RAG Entire Context]",
                relevance_score=c.retrieval_score,
            )
            for c in retrieval_res.chunks
        ]

        # Step 5: Return result
        # The naive pipeline doesn't know how to grade itself, so it's CONFIDENT
        result = GenerationResult(
            query=question,
            answer=llm_response.content,
            response_type=ResponseType.CONFIDENT,
            citations=citations,
            metadata={
                "input_tokens": llm_response.input_tokens,
                "output_tokens": llm_response.output_tokens,
                "model": llm_response.model_name,
            },
        )

        if callback:
            await callback(
                ProgressEvent(
                    event_type=ProgressEventType.COMPLETE,
                    message="Pipeline complete.",
                    completion_pct=1.0,
                )
            )

        return result
