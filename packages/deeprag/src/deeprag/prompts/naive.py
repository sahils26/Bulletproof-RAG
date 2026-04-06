"""Prompt Templates for the Naive RAG Pipeline."""

from shared.models.retrieval import ScoredChunk

NAIVE_SYSTEM_PROMPT = """You are a highly helpful and intelligent assistant.
Your goal is to answer the user's question accurately using ONLY the provided context.

Rules:
1. Base your answer strictly on the provided context fragments.
2. If the answer cannot be confidently deduced from the context, clearly state: 
   "I'm sorry, I don't have enough information in my knowledge base to answer that."
3. Do not invent, hallucinate, or bring in outside knowledge.
4. When you state a fact, optionally cite the [Source X] if helpful to the user.
"""

NAIVE_USER_PROMPT = """Context fragments:
{context}

User Question: {question}

Please answer the question using the context above.
"""


def format_context_chunks(chunks: list[ScoredChunk]) -> str:
    """Format matching chunks into a clean prompt string."""
    if not chunks:
        return "No relevant context found."

    formatted = []
    for i, c in enumerate(chunks):
        source = c.chunk.metadata.get("source", "Unknown Source")
        doc_id = c.chunk.document_id
        text = c.chunk.content.strip()
        formatted.append(f"[Source {i+1} - {source} ({doc_id})]\n{text}\n")

    return "\n".join(formatted)
