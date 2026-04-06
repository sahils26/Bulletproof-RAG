import asyncio
import os
from shared.config import AppConfig
from shared.llm.service import LLMService, LLMResponse
from deeprag.vectorstore.chromadb_adapter import ChromaDBAdapter
from deeprag.retrieval.naive import NaiveRetriever
from deeprag.embeddings.service import EmbeddingService
from deeprag.pipeline.naive import NaiveRAGPipeline

async def main():
    # 1. Setup everything using our clean Day 8 classes
    config = AppConfig()
    
    # Check if we have the port-mapped Vector URL
    print(f"📡 Vector Store: {config.vector_store.url}")
    print(f"🧠 LLM Base: {config.llm.base_url}")
    print(f"🤖 Model: {config.llm.model}")

    vector_store = ChromaDBAdapter()
    embed_service = EmbeddingService()
    llm_service = LLMService(config.llm)
    
    retriever = NaiveRetriever(vector_store, embed_service)
    pipeline = NaiveRAGPipeline(retriever, llm_service)

    # 2. Ask a question!
    # Tip: Make sure you have ingested data into 'my-slurm-kb' first!
    question = "who all are the authors of this paper?" 
    
    print(f"\n🚀 Asking: '{question}'...")
    
    # We use a simple callback to see what's happening live
    async def progress_logger(event):
        print(f"   [EVENT] {event.event_type.upper()}: {event.message}")

    try:
        result = await pipeline.query(
            question=question,
            collection="my-slurm-kb",
            top_k=3,
            callback=progress_logger
        )

        print("\n" + "="*50)
        print("💡 QWEN 32B ANSWER:")
        print(result.answer)
        print("="*50)
        
        print(f"\n📚 Citations used: {len(result.citations)}")
        print(f"🎫 Tokens: {result.metadata['input_tokens']} prompt | {result.metadata['output_tokens']} generated")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Tip: Check if your SSH tunnel is still running on port 8000!")

if __name__ == "__main__":
    asyncio.run(main())
