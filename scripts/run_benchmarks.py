"""Run golden benchmark against Naive RAG and record metrics."""

import asyncio
import json
import time
from typing import Any

from rich.console import Console
from rich.progress import Progress

from shared.config import AppConfig
from shared.llm.service import LLMService
from deeprag.vectorstore.chromadb_adapter import ChromaDBAdapter
from deeprag.embeddings.service import EmbeddingService
from deeprag.retrieval.naive import NaiveRetriever
from deeprag.pipeline.naive import NaiveRAGPipeline


async def run_benchmark():
    console = Console()
    
    with open("benchmarks/golden_tests.json", "r") as f:
        tests = json.load(f)
        
    config = AppConfig()
    
    vector_store = ChromaDBAdapter()
    embed_service = EmbeddingService()
    llm_service = LLMService(config.llm)
    
    retriever = NaiveRetriever(vector_store, embed_service)
    pipeline = NaiveRAGPipeline(retriever, llm_service)
    
    results: list[dict[str, Any]] = []
    
    # Pricing for Qwen Open-Source (approx $0 for local, so we just log 0 unless API is used)
    # Using litellm default rough estimate
    total_latency = 0.0
    total_tokens = 0
    total_cost = 0.0
    
    console.print(f"[bold blue]🚀 Starting Phase 1 Benchmark on {len(tests)} test cases...[/bold blue]")
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Running queries...", total=len(tests))
        
        for t in tests:
            start_time = time.perf_counter()
            
            try:
                res = await pipeline.query(
                    question=t["question"],
                    collection="phase1-benchmark",
                    top_k=3
                )
                
                latency_ms = (time.perf_counter() - start_time) * 1000
                total_latency += latency_ms
                
                in_t = res.metadata.get("input_tokens", 0)
                out_t = res.metadata.get("output_tokens", 0)
                toks = in_t + out_t
                total_tokens += toks
                
                results.append({
                    "id": t["id"],
                    "category": t["category"],
                    "question": t["question"],
                    "expected_answer": t["expected_answer"],
                    "expected_response_type": t["expected_response_type"],
                    "actual_answer": res.answer,
                    "actual_response_type": res.response_type,
                    "latency_ms": round(latency_ms, 2),
                    "total_tokens": toks,
                    "chunks_retrieved": len(res.citations),
                    "manual_accuracy": None # To be filled by user
                })
                
            except Exception as e:
                console.print(f"[red]Error on test {t['id']}: {e}[/red]")
                
            progress.advance(task)
            
    # Compute Aggregates
    avg_latency = total_latency / len(tests)
    avg_tokens = total_tokens / len(tests)
    
    report = {
        "metadata": {
            "model_used": config.llm.model,
            "total_questions": len(tests),
            "average_latency_ms": round(avg_latency, 2),
            "average_tokens": round(avg_tokens, 2),
            "total_cost_usd": total_cost, # 0 for local slurm
            "accuracy": None # Pending LLM-as-a-judge in Phase 2
        },
        "results": results
    }
    
    with open("benchmarks/phase1_baseline.json", "w") as f:
        json.dump(report, f, indent=2)
        
    console.print("\n[bold green]✅ Benchmarking Complete![/bold green]")
    console.print(f"Results saved to [cyan]benchmarks/phase1_baseline.json[/cyan]")
    console.print(f"Average Latency: {round(avg_latency, 2)}ms")
    console.print(f"Average Tokens: {round(avg_tokens, 2)}")
    

if __name__ == "__main__":
    asyncio.run(run_benchmark())
