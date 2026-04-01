"""CLI for the DeepRAG pipeline — powered by Typer and Rich.

Usage:
    deeprag ingest --source ./docs --collection my-kb

This is the user-facing command that kicks off the ingestion pipeline.
It loads documents from a folder, chunks them, embeds them, and stores
them in ChromaDB — all in one command.
"""

import asyncio

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from deeprag.embeddings.service import EmbeddingService
from deeprag.ingestion.pipeline import run_ingestion
from deeprag.vectorstore.chromadb_adapter import ChromaDBAdapter

app = typer.Typer(
    name="deeprag",
    help="DeepRAG — a self-correcting, observable RAG system.",
)
console = Console()


@app.command()
def ingest(
    source: str = typer.Option(
        ...,
        "--source",
        "-s",
        help="Path to the directory containing documents to ingest.",
    ),
    collection: str = typer.Option(
        "default",
        "--collection",
        "-c",
        help="Name of the ChromaDB collection to store chunks in.",
    ),
    chunk_strategy: str = typer.Option(
        "recursive",
        "--chunk-strategy",
        help="Chunking strategy: recursive, sliding_window, or semantic.",
    ),
    chunk_size: int = typer.Option(
        512,
        "--chunk-size",
        help="Maximum characters per chunk.",
    ),
    chunk_overlap: int = typer.Option(
        50,
        "--chunk-overlap",
        help="Character overlap between chunks.",
    ),
    chromadb_host: str = typer.Option(
        "localhost",
        "--chromadb-host",
        help="ChromaDB server hostname.",
    ),
    chromadb_port: int = typer.Option(
        8000,
        "--chromadb-port",
        help="ChromaDB server port.",
    ),
) -> None:
    """Ingest documents into the vector store."""
    console.print(
        "\n[bold blue]🚀 DeepRAG Ingestion[/bold blue]",
    )
    console.print(f"  Source:     [cyan]{source}[/cyan]")
    console.print(f"  Collection: [cyan]{collection}[/cyan]")
    console.print(f"  Strategy:   [cyan]{chunk_strategy}[/cyan]")
    console.print(
        f"  Chunk size: [cyan]{chunk_size}[/cyan] (overlap: {chunk_overlap})\n"
    )

    vector_store = ChromaDBAdapter(host=chromadb_host, port=chromadb_port)
    embedding_service = EmbeddingService()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Starting ingestion...", total=None)

        async def progress_callback(event):
            progress.update(task, description=event.message)

        result = asyncio.run(
            run_ingestion(
                source_dir=source,
                collection=collection,
                vector_store=vector_store,
                embedding_service=embedding_service,
                chunk_strategy=chunk_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                callback=progress_callback,
            )
        )

    console.print("\n[bold green]✅ Ingestion Complete![/bold green]")
    console.print(f"  Documents loaded: [cyan]{result.document_count}[/cyan]")
    console.print(f"  Chunks created:   [cyan]{result.chunk_count}[/cyan]")
    console.print(f"  Time taken:       [cyan]{result.duration_ms:.0f}ms[/cyan]")

    if result.failed_files:
        console.print(
            f"\n[bold yellow]⚠️  {len(result.failed_files)} files skipped:[/bold yellow]"
        )
        for f in result.failed_files:
            console.print(f"    [dim]{f}[/dim]")

    console.print()


if __name__ == "__main__":
    app()
