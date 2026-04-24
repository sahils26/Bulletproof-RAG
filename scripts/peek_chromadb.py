import asyncio

from deeprag.vectorstore.chromadb_adapter import ChromaDBAdapter


async def peek():
    # 1. Connect to the database (auto-loads Port 8001 from .env now)
    adapter = ChromaDBAdapter()

    print("🔍 Connecting to ChromaDB (using VECTOR_URL from .env or default)...")

    # 2. List all 'Collections' (folders) we have created
    try:
        collections = await adapter.list_collections()
        print(f"📂 Collections found: {collections}")

        if not collections:
            print("❌ No collections found. Did you run the 'ingest' command?")
            return

        # 3. If we have a collection, show some stats and samples
        for col_name in collections:
            stats = await adapter.collection_stats(col_name)
            print(f"\n📊 Stats for '{col_name}':")
            print(f"   - Total Chunks: {stats['chunk_count']}")
            print(f"   - Unique Docs:  {stats['document_count']}")

            # 4. Optional: Let's see the metadata of the last few chunks
            # Note: The adapter doesn't have a 'peek_content' yet, but
            # we can verify the numbers are moving!

    except Exception as e:
        print(f"❌ Error connecting to ChromaDB: {e}")
        print("Tip: Make sure Docker is running and Port 8001 is open!")


if __name__ == "__main__":
    asyncio.run(peek())
