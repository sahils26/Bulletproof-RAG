"""Script to download a sample knowledge base for Phase 1 Benchmarking.

We use a subset of the SQuAD v2 dataset. It contains high-quality Wikipedia
paragraphs which are great for testing factual and multi-hop queries.
The script extracts unique contexts and saves them as .txt files in docs/.
"""

import os
from collections import OrderedDict

from datasets import load_dataset


def get_squad_sample():
    print("⏳ Downloading SQuAD v2 dataset (this may take a few seconds)...")
    # Load the validation split to keep it fast
    # Streaming = False because SQuAD is small enough to fit in RAM
    dataset = load_dataset("squad_v2", split="validation")

    print("✅ Dataset loaded! Extracting unique contexts from 3 different topics...")

    # We want exactly 20 unique documents/contexts for 3 different topics
    topics_needed = 3
    paragraphs_per_topic = 20

    unique_contexts = OrderedDict()
    topic_counts = {}

    for row in dataset:
        title = row["title"]
        context = row["context"]

        # Track topics we're collecting
        if title not in topic_counts and len(topic_counts) >= topics_needed:
            continue

        if title not in topic_counts:
            topic_counts[title] = 0

        if topic_counts[title] >= paragraphs_per_topic:
            continue

        doc_key = f"{title}_{context[:30]}"

        if doc_key not in unique_contexts:
            unique_contexts[doc_key] = {
                "title": title,
                "context": context,
                "example_question": row["question"],
            }
            topic_counts[title] += 1

        # Check if we have enough
        if len(unique_contexts) >= (topics_needed * paragraphs_per_topic):
            break

    # Ensure docs directory exists
    os.makedirs("docs", exist_ok=True)
    os.makedirs("benchmarks", exist_ok=True)

    # Clean up old txt files in docs/ before saving
    for file in os.listdir("docs"):
        if file.endswith(".txt"):
            os.remove(os.path.join("docs", file))

    print(f"📂 Saving {len(unique_contexts)} context files to docs/...")

    for i, (_key, data) in enumerate(unique_contexts.items(), 1):
        safe_title = data["title"].replace("/", "_").replace(" ", "_")
        filename = f"docs/squad_{safe_title}_{i}.txt"

        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Title: {data['title']}\n\n")
            f.write(data["context"])

    print("✨ Successfully saved 20 sample documents to 'docs/'!")

    print(
        "\nNext step: Run "
        "`uv run deeprag ingest --source ./docs "
        "--collection phase1-benchmark`"
    )


if __name__ == "__main__":
    get_squad_sample()
