"""Script to download a realistic knowledge base for benchmarking.

Instead of short SQuAD paragraphs, this script downloads 5 full 
Wikipedia articles. Each article is thousands of words long. 
This forces the RAG pipeline to actually use its chunking 
strategies and test 'needle in a haystack' retrieval.
"""

import urllib.request
import os
import json

def get_full_articles():
    topics = [
        "Norman_Conquest", 
        "Computational_complexity_theory", 
        "Southern_California",
        "Apollo_11",
        "Immune_system"
    ]

    os.makedirs("docs", exist_ok=True)
    os.makedirs("benchmarks", exist_ok=True)

    print("🗑️  Cleaning up old txt files...")
    for file in os.listdir("docs"):
        if file.endswith(".txt"):
            os.remove(os.path.join("docs", file))

    print(f"⏳ Downloading {len(topics)} full Wikipedia articles...")
    
    for topic in topics:
        print(f"  ⬇️ Downloading {topic}...")
        url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&titles={topic}&format=json"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            pages = data['query']['pages']
            for page_id, page_data in pages.items():
                title = page_data['title']
                content = page_data.get('extract', '')
                with open(f"docs/{topic}.txt", "w", encoding="utf-8") as f:
                    f.write(f"Title: {title}\n\n")
                    f.write(content)
                    
    print("\n✨ Successfully saved full articles to 'docs/'!")
    print("\nNext step: Run `uv run deeprag ingest --source ./docs --collection phase1-hardened`")

if __name__ == "__main__":
    get_full_articles()
