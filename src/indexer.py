import json
from pathlib import Path
from typing import Iterable

from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

from embedder import embed_documents
from chunker import chunk_text


DATA_PATH = Path("data/processed_wikipedia_structured.jsonl")
INDEX_NAME = "ww2_wiki"

# BGE-small -> 384 dims
EMBEDDING_DIMS = 384

def get_es_client():
    return Elasticsearch(
        ["http://localhost:9200"],
        verify_certs=False,
        ssl_show_warn=False
    )

def create_index(client: Elasticsearch):
    # HEAD /index is buggy in ES 8.14 → causes 400
    try:
        client.indices.get(index=INDEX_NAME)
        print(f"[INFO] Index '{INDEX_NAME}' already exists, skipping create.")
        return
    except Exception:
        # Index does not exist → create it
        pass

    mapping = {
        "mappings": {
            "properties": {
                "topic": {"type": "text"},
                "summary": {"type": "text"},
                "key_points": {"type": "text"},
                "locations": {"type": "text"},
                "people": {"type": "text"},
                "date": {"type": "text"},
                "raw_text": {"type": "text"},
                "source": {"type": "text"},
                "url": {"type": "text"},
                "chunk_id": {"type": "integer"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": EMBEDDING_DIMS
                },
            }
        }
    }

    client.indices.create(index=INDEX_NAME, body=mapping)
    print(f"[OK] Index '{INDEX_NAME}' created.")

def iter_documents() -> Iterable[dict]:
    """
    Yields docs ready to be embedded/indexed from processed_wikipedia_structured.jsonl
    """
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)

            topic = rec.get("topic", "")
            summary = rec.get("summary", "")
            key_points = rec.get("key_points", [])
            locations = rec.get("locations", [])
            people = rec.get("people", [])
            date = rec.get("date", "")
            raw_text = rec.get("raw_text", "")
            source = rec.get("source", "wikipedia")
            url = rec.get("url", "")

            chunks = chunk_text(raw_text, max_chars=900, overlap=150)

            for i, chunk in enumerate(chunks):
                yield {
                    "topic": topic,
                    "summary": summary,
                    "key_points": ", ".join(key_points),
                    "locations": ", ".join(locations),
                    "people": ", ".join(people),
                    "date": date,
                    "raw_text": chunk,
                    "source": source,
                    "url": url,
                    "chunk_id": i,
                }

def bulk_index():
    client = get_es_client()
    create_index(client)

    docs = list(iter_documents())
    print(f"[INFO] Total chunks to index: {len(docs)}")

    # Embed in batches to not blow up memory
    batch_size = 64
    actions = []
    for i in tqdm(range(0, len(docs), batch_size), desc="Indexing batches"):
        batch = docs[i : i + batch_size]
        texts = [d["raw_text"] for d in batch]
        embeddings = embed_documents(texts)

        for doc, emb in zip(batch, embeddings):
            action = {
                "_index": INDEX_NAME,
                "_source": {
                    "topic": doc["topic"],
                    "summary": doc["summary"],
                    "key_points": doc["key_points"],
                    "locations": doc["locations"],
                    "people": doc["people"],
                    "date": doc["date"],
                    "raw_text": doc["raw_text"],
                    "source": doc["source"],
                    "url": doc["url"],
                    "chunk_id": doc["chunk_id"],
                    "embedding": emb,
                },
            }
            actions.append(action)

        if len(actions) >= batch_size:
            helpers.bulk(client, actions)
            actions = []

    if actions:
        helpers.bulk(client, actions)

    print("[OK] Indexing completed.")

if __name__ == "__main__":
    bulk_index()