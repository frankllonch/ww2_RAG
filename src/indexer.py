import json
from pathlib import Path
from typing import Iterable

from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

from chunker import chunk_text
from embedder import embed_documents

DATA_PATH = Path("data/raw_wiki.jsonl")
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
                "title": {"type": "text"},
                "url": {"type": "keyword"},
                "text": {"type": "text"},
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
    Yields docs ready to be embedded/indexed from raw_wiki.jsonl
    """
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            title = rec["title"]
            url = rec["url"]
            content = rec["content"]
            chunks = chunk_text(content, max_chars=1000, overlap=200)

            for i, chunk in enumerate(chunks):
                yield {
                    "title": title,
                    "url": url,
                    "text": chunk,
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
        texts = [d["text"] for d in batch]
        embeddings = embed_documents(texts)

        for doc, emb in zip(batch, embeddings):
            action = {
                "_index": INDEX_NAME,
                "_source": {
                    "title": doc["title"],
                    "url": doc["url"],
                    "text": doc["text"],
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