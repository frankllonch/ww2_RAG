from typing import List, Dict, Any

from elasticsearch import Elasticsearch

from src.embedder import embed_query

INDEX_NAME = "ww2_wiki"

def get_es_client() -> Elasticsearch:
    return Elasticsearch("http://localhost:9200")

def retrieve(query: str, k: int = 5) -> List[Dict[str, Any]]:
    client = get_es_client()
    q_vector = embed_query(query)

    body = {
        "_source": ["topic", "summary", "raw_text", "url", "key_points", "locations", "people", "date"],
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": q_vector}
                }
            }
        }
    }

    resp = client.search(index=INDEX_NAME, body=body)
    hits = resp["hits"]["hits"]
    results = [
        {
            "score": h["_score"],
            "topic": h["_source"].get("topic", ""),
            "summary": h["_source"].get("summary", ""),
            "raw_text": h["_source"].get("raw_text", ""),
            "url": h["_source"].get("url", ""),
            "key_points": h["_source"].get("key_points", ""),
            "locations": h["_source"].get("locations", ""),
            "people": h["_source"].get("people", ""),
            "date": h["_source"].get("date", ""),
        }
        for h in hits
    ]
    return results

if __name__ == "__main__":
    docs = retrieve("What was Operation Barbarossa?", k=3)
    for i, d in enumerate(docs):
        print(f"\n--- hit {i} (score={d['score']}) ---")
        print(d["title"], d["url"])
        print(d["text"][:400], "...")