from typing import List

from sentence_transformers import SentenceTransformer

_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model

def embed_documents(texts: List[str]) -> List[list]:
    """
    Embeddings para documentos/pasajes.
    BGE recomienda prefijo 'passage: '.
    """
    model = get_model()
    to_encode = [f"passage: {t}" for t in texts]
    embeddings = model.encode(to_encode, normalize_embeddings=True)
    return embeddings.tolist()

def embed_query(text: str) -> list:
    """
    Embedding para queries.
    BGE recomienda prefijo 'query: '.
    """
    model = get_model()
    emb = model.encode([f"query: {text}"], normalize_embeddings=True)[0]
    return emb.tolist()

if __name__ == "__main__":
    vec = embed_query("What caused World War II?")
    print(f"Embedding dims: {len(vec)}")