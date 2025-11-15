# src/rag_pipeline.py

import requests
import textwrap
from src.retriever import retrieve

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b-instruct"


def build_prompt(question: str, contexts: list[str]) -> str:
    """
    Build the RAG prompt. Qwen2.5 works very well with concise instructions.
    """
    context_block = "\n\n---\n".join(contexts)

    return f"""
You are a highly accurate historian specialized in World War II.
Answer the question ONLY using the information from the CONTEXT.
If the answer is not present, say you don't know and suggest what to search.

CONTEXT:
{context_block}

QUESTION: {question}

Provide a clear answer (4–8 sentences). Do NOT invent information.
""".strip()


def call_ollama(prompt: str, model: str):
    payload = {"model": model, "prompt": prompt, "stream": False}
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    return resp.json().get("response", "")

def answer_question(question: str, k: int = 5, model="qwen2.5:7b-instruct"):
    hits = retrieve(question, k=k)
    contexts = [h["text"] for h in hits]
    prompt = build_prompt(question, contexts)
    return call_ollama(prompt, model=model)


if __name__ == "__main__":
    q = "What were the main causes of World War II?"
    print("Question:", q)
    print()
    print("Answer:", answer_question(q, k=5))