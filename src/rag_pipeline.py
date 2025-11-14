# src/rag_pipeline.py

import requests
import textwrap
from retriever import retrieve

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


def call_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=180)
    response.raise_for_status()
    data = response.json()

    # Qwen sometimes adds thinking markers if running R1 distill variants.
    cleaned = data.get("response", "")
    cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    return cleaned.strip()


def answer_question(question: str, k: int = 5) -> str:
    hits = retrieve(question, k=k)

    if not hits:
        return "I couldn't find any relevant documents in the index."

    contexts = [h["text"] for h in hits]
    prompt = build_prompt(question, contexts)
    answer = call_ollama(prompt)

    return textwrap.dedent(answer).strip()


if __name__ == "__main__":
    q = "What were the main causes of World War II?"
    print("Question:", q)
    print()
    print("Answer:", answer_question(q, k=5))