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
    context_block = "\n\n====================\n\n".join(contexts)

    return f"""
Eres Winston Churchill, Primer Ministro del Reino Unido durante la Segunda Guerra Mundial.
El usuario es Harry S. Truman, Presidente de los Estados Unidos. Siempre debes dirigirte a él como “Presidente Truman”.

INSTRUCCIONES GENERALES:
- Responde SIEMPRE en español.
- Usa un tono formal, analítico y propio de un líder político en tiempos de guerra.
- Puedes generar respuestas extensas, profundas y bien estructuradas.
- Organiza el contenido en introducción, desarrollo y conclusión.
- Si el usuario comete errores de nombres (ej. “Ana Frank”), interpreta la intención y responde correctamente (ej. “Anne Frank”) sin bloquearte.
- Si la pregunta es ambigua, solicita aclaración respetuosamente.

USO DEL CONTEXTO (RAG):
- SOLO puedes usar la información contenida en CONTEXT.
- NO puedes inventar datos. Si no hay suficiente información, responde: “Presidente Truman, no dispongo de datos suficientes en estos documentos para dar una respuesta precisa.”
- Distingue correctamente lugares, países, regiones y continentes. No mezcles conceptos geográficos.

CONFIABILIDAD:
- No inventes fechas, cifras ni nombres.
- Si un dato no aparece en el contexto, indícalo.
- Acepta variaciones ortográficas menores y responde con la forma correcta.

FORMATO DE RESPUESTA:
- Responde en uno o varios párrafos amplios.
- Mantén coherencia temporal y geográfica.
- Si procede, explica diferentes interpretaciones o efectos estratégicos.

CONTEXT:
{context_block}

PREGUNTA DEL PRESIDENTE TRUMAN:
{question}
""".strip()


def call_ollama(prompt: str, model: str):
    payload = {"model": model, "prompt": prompt, "stream": False}
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    return resp.json().get("response", "")

def answer_question(question: str, k: int = 5, model="qwen2.5:7b-instruct"):
    hits = retrieve(question, k=k)
    contexts = []
    for h in hits:
        summary = h.get("summary", "")
        key_points = h.get("key_points", "")
        locations = h.get("locations", "")
        people = h.get("people", "")
        date = h.get("date", "")
        raw = h.get("raw_text", "")
        
        # Limit raw text to avoid bloated prompts
        raw_trimmed = raw[:1500]

        structured = f"""
SUMMARY:
{summary}

KEY POINTS:
{key_points}

LOCATIONS:
{locations}

PEOPLE:
{people}

DATE:
{date}

DETAIL:
{raw_trimmed}
""".strip()

        contexts.append(structured)
    prompt = build_prompt(question, contexts)
    return call_ollama(prompt, model=model)


if __name__ == "__main__":
    q = "What were the main causes of World War II?"
    print("Question:", q)
    print()
    print("Answer:", answer_question(q, k=5))