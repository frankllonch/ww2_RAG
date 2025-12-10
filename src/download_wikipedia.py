import os
import json
import re
from pathlib import Path
import requests
from bs4 import BeautifulSoup

import wikipedia
from tqdm import tqdm

RELEVANT_KEYWORDS = [
    "war", "world war", "battle", "operation", "campaign", "front",
    "invasion", "occupation", "resistance", "allies", "axis",
    "wehrmacht", "nazi", "fascist", "imperial", "fleet", "u-boat",
    "submarine", "navy", "air force", "theatre", "theater"
]

STOPWORDS = [
    "film", "album", "song", "actor", "actress", "musician",
    "tv series", "football", "basketball", "novel", "poem", "fiction",
    "2022", "2021", "2020", "2019", "2018", "2017", "2016"
]

OUTPUT_PATH = Path("data/raw_wiki.jsonl")

# --- LLM (Ollama) config for structured summaries ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b-instruct"  # or any other model you prefer from your local list
def clean_wikipedia_content(html: str) -> str:
    """
    Limpia el HTML de Wikipedia eliminando tablas, referencias, índice, etc.
    Devuelve solo texto plano útil para el resumen y el RAG.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Eliminar elementos que normalmente no aportan al contexto histórico
    for tag in soup.select("table, sup, .reference, .toc"):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # Normalizar saltos de línea
    text = re.sub(r"\n+", "\n", text).strip()

    # Cortar secciones poco útiles
    cutoff_sections = ["See also", "References", "Further reading", "External links", "Bibliography", "Notas", "Referencias"]
    for cut in cutoff_sections:
        idx = text.lower().find(cut.lower())
        if idx != -1:
            text = text[:idx].strip()

    return text

def summarize_with_llm(topic: str, cleaned_text: str) -> dict:
    """
    Usa el LLM local (Ollama) para generar un resumen estructurado
    en el formato óptimo para RAG.

    Formato esperado de salida:
    {
      "topic": "...",
      "summary": "...",
      "key_points": ["...", "..."],
      "locations": ["..."],
      "people": ["..."],
      "date": "..."
    }
    """
    # Recortamos por seguridad para no enviar textos gigantes
    max_chars = 7000
    truncated_text = cleaned_text[:max_chars]

    prompt = f"""
Eres un experto historiador militar y analista de inteligencia.
A partir del siguiente texto de Wikipedia sobre "{topic}", quiero que generes
un objeto JSON válido con este formato EXACTO:

{{
  "topic": "título del evento o tema principal (string)",
  "summary": "resumen corto en 5-10 frases, muy conciso y factual (string)",
  "key_points": [
    "lista de 5-10 puntos clave sobre lo sucedido, cronología, consecuencias, etc.",
    "cada elemento es una frase corta y clara"
  ],
  "locations": [
    "lista de lugares importantes mencionados (países, ciudades, mares, etc.)"
  ],
  "people": [
    "lista de personas, organizaciones o actores clave (comandantes, países, servicios de inteligencia, etc.)"
  ],
  "date": "fecha principal del evento o rango de fechas en formato libre, por ejemplo 'June 1944' o '1945-1947'"
}}

Responde **solo** con el JSON, sin texto adicional.

Texto:
{truncated_text}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
        },
        timeout=600,
    )
    response.raise_for_status()
    raw = response.json().get("response", "")

    # Intentar parsear directamente como JSON
    try:
        data = json.loads(raw)
        return data
    except Exception:
        # Si el modelo rodea el JSON con texto, intentar extraer el bloque {...}
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass

    # Fallback robusto para no reventar el pipeline
    return {
        "topic": topic,
        "summary": raw[:500],
        "key_points": [],
        "locations": [],
        "people": [],
        "date": "",
    }

def build_rag_record_from_page(title: str, html: str, url: str | None = None) -> dict:
    """
    Dado un título de Wikipedia, su HTML y su URL,
    genera un registro completo listo para indexar en el RAG.

    Mantiene el mismo nivel de contexto que el pipeline actual
    (texto largo completo) pero añade una capa estructurada con el LLM.
    """
    cleaned_text = clean_wikipedia_content(html)
    structured = summarize_with_llm(title, cleaned_text)

    record = {
        "topic": structured.get("topic", title),
        "summary": structured.get("summary", ""),
        "key_points": structured.get("key_points", []),
        "locations": structured.get("locations", []),
        "people": structured.get("people", []),
        "date": structured.get("date", ""),
        # Contexto completo para el RAG (chunker/embeddings posteriores)
        "raw_text": cleaned_text,
        "source": "wikipedia",
        "url": url,
    }
    return record

WW2_TITLES = [
    "World War II",
    "Invasion of Poland",
    "Battle of Britain",
    "Operation Barbarossa",
    "Pearl Harbor",
    "Battle of Stalingrad",
    "D-Day",
    "Battle of Midway",
    "Winston Churchill",
    "Adolf Hitler",
    "Nazi Germany",
    "Allies of World War II",
    "Axis powers",
    "Holocaust",
    "Franklin D. Roosevelt",
    "Joseph Stalin",
    "Japanese expansionism",
    "Pacific War",
    "Eastern Front (World War II)",
    "Western Front (World War II)",
    "North African campaign",
    "Italian campaign (World War II)",
    "Battle of Kursk",
    "Operation Torch",
    "Operation Market Garden",
    "Battle of the Bulge",
    "Battle of Okinawa",
    "Battle of Berlin",
    "Manhattan Project",
    "Yalta Conference",
    "Tehran Conference",
    "Potsdam Conference",
    "Gestapo",
    "Hermann Göring",
    "Heinrich Himmler",
    "Imperial Japanese Navy",
    "Imperial Japanese Army",
    "European theatre of World War II",
    "Pacific theatre of World War II",
    "Asian theatre of World War II",
    "African theatre of World War II",
    "American Theater (World War II)",
    "South American involvement in World War II",
    "Brazil in World War II",
    "Argentina during World War II",
    "Battle of the Atlantic",
    "Battle of El Alamein",
    "Second Battle of El Alamein",
    "Battle of Tobruk",
    "East African campaign (World War II)",
    "Western Desert campaign",
    "Battle of Greece",
    "Battle of Crete",
    "Siege of Leningrad",
    "Battle of Moscow",
    "Battle of Normandy",
    "Operation Overlord",
    "Operation Bagration",
    "Battle of Guadalcanal",
    "Battle of Iwo Jima",
    "Battle of Saipan",
    "Battle of the Philippine Sea",
    "Battle of Leyte Gulf",
    "Battle of the Java Sea",
    "Philippines campaign (1941–1942)",
    "Philippines campaign (1944–1945)",
    "Burma campaign",
    "China Burma India Theater",
    "Second Sino-Japanese War",
    "Nanjing Massacre",
    "Atomic bombings of Hiroshima and Nagasaki",
    "Strategic bombing during World War II",
    "Bombing of Dresden in World War II",
    "Bombing of Tokyo in World War II",
    "Tripartite Pact",
    "Molotov–Ribbentrop Pact",
    "Vichy France",
    "Free France",
    "French Resistance",
    "Resistance movements during World War II",
    "Polish resistance movement in World War II",
    "Home Army (Poland)",
    "Yugoslav Partisans",
    "Italian resistance movement",
    "Kingdom of Italy",
    "Fascist Italy",
    "Italian Social Republic",
    "Empire of Japan",
    "Empire of Italy",
    "Italian Libya",
    "Kriegsmarine",
    "Luftwaffe",
    "Royal Navy",
    "Royal Air Force",
    "United States Navy",
    "United States Army Air Forces",
    "U-boat",
    "Concentration camp",
    "Extermination camp",
    "Auschwitz concentration camp",
    "Treblinka extermination camp",
    "Italian imperialism under Fascism",
]

WW2_TITLES += [
    "German submarine U-530",
    "German submarine U-977",
    "U-boat operations in South America",
    "Axis naval activity in South America",
    "Nazi submarines in Argentina"
]

def fetch_page(title: str) -> dict | None:
    try:
        # First attempt: strict fetch
        page = wikipedia.page(title, auto_suggest=False)
        return {
            "title": page.title,
            "url": page.url,
            "content": page.content,
        }
    except Exception:
        # Fallback 1: allow auto_suggest
        try:
            page = wikipedia.page(title, auto_suggest=True)
            return {
                "title": page.title,
                "url": page.url,
                "content": page.content,
            }
        except Exception:
            pass

        # Fallback 2: attempt using search results
        try:
            results = wikipedia.search(title)
            if results:
                best = results[0]
                page = wikipedia.page(best, auto_suggest=True)
                return {
                    "title": page.title,
                    "url": page.url,
                    "content": page.content,
                }
        except Exception:
            pass

        print(f"[WARN] Could not fetch '{title}' even with fallbacks.")
        return None

def expand_links(page_data: dict, max_links: int = 20) -> list[str]:
    try:
        title = page_data["title"]
        page = wikipedia.page(title, auto_suggest=False)
        raw_links = page.links[:max_links]

        filtered = []
        for link in raw_links:
            lower = link.lower()

            # Skip stopwords
            if any(sw in lower for sw in STOPWORDS):
                continue

            # Accept only if it contains a relevant keyword
            if any(kw in lower for kw in RELEVANT_KEYWORDS):
                filtered.append(link)

        return filtered
    except Exception:
        return []

def main():
    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    collected = set(WW2_TITLES)
    structured_records: list[dict] = []
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for title in tqdm(WW2_TITLES, desc="Downloading Wikipedia pages"):
            data = fetch_page(title)
            if data:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                # Expand linked pages for richer context
                new_links = expand_links(data)

                    if link not in collected:
                        collected.add(link)
                # --- NUEVO: construir registro estructurado para el RAG ---
                # Intentar obtener HTML real de la página para limpieza
                try:
                    page_obj = wikipedia.page(data["title"], auto_suggest=False)
                    html = page_obj.html()
                except Exception:
                    html = data.get("content", "")
                url = data.get("url", None)
                title_actual = data.get("title", None)
                record = build_rag_record_from_page(title_actual, html, url=url)
                structured_records.append(record)
    print("[INFO] Downloading expanded linked pages...")
    for title in tqdm(list(collected), desc="Linked pages"):
        data = fetch_page(title)
        if data:
            with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
            # --- NUEVO: construir registro estructurado para el RAG ---
            try:
                page_obj = wikipedia.page(data["title"], auto_suggest=False)
                html = page_obj.html()
            except Exception:
                html = data.get("content", "")
            url = data.get("url", None)
            title_actual = data.get("title", None)
            record = build_rag_record_from_page(title_actual, html, url=url)
            structured_records.append(record)
    print(f"[OK] Saved pages to {OUTPUT_PATH}")

    # --- Guardar dataset estructurado para el RAG ---
    output_path = Path("data/processed_wikipedia_structured.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for rec in structured_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[OK] Guardado dataset estructurado en {output_path}")

if __name__ == "__main__":
    wikipedia.set_lang("en")
    main()