import os
import json
from pathlib import Path

import wikipedia
from tqdm import tqdm

OUTPUT_PATH = Path("data/raw_wiki.jsonl")

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
]

def fetch_page(title: str) -> dict | None:
    try:
        page = wikipedia.page(title, auto_suggest=False)
        return {
            "title": page.title,
            "url": page.url,
            "content": page.content,
        }
    except Exception as e:
        print(f"[WARN] Could not fetch '{title}': {e}")
        return None

def main():
    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for title in tqdm(WW2_TITLES, desc="Downloading Wikipedia pages"):
            data = fetch_page(title)
            if data:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
    print(f"[OK] Saved pages to {OUTPUT_PATH}")

if __name__ == "__main__":
    wikipedia.set_lang("en")
    main()