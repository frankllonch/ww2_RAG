# WW2 RAG — Local Retrieval-Augmented Generation System

A fully local RAG system (Retrieval-Augmented Generation) built from World War II Wikipedia articles.  
Includes Elasticsearch vector search, BGE sentence embeddings, a local LLM served by Ollama, and a Streamlit UI with WW2 aesthetics, custom fonts, avatar icons, and an interactive 3D helmet.

Runs 100% offline on any computer.

## Overview

This project performs:

- Document chunking  
- Embedding generation  
- Vector indexing in Elasticsearch  
- Context retrieval using cosine similarity  
- Local LLM inference via Ollama  
- Streamlit-based interface with animations and 3D rendering

Pipeline:

Streamlit UI → RAG Pipeline → Elasticsearch Vector Store → Ollama LLM

---

## Requirements

You need:

- Python 3.10+  
- Docker Desktop  
- Ollama installed  
- macOS, Linux, or Windows (WSL2 recommended)

---

## 1. Create Python Environment

Inside the project folder, create a virtual environment:

"""python3 -m venv ww2env"""

Activate it:

"""source ww2env/bin/activate"""

Install dependencies:

"""pip install -r requirements.txt"""

---

## 2. Start Elasticsearch with Docker

Start services:

"""docker compose up -d"""

Check Elasticsearch:

"""curl http://localhost:9200"""

If port conflicts happen:

"""docker stop $(docker ps -q)"""  
"""docker compose up -d"""

---

## 3. Prepare the Dataset

Place your dataset at:

`data/raw_wiki.jsonl`

Each item must follow:

"""
{
  "title": "Article Title",
  "url": "https://en.wikipedia.org/...",
  "content": "Full text of the article..."
}
"""

---

## 4. Index the Embeddings

The indexing script:

- Splits articles into ~1000-char chunks  
- Generates embeddings  
- Stores vectors in Elasticsearch

Run:

"""python src/indexer.py"""

You should see:

- Index created  
- Number of chunks  
- “Indexing completed”

---

## 5. Install Ollama Models

Pull a model:

"""ollama pull qwen2.5:7b-instruct"""

Other supported models:

"""ollama pull deepseek-r1:7b"""  
"""ollama pull llama3.1:8b"""  
"""ollama pull mistral:7b-instruct"""

Start Ollama server:

"""ollama serve"""

(Keep this terminal open)

---

## 6. Test Retrieval

Test vector retrieval:

"""python src/retriever.py"""

Test full RAG generation:

"""python src/rag_pipeline.py"""

---

## 7. Run the Streamlit UI

Launch:

"""streamlit run app.py"""

Then open:  
http://localhost:8501

Features:

- Animated WW2-style Fraktur title  
- Hitler & Stalin avatars  
- LLM model switcher  
- RAG context injection  
- Embedded 3D helmet  
- Custom fonts  
- Gun cursor  
- Smooth hover animations  

---

## 8. Project Structure

"""
ww2_RAG/
  app.py
  docker-compose.yml
  requirements.txt
  static/
      helmet_base64.txt
      hitler.png
      stalin.png
      cursor_gun.png
      fonts/
          fraktur_regular.ttf
  data/
      raw_wiki.jsonl
  src/
      indexer.py
      retriever.py
      rag_pipeline.py
      embedder.py
      utils.py
"""

---

## 9. Troubleshooting

**Elasticsearch errors**

Reset ES:

"""docker compose down -v"""  
"""docker compose up -d"""

**Ollama connection refused**

Ensure Ollama is running:

"""ollama serve"""

**Module import errors**

Add:

"""import sys, os; sys.path.append(os.getcwd())"""

**3D model not showing**

Ensure you Base64-encoded it:

"""
import base64
with open("static/helmet.glb","rb") as f:
    encoded = base64.b64encode(f.read()).decode()
with open("static/helmet_base64.txt","w") as f:
    f.write(encoded)
"""

---

## 10. Done 🎉

You now have a local WW2 RAG system with:

- Local embeddings  
- Vector search  
- Local LLM  
- Animated UI  
- Custom fonts  
- 3D rendering

If you want, I can also generate:

- An installer script  
- A Makefile  
- Docker-only version  
- A full developer guide  

Just ask!
