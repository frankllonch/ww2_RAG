from typing import List

def chunk_text(text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
    """
    Simple char-based chunking with overlap.
    """
    text = text.replace("\n", " ").strip()
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = end - overlap  # step back by overlap

    return chunks

if __name__ == "__main__":
    t = "This is a small test text." * 50
    c = chunk_text(t, max_chars=50, overlap=10)
    print(f"Chunks: {len(c)}")
    for i, ch in enumerate(c[:3]):
        print(f"\n--- chunk {i} ---\n{ch}")