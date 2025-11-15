def wrap_letters(text: str) -> str:
    return "".join(f"<span>{c}</span>" for c in text)