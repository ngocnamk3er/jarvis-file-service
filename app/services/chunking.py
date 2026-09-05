def chunk_text(text: str, chunk_words: int = 300, overlap_words: int = 50) -> list[str]:
    """Word-count chunking with overlap — simpler than a token-accurate
    splitter (no tokenizer dependency) and close enough for embedding-sized
    chunks at this project's scale."""
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text.strip()]

    step = chunk_words - overlap_words
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + chunk_words]))
        if start + chunk_words >= len(words):
            break
    return chunks
