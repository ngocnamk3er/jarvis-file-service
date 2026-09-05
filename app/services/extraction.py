"""Best-effort plaintext extraction, run once per upload (see services/ingest.py).
Powers grep_files directly (stored in FileNode.extracted_text) and is the
chunking source for embeddings in phase 3. Returns None for anything not
worth indexing (binary/unsupported) — indexing_status becomes "skipped"."""

import io

_TEXT_EXTENSIONS = {
    ".txt", ".md", ".csv", ".json", ".py", ".js", ".ts", ".tsx", ".jsx",
    ".yaml", ".yml", ".html", ".css", ".sh", ".sql", ".log", ".toml",
}


def extract_text(filename: str, mime_type: str | None, content: bytes) -> str | None:
    ext = _ext(filename)
    mime_type = mime_type or ""

    if ext == ".pdf" or mime_type == "application/pdf":
        return _extract_pdf(content)
    if ext == ".docx" or "wordprocessingml.document" in mime_type:
        return _extract_docx(content)
    if ext in _TEXT_EXTENSIONS or mime_type.startswith("text/"):
        return _decode_text(content)
    return None


def _ext(filename: str) -> str:
    i = filename.rfind(".")
    return filename[i:].lower() if i != -1 else ""


def _decode_text(content: bytes) -> str | None:
    return content.decode("utf-8", errors="replace").strip() or None


def _extract_pdf(content: bytes) -> str | None:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(content))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        return text.strip() or None
    except Exception:
        # Corrupt/encrypted/scanned-image PDF — treat as unextractable rather
        # than failing the whole upload.
        return None


def _extract_docx(content: bytes) -> str | None:
    import docx

    try:
        document = docx.Document(io.BytesIO(content))
        text = "\n".join(p.text for p in document.paragraphs)
        return text.strip() or None
    except Exception:
        return None
