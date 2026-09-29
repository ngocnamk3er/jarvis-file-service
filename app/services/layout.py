"""Text with the page and the box it came from, instead of a flat string.

`extraction.extract_text` answers "what does this document say". This
module answers "and where on the page does it say it", which is what a
retrieval hit needs to be shown rather than merely quoted.

Two sources feed one shape:

  * a PDF page with a text layer — pdfplumber reports a box per word, so
    nothing has to be inferred;
  * a page or image with none — PaddleOCR reads it and reports a box per
    detected line.

Both land as `Span`, so everything downstream stays unaware of which one
produced it. The decision is per *page*, never per file: real documents
mix the two, a typed contract with a scanned signature page being the
ordinary case, and OCR-ing a page that already has good text is both
slower and worse.

**Coordinates are fractions of the page, not points or pixels.** A PDF
text layer is measured in points from the top-left; OCR is measured in
pixels of whatever raster we happened to render. Storing either raw would
make every consumer carry the page size and the render scale around to
interpret it. Normalising once, here, means a viewer can multiply by its
own rendered width and height and be done.
"""

from dataclasses import dataclass

from app.core.config import settings

# Rendering for OCR. 2x (~144 DPI) is the usual floor for reliable
# recognition of body text; higher mostly costs time, and the boxes are
# normalised straight afterwards so it does not affect what is stored.
_OCR_RENDER_SCALE = 2.0
# Below this many words, a PDF page is treated as having no text layer.
# Not zero: a scanned page often carries a stray header or a page number
# in real text, which would otherwise pass as a full text layer and leave
# the actual content unindexed.
_MIN_WORDS_FOR_TEXT_LAYER = 8


@dataclass(frozen=True)
class Span:
    """One piece of text and where it sits.

    `bbox` is (x0, top, x1, bottom), each 0..1 relative to the page box,
    origin top-left. `page` is 1-based, matching how a reader counts and
    how every PDF viewer numbers pages.
    """

    text: str
    page: int
    bbox: tuple[float, float, float, float]


def _norm(x0: float, top: float, x1: float, bottom: float, w: float, h: float
          ) -> tuple[float, float, float, float]:
    """Clamp to 0..1 as well as scale: pdfplumber occasionally reports a
    word a hair outside the mediabox, and a viewer drawing at -0.01 would
    put the highlight off-canvas."""
    if w <= 0 or h <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        min(max(x0 / w, 0.0), 1.0),
        min(max(top / h, 0.0), 1.0),
        min(max(x1 / w, 0.0), 1.0),
        min(max(bottom / h, 0.0), 1.0),
    )


def ocr_available() -> bool:
    """Whether OCR can run at all.

    PaddleOCR and its runtime are a large dependency, so an image can
    legitimately ship without them; the setting then keeps ingest on the
    text-layer path instead of failing a file it cannot read. Import is
    checked rather than assumed because the setting being on does not make
    the wheels present.
    """
    if not settings.OCR_ENABLED:
        return False
    try:
        import paddleocr  # noqa: F401
    except Exception:
        return False
    return True


_ocr_engine = None


def _get_ocr():
    """One PaddleOCR instance for the process — construction loads model
    weights off disk and costs seconds, which is not something to repeat
    per page."""
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR

        _ocr_engine = PaddleOCR(use_angle_cls=True, lang=settings.OCR_LANG, show_log=False)
    return _ocr_engine


def _ocr_image(image, page: int) -> list[Span]:
    """Spans for one rendered page or image.

    PaddleOCR returns a quadrilateral per line rather than a rectangle,
    because text can sit at an angle; the enclosing rectangle is what a
    highlight overlay can actually draw, so take its extremes.
    """
    import numpy as np

    arr = np.array(image.convert("RGB"))
    h, w = arr.shape[0], arr.shape[1]
    raw = _get_ocr().ocr(arr, cls=True)
    if not raw or raw[0] is None:
        return []

    spans = []
    for quad, (text, confidence) in raw[0]:
        if not text.strip() or confidence < settings.OCR_MIN_CONFIDENCE:
            continue
        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]
        spans.append(Span(text.strip(), page, _norm(min(xs), min(ys), max(xs), max(ys), w, h)))
    return spans


def _render_page(pdf_bytes: bytes, page_index: int):
    """One PDF page as a PIL image. pypdfium2 ships as a self-contained
    wheel, so this adds no system packages to the image the way poppler
    (pdf2image) or ImageMagick (wand) would."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf_bytes)
    try:
        return doc[page_index].render(scale=_OCR_RENDER_SCALE).to_pil()
    finally:
        doc.close()


def _pdf_spans(content: bytes) -> list[Span]:
    import io

    import pdfplumber

    spans: list[Span] = []
    try:
        pdf_context = pdfplumber.open(io.BytesIO(content))
    except Exception:
        # Returning empty rather than raising, so extract_spans yields None
        # and ingest falls back to flat extraction. pdfminer (under
        # pdfplumber) rejects some malformed files that pypdf still reads,
        # and a file that only loses its coordinates is better than one
        # marked failed and left unsearchable.
        return []
    with pdf_context as pdf:
        for index, page in enumerate(pdf.pages):
            number = index + 1
            width, height = page.width, page.height
            words = page.extract_words() or []
            if len(words) >= _MIN_WORDS_FOR_TEXT_LAYER:
                spans.extend(
                    Span(w["text"], number,
                         _norm(w["x0"], w["top"], w["x1"], w["bottom"], width, height))
                    for w in words if w["text"].strip()
                )
                continue
            if not ocr_available():
                # Keep whatever the thin text layer held rather than
                # dropping the page entirely — a page number is poor
                # content but it is not nothing, and silently losing pages
                # is worse than indexing them badly.
                spans.extend(
                    Span(w["text"], number,
                         _norm(w["x0"], w["top"], w["x1"], w["bottom"], width, height))
                    for w in words if w["text"].strip()
                )
                continue
            try:
                spans.extend(_ocr_image(_render_page(content, index), number))
            except Exception:
                # One unreadable page must not cost the rest of the
                # document its indexing.
                continue
    return spans


def _image_spans(content: bytes) -> list[Span]:
    import io

    from PIL import Image

    if not ocr_available():
        return []
    try:
        with Image.open(io.BytesIO(content)) as image:
            return _ocr_image(image, 1)
    except Exception:
        # Same reasoning as the PDF path: an unreadable image costs its
        # own indexing, never the upload.
        return []


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


def extract_spans(filename: str, mime_type: str | None, content: bytes) -> list[Span] | None:
    """Positioned text, or None for a format this cannot place.

    None is not failure — it means "no coordinates available", and ingest
    falls back to flat extraction so the file is still searchable, just
    without page and box. A .docx has no fixed pagination until something
    lays it out, so it takes that path by design rather than by omission.
    """
    extension = filename[filename.rfind(".") :].lower() if "." in filename else ""
    mime_type = mime_type or ""

    if extension == ".pdf" or mime_type == "application/pdf":
        return _pdf_spans(content) or None
    if extension in _IMAGE_EXTENSIONS or mime_type.startswith("image/"):
        return _image_spans(content) or None
    return None
