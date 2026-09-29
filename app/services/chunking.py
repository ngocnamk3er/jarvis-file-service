from dataclasses import dataclass, field

from app.services.layout import Span

# Rows this close together vertically are one line of text. Words on a
# line rarely agree on `top` to the pixel — descenders, superscripts and
# mixed font sizes all shift it — so grouping on exact equality would
# leave a box per word and a payload several times larger than the text.
_LINE_TOLERANCE = 0.006


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


@dataclass
class Chunk:
    """One embedded unit, and everywhere it appears on the page.

    `boxes` is a list rather than one rectangle because a 300-word chunk
    runs over many lines and often over a column break or a page break.
    The rectangle enclosing all of that would cover most of the page and
    highlight nothing useful, so each line keeps its own box.

    `char_start`/`char_end` index into the same flat string ingest stores
    as `extracted_text`, which is what read_file returns — so a hit can be
    located in the text a caller already has, without re-deriving it.
    """

    text: str
    page_start: int
    page_end: int
    boxes: list[tuple[int, tuple[float, float, float, float]]] = field(default_factory=list)
    char_start: int = 0
    char_end: int = 0


def spans_to_text(spans: list[Span]) -> str:
    """The flat string for a list of spans.

    The single space here is the contract `char_start`/`char_end` are
    offsets into, so ingest must store exactly this and nothing prettier.
    """
    return " ".join(s.text for s in spans)


def _merge_boxes(spans: list[Span]) -> list[tuple[int, tuple[float, float, float, float]]]:
    """One box per line of text, in reading order.

    Spans arrive per word (text layer) or per line (OCR); either way the
    useful highlight unit is the line, so words on the same page at the
    same height merge into the rectangle enclosing them.
    """
    boxes: list[tuple[int, tuple[float, float, float, float]]] = []
    for span in spans:
        x0, top, x1, bottom = span.bbox
        if boxes:
            last_page, (lx0, ltop, lx1, lbottom) = boxes[-1]
            if last_page == span.page and abs(ltop - top) <= _LINE_TOLERANCE:
                boxes[-1] = (
                    last_page,
                    (min(lx0, x0), min(ltop, top), max(lx1, x1), max(lbottom, bottom)),
                )
                continue
        boxes.append((span.page, (x0, top, x1, bottom)))
    return boxes


def chunk_spans(spans: list[Span], chunk_words: int = 300, overlap_words: int = 50) -> list[Chunk]:
    """Chunk positioned text, keeping each chunk's page range and boxes.

    Windowing is over spans rather than over words of the flat string, so
    a chunk boundary always falls between spans and every chunk maps back
    to an exact set of boxes. For a PDF text layer a span is one word, so
    this counts the same units `chunk_text` does; for OCR a span is a
    line, which makes the windows coarser — accepted, because a line is
    also the smallest thing OCR can place.
    """
    if not spans:
        return []

    # Offsets into spans_to_text()'s output, computed once so each chunk
    # can name its slice without re-joining anything.
    starts: list[int] = []
    cursor = 0
    for span in spans:
        starts.append(cursor)
        cursor += len(span.text) + 1  # +1 for the joining space
    full_length = max(cursor - 1, 0)

    step = max(chunk_words - overlap_words, 1)
    chunks: list[Chunk] = []
    for start in range(0, len(spans), step):
        window = spans[start : start + chunk_words]
        if not window:
            break
        end_index = start + len(window) - 1
        pages = [s.page for s in window]
        chunks.append(
            Chunk(
                text=spans_to_text(window),
                page_start=min(pages),
                page_end=max(pages),
                boxes=_merge_boxes(window),
                char_start=starts[start],
                char_end=min(starts[end_index] + len(spans[end_index].text), full_length),
            )
        )
        if start + chunk_words >= len(spans):
            break
    return chunks
