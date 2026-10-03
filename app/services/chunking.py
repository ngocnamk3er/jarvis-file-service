import re
from dataclasses import dataclass, field

from app.services.layout import Span

# Rows this close together vertically are one line of text. Words on a
# line rarely agree on `top` to the pixel — descenders, superscripts and
# mixed font sizes all shift it — so grouping on exact equality would
# leave a box per word and a payload several times larger than the text.
_LINE_TOLERANCE = 0.006


@dataclass
class Chunk:
    """One embedded unit, and everywhere it appears on the page.

    `boxes` is a list rather than one rectangle because a 300-word chunk
    runs over many lines and often over a column break or a page break.
    The rectangle enclosing all of that would cover most of the page and
    highlight nothing useful, so each line keeps its own box.

    Each box carries a third element, (local_start, local_end): that
    line's char range within `text` (this chunk's own, not the whole
    document's). Exact-match search uses it to narrow `boxes` down to
    just the line(s) a hit actually matched — a chunk can be the entire
    text of a short document (see a 2026-10-03 trace: a 2-page PDF under
    the 300-word chunk threshold highlighted almost in full, because the
    whole thing was one chunk), and exact search knows precisely which
    substring matched, unlike semantic search's whole-chunk relevance.

    `char_start`/`char_end` index into the same flat string ingest stores
    as `extracted_text`, which is what read_file returns — so a hit can be
    located in the text a caller already has, without re-deriving it.
    """

    text: str
    page_start: int
    page_end: int
    boxes: list[tuple[int, tuple[float, float, float, float], tuple[int, int]]] = field(
        default_factory=list
    )
    char_start: int = 0
    char_end: int = 0


def spans_to_text(spans: list[Span]) -> str:
    """The flat string for a list of spans.

    The single space here is the contract `char_start`/`char_end` are
    offsets into, so ingest must store exactly this and nothing prettier.
    """
    return " ".join(s.text for s in spans)


def chunk_text(text: str, chunk_words: int = 300, overlap_words: int = 50) -> list[Chunk]:
    """Word-count chunking with overlap, for a format with no page geometry
    to report (.txt, .docx, a PDF that yielded no spans). Mirrors
    chunk_spans' shape — page 0 and no boxes, since there is nothing to
    point a viewer at beyond the text itself — so ingest can pick between
    the two with a plain `if spans else` and never build a Chunk by hand.

    Bug fixed here: this used to return `list[str]`, and the caller built
    `Chunk(text=t, page_start=0, page_end=0)` with no char_start/char_end —
    every chunk landed with those at their dataclass default of 0, so
    TextExcerpt's `text.slice(char_start, char_end)` highlighted nothing
    for every plain-text file. Caught 2026-09-30 via a live citation whose
    char_start/char_end were both 0 despite matching real content.

    Offsets come from `re.finditer(r"\\S+", text)` rather than
    `text.split()`, which discards exactly the position information an
    index into the ORIGINAL string needs — `text.split()` only gives back
    the words, and re-joining them with `" ".join(...)` produces a string
    that no longer matches `text` character-for-character (runs of
    whitespace collapse to one space each). Slicing `text[start:end]`
    instead keeps `chunk.text == text[chunk.char_start:chunk.char_end]`
    exactly, the same invariant chunk_spans already gives for the
    span-based path.
    """
    words = list(re.finditer(r"\S+", text))
    if not words:
        return []

    def _window(matches: list[re.Match]) -> Chunk:
        start, end = matches[0].start(), matches[-1].end()
        return Chunk(text=text[start:end], page_start=0, page_end=0, char_start=start, char_end=end)

    if len(words) <= chunk_words:
        return [_window(words)]

    step = chunk_words - overlap_words
    chunks = []
    for i in range(0, len(words), step):
        window = words[i : i + chunk_words]
        chunks.append(_window(window))
        if i + chunk_words >= len(words):
            break
    return chunks


def _merge_boxes(
    spans: list[Span], local_starts: list[int]
) -> list[tuple[int, tuple[float, float, float, float], tuple[int, int]]]:
    """One box per line of text, in reading order, each paired with the
    char range (local to this chunk) the line covers.

    Spans arrive per word (text layer) or per line (OCR); either way the
    useful highlight unit is the line, so words on the same page at the
    same height merge into the rectangle enclosing them — and their char
    ranges merge the same way, into the span from the first word's start
    to the last word's end.

    `local_starts[i]` is where `spans[i].text` begins in `spans_to_text
    (spans)` — the caller computes it because that depends on the whole
    window, not any one span.
    """
    boxes: list[tuple[int, tuple[float, float, float, float], tuple[int, int]]] = []
    for span, local_start in zip(spans, local_starts):
        x0, top, x1, bottom = span.bbox
        local_end = local_start + len(span.text)
        if boxes:
            last_page, (lx0, ltop, lx1, lbottom), (cs, _ce) = boxes[-1]
            if last_page == span.page and abs(ltop - top) <= _LINE_TOLERANCE:
                boxes[-1] = (
                    last_page,
                    (min(lx0, x0), min(ltop, top), max(lx1, x1), max(lbottom, bottom)),
                    (cs, local_end),
                )
                continue
        boxes.append((span.page, (x0, top, x1, bottom), (local_start, local_end)))
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
        # Local to this window, not the document: _merge_boxes' char
        # ranges are offsets into `chunk.text` itself, the same frame a
        # search hit's matched substring is found in.
        window_starts: list[int] = []
        cursor = 0
        for s in window:
            window_starts.append(cursor)
            cursor += len(s.text) + 1
        chunks.append(
            Chunk(
                text=spans_to_text(window),
                page_start=min(pages),
                page_end=max(pages),
                boxes=_merge_boxes(window, window_starts),
                char_start=starts[start],
                char_end=min(starts[end_index] + len(spans[end_index].text), full_length),
            )
        )
        if start + chunk_words >= len(spans):
            break
    return chunks
