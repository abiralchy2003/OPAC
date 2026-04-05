"""Split long text into overlapping chunks for the LLM context window."""
from typing import List
from config.settings import CHUNK_MAX_CHARS, CHUNK_OVERLAP


def chunk_text(text: str,
               max_chars: int = CHUNK_MAX_CHARS,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into chunks of at most `max_chars` characters,
    with `overlap` characters of context carried between chunks.
    Tries to split on sentence/paragraph boundaries when possible.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunks.append(text[start:])
            break

        # Try to cut at a paragraph break first, then sentence, then word
        cut = _find_split(text, end, end - overlap)
        chunks.append(text[start:cut])
        start = cut - overlap  # carry overlap forward
        if start < 0:
            start = 0

    return [c.strip() for c in chunks if c.strip()]


def _find_split(text: str, ideal: int, minimum: int) -> int:
    """Find a natural split point near `ideal` but no earlier than `minimum`."""
    # Paragraph break
    pos = text.rfind("\n\n", minimum, ideal)
    if pos != -1:
        return pos + 2

    # Single newline
    pos = text.rfind("\n", minimum, ideal)
    if pos != -1:
        return pos + 1

    # Sentence end
    for punct in (". ", "! ", "? "):
        pos = text.rfind(punct, minimum, ideal)
        if pos != -1:
            return pos + 2

    # Word boundary
    pos = text.rfind(" ", minimum, ideal)
    if pos != -1:
        return pos + 1

    return ideal  # hard cut
