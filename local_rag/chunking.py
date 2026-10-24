"""Structure-aware chunking.

A chunk never crosses a section (heading) or PDF page boundary. Inside a section,
paragraphs are packed greedily up to ``chunk_chars``; a paragraph that is too long is split
at sentence boundaries, and a sentence that is still too long at word boundaries. Each new
chunk in a section starts with up to ``overlap_chars`` of the previous chunk's tail (cut at a
sentence or word boundary), so a fact that spans the cut is retrievable from either side.

The heading breadcrumb ("Article 1 > Section 8") is stored with the chunk and prepended to
the text that gets embedded and BM25-indexed, so a query naming a section finds it.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from local_rag.loaders import Document

_SENTENCE = re.compile(r"(?<=[.!?;:])[\"'”’)\]]*\s+(?=[\"'“‘(\[]*[A-Z0-9_])")


@dataclass(frozen=True)
class Chunk:
    source: str
    ordinal: int
    title: str
    headings: tuple[str, ...]
    page: int | None
    text: str

    @property
    def breadcrumb(self) -> str:
        parts = [self.title, *[h for h in self.headings if h != self.title]]
        crumb = " > ".join(p for p in parts if p)
        return f"{crumb} (page {self.page})" if self.page is not None else crumb

    @property
    def indexed_text(self) -> str:
        """What gets embedded and BM25-indexed: breadcrumb + body."""
        return f"{self.breadcrumb}\n\n{self.text}"

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.indexed_text.encode("utf-8")).hexdigest()[:16]


def _flatten(paragraph: str) -> str:
    return " ".join(paragraph.split())


def _split_long(unit: str, limit: int) -> list[str]:
    if len(unit) <= limit:
        return [unit]
    out: list[str] = []
    for sentence in _SENTENCE.split(unit):
        if len(sentence) <= limit:
            out.append(sentence)
            continue
        words = sentence.split(" ")
        buf = ""
        for word in words:
            while len(word) > limit:  # a pathological "word" (URL, base64 blob)
                if buf:
                    out.append(buf)
                    buf = ""
                out.append(word[:limit])
                word = word[limit:]
            candidate = f"{buf} {word}" if buf else word
            if len(candidate) > limit and buf:
                out.append(buf)
                buf = word
            else:
                buf = candidate
        if buf:
            out.append(buf)
    return out


def _tail(text: str, size: int) -> str:
    """The last ``size`` characters of ``text``, starting at a sentence or word boundary."""
    if size <= 0 or not text:
        return ""
    if len(text) <= size:
        return text
    window = text[-size:]
    for match in _SENTENCE.finditer(window):
        if match.end() < len(window) - 20:
            return window[match.end() :]
    space = window.find(" ")
    return window[space + 1 :] if space != -1 else window


def _pack(units: list[str], limit: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for unit in units:
        sep = "\n\n" if current else ""
        if current and len(current) + len(sep) + len(unit) > limit:
            chunks.append(current)
            carry = _tail(current, overlap)
            current = f"{carry} {unit}" if carry and len(carry) + 1 + len(unit) <= limit else unit
            continue
        current = f"{current}{sep}{unit}"
    if current:
        chunks.append(current)
    return chunks


def chunk_document(doc: Document, *, chunk_chars: int, overlap_chars: int) -> list[Chunk]:
    if overlap_chars >= chunk_chars:
        raise ValueError("overlap must be smaller than the chunk size")
    chunks: list[Chunk] = []
    for section in doc.sections:
        paragraphs = [_flatten(p) for p in re.split(r"\n\s*\n", section.text)]
        units = [u for p in paragraphs if p for u in _split_long(p, chunk_chars)]
        for body in _pack(units, chunk_chars, overlap_chars):
            chunks.append(
                Chunk(
                    source=doc.source,
                    ordinal=len(chunks),
                    title=doc.title,
                    headings=section.headings,
                    page=section.page,
                    text=body,
                )
            )
    return chunks
