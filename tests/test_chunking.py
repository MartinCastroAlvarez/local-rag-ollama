from __future__ import annotations

from pathlib import Path

import pytest

from local_rag.chunking import _split_long
from local_rag.chunking import _tail
from local_rag.chunking import chunk_document
from local_rag.loaders import Document
from local_rag.loaders import Section


def doc(*sections: Section, title: str = "Doc") -> Document:
    return Document(
        path=Path("/x"), source="x.md", kind="markdown", title=title, sections=list(sections)
    )


def test_chunks_never_cross_sections() -> None:
    d = doc(Section(("A",), "alpha one.\n\nalpha two."), Section(("B",), "beta."))
    chunks = chunk_document(d, chunk_chars=500, overlap_chars=50)
    assert [(c.headings, c.text) for c in chunks] == [
        (("A",), "alpha one.\n\nalpha two."),
        (("B",), "beta."),
    ]
    assert [c.ordinal for c in chunks] == [0, 1]


def test_packing_respects_size_and_overlaps() -> None:
    paragraphs = [f"Sentence number {i} is here. It has two parts." for i in range(30)]
    d = doc(Section(("S",), "\n\n".join(paragraphs)))
    chunks = chunk_document(d, chunk_chars=300, overlap_chars=60)
    assert len(chunks) > 3
    assert all(len(c.text) <= 300 for c in chunks)
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        carried = nxt.text.split(" Sentence number")[0]
        assert carried and carried in prev.text  # the next chunk starts with the previous tail


def test_long_paragraph_splits_on_sentences_then_words() -> None:
    sentence = "Word " + "word " * 100
    pieces = _split_long(f"Short one. {sentence.strip()}. End.", 120)
    assert all(len(p) <= 120 for p in pieces)
    assert pieces[0] == "Short one."
    huge = _split_long("x" * 500, 200)
    assert [len(p) for p in huge] == [200, 200, 100]


def test_long_word_after_text_flushes_buffer() -> None:
    pieces = _split_long("hello " + "y" * 250, 100)
    assert pieces[0] == "hello" and all(len(p) <= 100 for p in pieces)


def test_tail_cuts_at_boundaries() -> None:
    assert _tail("", 10) == ""
    assert _tail("short", 10) == "short"
    assert _tail("First sentence here. Second sentence is longer than that one.", 50).startswith(
        "Second"
    )
    assert _tail("aaaa bbbb cccc dddd", 9) == "dddd"
    assert _tail("abcdefghij", 4) == "ghij"
    assert _tail("abc", 0) == ""


def test_breadcrumb_and_indexed_text() -> None:
    d = doc(Section(("Doc", "Article 1"), "body"), title="Doc")
    chunk = chunk_document(d, chunk_chars=300, overlap_chars=0)[0]
    assert chunk.breadcrumb == "Doc > Article 1"
    assert chunk.indexed_text.startswith("Doc > Article 1\n\nbody")
    assert len(chunk.digest) == 16
    pdf = Document(Path("/p"), "p.pdf", "pdf", "P", [Section((), "page text", page=3)])
    assert chunk_document(pdf, chunk_chars=300, overlap_chars=0)[0].breadcrumb == "P (page 3)"


def test_overlap_must_be_smaller() -> None:
    with pytest.raises(ValueError):
        chunk_document(doc(Section((), "x")), chunk_chars=100, overlap_chars=100)


def test_line_wraps_are_flattened() -> None:
    d = doc(Section((), "a hard\nwrapped\nline"))
    assert chunk_document(d, chunk_chars=300, overlap_chars=0)[0].text == "a hard wrapped line"
