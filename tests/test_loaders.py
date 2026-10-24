from __future__ import annotations

import os
from pathlib import Path

import pytest

from local_rag.loaders import IngestError
from local_rag.loaders import clean
from local_rag.loaders import discover
from local_rag.loaders import load
from local_rag.loaders import parse_html
from local_rag.loaders import parse_markdown
from local_rag.loaders import parse_pdf
from local_rag.loaders import parse_text
from local_rag.loaders import read_bytes
from tests.conftest import make_pdf

MB = 1024 * 1024


def test_markdown_headings_nest_and_skip_code_fences() -> None:
    text = (
        "---\ntitle: x\n---\n# Title\n\nintro\n\n## A\n\nalpha\n\n```\n# not a heading\n```\n\n"
        "### A1\n\ndeep\n\n## B\n\nbeta\n\nSetext\n------\n\nunder setext\n"
    )
    title, sections = parse_markdown(text, "fallback")
    assert title == "Title"
    paths = [s.headings for s in sections]
    assert paths == [
        ("Title",),
        ("Title", "A"),
        ("Title", "A", "A1"),
        ("Title", "B"),
        ("Title", "Setext"),
    ]
    assert "# not a heading" in sections[1].text
    assert "title: x" not in "".join(s.text for s in sections)


def test_markdown_without_h1_uses_fallback_title() -> None:
    title, sections = parse_markdown("## Only h2\n\nbody", "fallback")
    assert title == "fallback"
    assert sections[0].headings == ("Only h2",)


def test_html_drops_scripts_and_styles_and_keeps_headings() -> None:
    html = (
        "<html><head><title>T &amp; co</title><style>p{}</style></head><body>"
        "<script>steal()</script><h1>Top</h1><p>one<br>two</p><h2>Sub</h2>"
        "<table><tr><td>a</td><td>b</td></tr></table><img src=x /><hr/></body></html>"
    )
    title, sections = parse_html(html, "fallback")
    assert title == "T & co"
    joined = " ".join(s.text for s in sections)
    assert "steal" not in joined and "p{}" not in joined
    assert sections[0].headings == ("Top",)
    assert sections[0].text == "one\ntwo"
    assert sections[1].headings == ("Top", "Sub")


def test_html_title_falls_back_to_first_h1() -> None:
    title, _ = parse_html("<h1>Heading</h1><p>x</p>", "fallback")
    assert title == "Heading"


def test_text_heading_heuristic() -> None:
    text = "MY BOOK\n\nCHAPTER ONE\n\nIt was a dark night.\n\nII. THE NEXT PART\n\nMore text.\n"
    title, sections = parse_text(text, "fallback")
    assert title == "My Book"
    assert [s.headings for s in sections] == [("CHAPTER ONE",), ("II. THE NEXT PART",)]


def test_text_lowercase_lines_are_not_headings() -> None:
    _, sections = parse_text("intro line\n\nnot a heading\n\nbody.", "t")
    assert len(sections) == 1 and sections[0].headings == ()


def test_pdf_pages_and_title() -> None:
    title, sections = parse_pdf(make_pdf(["first page", "second page"], title="Doc"), "x")
    assert title == "Doc"
    assert [s.page for s in sections] == [1, 2]
    assert "second page" in sections[1].text


def test_pdf_without_title_uses_fallback() -> None:
    title, _ = parse_pdf(make_pdf(["p"]), "fallback")
    assert title == "fallback"


def test_garbage_pdf_is_an_ingest_error() -> None:
    with pytest.raises(IngestError):
        parse_pdf(b"%PDF-1.4 this is not a pdf", "x")


def test_clean_strips_control_characters() -> None:
    assert clean("a\x00b\r\nc\x1b\n\n\n\nd  ") == "ab\nc\n\nd"


def test_load_dispatches_on_suffix(corpus: Path) -> None:
    md = load(corpus / "guides" / "garden.md", "guides/garden.md", max_bytes=MB)
    assert md.kind == "markdown" and md.title == "Gardening"
    pdf = load(corpus / "astronomy.pdf", "astronomy.pdf", max_bytes=MB)
    assert pdf.kind == "pdf" and "Great Red Spot" in pdf.text
    with pytest.raises(IngestError):
        load(corpus / "x.exe", "x.exe", max_bytes=MB)


# --------------------------------------------------------------------------- #
# Discovery / path traversal
# --------------------------------------------------------------------------- #
def test_discover_lists_supported_files_relative_to_root(corpus: Path) -> None:
    (corpus / "notes.exe").write_bytes(b"MZ")
    (corpus / ".env").write_text("SECRET=1")
    (corpus / ".git").mkdir()
    (corpus / ".git" / "README.md").write_text("# hidden")
    sources = sorted(s for _, s in discover(corpus, max_bytes=MB))
    assert sources == ["astronomy.pdf", "bikes.html", "bread.txt", "guides/garden.md"]


def test_symlink_outside_root_is_skipped(tmp_path: Path, corpus: Path) -> None:
    secret = tmp_path / "secret.md"
    secret.write_text("# top secret")
    os.symlink(secret, corpus / "leak.md")
    outside_dir = tmp_path / "elsewhere"
    outside_dir.mkdir()
    (outside_dir / "x.md").write_text("# outside")
    os.symlink(outside_dir, corpus / "linked-dir")
    sources = {s for _, s in discover(corpus, max_bytes=MB)}
    assert "leak.md" not in sources
    assert not any(s.startswith("linked-dir") for s in sources)


def test_symlink_inside_root_is_followed_once(corpus: Path) -> None:
    os.symlink(corpus / "bread.txt", corpus / "alias.txt")
    found = dict((s, p) for p, s in discover(corpus, max_bytes=MB))
    assert found["alias.txt"] == (corpus / "bread.txt").resolve()


def test_dotdot_root_is_resolved_and_contained(tmp_path: Path, corpus: Path) -> None:
    sneaky = corpus / "guides" / ".." / ".."
    sources = {s for _, s in discover(sneaky, max_bytes=MB)}
    # The root resolves to tmp_path; sources are relative to it, never "../".
    assert all(not s.startswith("..") for s in sources)


def test_broken_symlink_and_oversize_files_are_skipped(corpus: Path) -> None:
    os.symlink(corpus / "missing.md", corpus / "broken.md")
    (corpus / "big.txt").write_text("x" * 5000)
    sources = {s for _, s in discover(corpus, max_bytes=4000)}
    assert "broken.md" not in sources and "big.txt" not in sources


def test_single_file_root(corpus: Path) -> None:
    assert [s for _, s in discover(corpus / "bread.txt", max_bytes=MB)] == ["bread.txt"]
    with pytest.raises(IngestError, match="unsupported"):
        list(discover(_exe(corpus), max_bytes=MB))
    with pytest.raises(IngestError, match="exceeds"):
        list(discover(corpus / "bread.txt", max_bytes=3))


def _exe(corpus: Path) -> Path:
    path = corpus / "tool.exe"
    path.write_bytes(b"MZ")
    return path


def test_missing_root_and_special_files(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="not found"):
        list(discover(tmp_path / "nope", max_bytes=MB))
    fifo = tmp_path / "pipe.md"
    os.mkfifo(fifo)
    with pytest.raises(IngestError, match="not a regular file"):
        list(discover(fifo, max_bytes=MB))


def test_read_bytes_refuses_a_symlink_swapped_in(tmp_path: Path) -> None:
    target = tmp_path / "t.md"
    target.write_text("x")
    link = tmp_path / "l.md"
    os.symlink(target, link)
    with pytest.raises(OSError):
        read_bytes(link, MB)


def test_read_bytes_limit(tmp_path: Path) -> None:
    path = tmp_path / "f.txt"
    path.write_text("abcdef")
    with pytest.raises(IngestError, match="grew"):
        read_bytes(path, 3)
