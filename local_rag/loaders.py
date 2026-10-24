"""Find files under an ingest root (safely) and turn them into structured documents.

Every loader returns a :class:`Document`: a title plus a list of :class:`Section` s, each
with the heading path it sits under (``("Article 1", "Section 8")``) and, for PDFs, the page.
The chunker keeps those boundaries, so a chunk never straddles two sections.
"""

from __future__ import annotations

import io
import logging
import os
import re
import stat
from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import field
from html.parser import HTMLParser
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

log = logging.getLogger(__name__)

SUFFIXES = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".text": "text",
    ".rst": "text",
    ".html": "html",
    ".htm": "html",
    ".pdf": "pdf",
}
MAX_PDF_PAGES = 2000
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class IngestError(ValueError):
    """A path cannot be ingested (outside the root, wrong type, too large, unreadable)."""


@dataclass(frozen=True)
class Section:
    headings: tuple[str, ...]
    text: str
    page: int | None = None


@dataclass(frozen=True)
class Document:
    path: Path
    """Absolute, resolved path of the file."""
    source: str
    """Path relative to the ingest root, used for display and citations."""
    kind: str
    title: str
    sections: list[Section] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(s.text for s in self.sections)


def clean(text: str) -> str:
    """Drop control characters and trailing whitespace; keep paragraph breaks."""
    text = _CONTROL.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))
    lines = [line.rstrip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _inside(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


def discover(root: Path | str, *, max_bytes: int) -> Iterator[tuple[Path, str]]:
    """Yield ``(resolved_path, source)`` for every ingestible file under ``root``.

    Guards: symlinks (files or directories) that resolve outside ``root`` are skipped,
    directory symlinks are never followed, hidden entries (``.git``, ``.env``) are skipped,
    only regular files with a known suffix and at most ``max_bytes`` are returned.
    """
    root_path = Path(root).expanduser()
    try:
        base = root_path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise IngestError(f"{root_path}: not found") from exc
    if base.is_file():
        if base.suffix.lower() not in SUFFIXES:
            raise IngestError(f"{root_path}: unsupported file type {base.suffix or '(none)'}")
        _check_size(base, max_bytes)
        yield base, base.name
        return
    if not base.is_dir():
        raise IngestError(f"{root_path}: not a regular file or directory")

    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        current = Path(dirpath)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if name.startswith(".") or Path(name).suffix.lower() not in SUFFIXES:
                continue
            candidate = current / name
            try:
                resolved = candidate.resolve(strict=True)
            except (OSError, RuntimeError):
                log.warning("skip %s: broken link", candidate)
                continue
            if not _inside(resolved, base):
                log.warning("skip %s: resolves outside the ingest root", candidate)
                continue
            try:
                _check_size(resolved, max_bytes)
            except IngestError as exc:
                log.warning("skip %s", exc)
                continue
            yield resolved, candidate.relative_to(base).as_posix()


def _check_size(path: Path, max_bytes: int) -> None:
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise IngestError(f"{path}: not a regular file")
    if info.st_size > max_bytes:
        raise IngestError(f"{path}: {info.st_size} bytes exceeds the {max_bytes}-byte limit")


def read_bytes(path: Path, max_bytes: int) -> bytes:
    """Read a resolved path without following a symlink swapped in after discovery."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as fh:
        data = fh.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise IngestError(f"{path}: grew past the {max_bytes}-byte limit")
    return data


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load(path: Path, source: str, *, max_bytes: int, data: bytes | None = None) -> Document:
    """Parse one file. ``data`` lets the caller pass bytes it already read (and hashed)."""
    kind = SUFFIXES.get(path.suffix.lower())
    if kind is None:
        raise IngestError(f"{path}: unsupported file type")
    if data is None:
        data = read_bytes(path, max_bytes)
    title_default = Path(source).stem.replace("-", " ").replace("_", " ")
    if kind == "pdf":
        title, sections = parse_pdf(data, title_default)
    else:
        text = data.decode("utf-8", errors="replace")
        if kind == "markdown":
            title, sections = parse_markdown(text, title_default)
        elif kind == "html":
            title, sections = parse_html(text, title_default)
        else:
            title, sections = parse_text(text, title_default)
    sections = [s for s in sections if s.text.strip()]
    return Document(path=path, source=source, kind=kind, title=title, sections=sections)


def _nest(stack: list[tuple[int, str]], level: int, title: str) -> tuple[str, ...]:
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, title))
    return tuple(t for _, t in stack)


_ATX = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


def parse_markdown(text: str, title: str) -> tuple[str, list[Section]]:
    text = clean(text)
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            text = text[end + 4 :].lstrip("\n")
    lines = text.split("\n")
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    path: tuple[str, ...] = ()
    buf: list[str] = []
    in_fence = False
    doc_title: str | None = None

    def flush() -> None:
        body = clean("\n".join(buf))
        if body:
            sections.append(Section(path, body))
        buf.clear()

    for i, line in enumerate(lines):
        if _FENCE.match(line):
            in_fence = not in_fence
            buf.append(line)
            continue
        heading: tuple[int, str] | None = None
        if not in_fence:
            m = _ATX.match(line)
            if m:
                heading = (len(m.group(1)), m.group(2).strip())
            elif (
                i + 1 < len(lines)
                and line.strip()
                and re.fullmatch(r"=+|-+", lines[i + 1].strip() or "x")
                and len(lines[i + 1].strip()) >= 3
            ):
                heading = (1 if lines[i + 1].strip()[0] == "=" else 2, line.strip())
            elif re.fullmatch(r"=+|-+", line.strip() or "x") and buf and buf[-1].strip():
                # underline of a setext heading already handled on the previous line
                continue
        if heading is None:
            buf.append(line)
            continue
        flush()
        level, name = heading
        if doc_title is None and level == 1:
            doc_title = name
        path = _nest(stack, level, name)
    flush()
    return doc_title or title, sections


_BLOCK = {
    "p",
    "div",
    "section",
    "article",
    "li",
    "ul",
    "ol",
    "br",
    "tr",
    "table",
    "pre",
    "blockquote",
    "dd",
    "dt",
    "dl",
    "header",
    "footer",
    "main",
    "aside",
    "figure",
    "hr",
}
_SKIP = {"script", "style", "noscript", "template", "svg", "head", "iframe", "object"}


class _HTMLSections(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sections: list[Section] = []
        self.stack: list[tuple[int, str]] = []
        self.path: tuple[str, ...] = ()
        self.buf: list[str] = []
        self.skip = 0
        self.heading_level: int | None = None
        self.heading_buf: list[str] = []
        self.in_title = False
        self.title = ""
        self.first_h1 = ""

    def flush(self) -> None:
        body = clean(re.sub(r"[ \t]+", " ", "".join(self.buf)))
        body = "\n".join(line.strip() for line in body.split("\n"))
        body = clean(body)
        if body:
            self.sections.append(Section(self.path, body))
        self.buf.clear()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self.in_title = True
        if tag in _SKIP:
            self.skip += 1
            return
        if re.fullmatch(r"h[1-6]", tag):
            self.flush()
            self.heading_level = int(tag[1])
            self.heading_buf = []
        elif tag in _BLOCK:
            self.buf.append("\n\n" if tag != "br" else "\n")
        elif tag in ("td", "th"):
            self.buf.append(" ")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("br", "hr"):
            self.buf.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False
        if tag in _SKIP:
            self.skip = max(0, self.skip - 1)
            return
        if self.heading_level is not None and re.fullmatch(r"h[1-6]", tag):
            name = " ".join("".join(self.heading_buf).split())
            if name:
                if self.heading_level == 1 and not self.first_h1:
                    self.first_h1 = name
                self.path = _nest(self.stack, self.heading_level, name)
            self.heading_level = None
        elif tag in _BLOCK:
            self.buf.append("\n\n")

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title += data
        if self.skip:
            return
        if self.heading_level is not None:
            self.heading_buf.append(data)
        else:
            self.buf.append(data)


def parse_html(text: str, title: str) -> tuple[str, list[Section]]:
    parser = _HTMLSections()
    parser.feed(text)
    parser.close()
    parser.flush()
    found = " ".join(parser.title.split()) or parser.first_h1
    return found or title, parser.sections


def _looks_like_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 80 or s.endswith((".", ",", ";", ":")) and not re.match(r"^[IVXLC]+\.", s):
        return False
    letters = [c for c in s if c.isalpha()]
    if len(letters) < 3:
        return False
    if re.match(r"^(chapter|part|book|section)\b", s, re.I):
        return True
    return all(c.isupper() for c in letters)


def parse_text(text: str, title: str) -> tuple[str, list[Section]]:
    """Plain text: short ALL-CAPS or ``CHAPTER``-style lines between blank lines are headings."""
    text = clean(text)
    lines = text.split("\n")
    sections: list[Section] = []
    path: tuple[str, ...] = ()
    buf: list[str] = []
    doc_title: str | None = None
    for i, line in enumerate(lines):
        prev_blank = i == 0 or not lines[i - 1].strip()
        next_blank = i + 1 >= len(lines) or not lines[i + 1].strip()
        if prev_blank and next_blank and _looks_like_heading(line):
            name = " ".join(line.split())
            body = clean("\n".join(buf))
            if not body and not sections and doc_title is None:
                doc_title = name.title()
            if body:
                sections.append(Section(path, body))
            buf = []
            path = (name,)
            continue
        buf.append(line)
    body = clean("\n".join(buf))
    if body:
        sections.append(Section(path, body))
    return doc_title or title, sections


def parse_pdf(data: bytes, title: str) -> tuple[str, list[Section]]:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise IngestError("encrypted PDF")
        meta_title = ""
        if reader.metadata is not None and reader.metadata.title:
            meta_title = str(reader.metadata.title).strip()
        sections = []
        for number, page in enumerate(reader.pages[:MAX_PDF_PAGES], start=1):
            text = clean(page.extract_text() or "")
            if text:
                sections.append(Section((), text, page=number))
    except (PdfReadError, ValueError, KeyError, TypeError) as exc:
        raise IngestError(f"unreadable PDF: {exc}") from exc
    if meta_title and meta_title.lower() != "untitled":
        title = meta_title
    return title, sections
