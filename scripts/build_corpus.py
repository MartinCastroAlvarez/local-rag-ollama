"""Rebuild the bundled benchmark corpus from Project Gutenberg downloads.

Usage (maintainers only; the result is committed under ``local_rag/data/corpus``)::

    python scripts/build_corpus.py <download-dir>

``<download-dir>`` must hold the files fetched from gutenberg.org::

    pg5.txt            The United States Constitution          (ebook #5)
    pg17405.txt        Sun Tzu, The Art of War, tr. L. Giles   (ebook #17405)
    pg37134-images.html  W. Strunk Jr., The Elements of Style  (ebook #37134)
    pg71.txt           H. D. Thoreau, On the Duty of Civil Disobedience (ebook #71)

All four works are in the public domain in the United States. The Project Gutenberg
header, footer and licence (which only govern use of the Gutenberg trademark) are
stripped, as the Project Gutenberg licence allows. Each work is stored in a different
format so the benchmark exercises every loader: Markdown, plain text, HTML and PDF.
The PDF is printed from a generated HTML page with headless Chrome (``CHROME`` env var).
"""

from __future__ import annotations

import html
import os
import re
import subprocess  # nosec B404 - maintainer script, fixed argv, never shell
import sys
import tempfile
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "local_rag" / "data" / "corpus"


def _body(text: str) -> str:
    text = text.replace("\r", "")
    start = text.index("*** START OF")
    start = text.index("\n", start) + 1
    end = text.index("*** END OF")
    return text[start:end].strip() + "\n"


def constitution(src: Path) -> str:
    body = _body(src.read_text(encoding="utf-8"))
    body = body[body.index("THE CONSTITUTION OF THE UNITED STATES OF AMERICA") :]
    words = {"THREE": "3", "FOUR": "4", "FIVE": "5", "SIX": "6", "SEVEN": "7"}
    out: list[str] = []
    for line in body.splitlines():
        if line.startswith("THE CONSTITUTION OF THE UNITED STATES"):
            out.append("# The Constitution of the United States of America (1787)")
            continue
        m = re.match(r"^(?:Article|ARTICLE) (\w+)\s*$", line)
        if m:
            out.append(f"## Article {words.get(m.group(1), m.group(1))}")
            continue
        m = re.match(r"^Section (\d+)\.\s+(.*)$", line)
        if m:
            out.extend([f"### Section {m.group(1)}", "", m.group(2)])
            continue
        out.append(line)
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def art_of_war(src: Path) -> str:
    body = _body(src.read_text(encoding="utf-8"))
    return re.sub(r"\n{4,}", "\n\n\n", body)


def elements_of_style(src: Path) -> str:
    raw = src.read_text(encoding="utf-8")
    start = raw.index('<div id="pg-start-separator">')
    start = raw.index("</header>", start) + len("</header>")
    end = raw.index('<div id="pg-end-separator">')
    end = raw.index('<div id="tnote-bottom">', start)
    body = raw[start:end]
    body = re.sub(r"<img[^>]*>", "", body)
    body = re.sub(r"<a class=\"pagenum\"[^>]*>\s*</a>", "", body)
    body = re.sub(r'<div id="tnote">.*?</div>', "", body, flags=re.S)
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        "<title>The Elements of Style, by William Strunk, Jr. (1918)</title>\n</head>\n"
        f"<body>\n{body.strip()}\n</body>\n</html>\n"
    )


def civil_disobedience_pdf(src: Path, dest: Path) -> None:
    body = _body(src.read_text(encoding="utf-8"))
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    title, *rest = paras
    rest = [p for p in rest if not p.startswith("by Henry")]
    page = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(' '.join(title.split()))} (Thoreau, 1849)</title>",
        "<style>body{font:11pt Georgia,serif;margin:2cm}h1{font-size:18pt}</style>",
        f"</head><body><h1>{html.escape(title)}</h1>",
    ]
    page += [f"<p>{html.escape(' '.join(p.split()))}</p>" for p in rest]
    page.append("</body></html>")
    chrome = os.environ.get("CHROME", str(Path.home() / ".cache/cairo/chrome"))
    with tempfile.TemporaryDirectory() as tmp:
        page_path = Path(tmp) / "civil.html"
        page_path.write_text("\n".join(page), encoding="utf-8")
        subprocess.run(  # nosec B603 - fixed argv
            [
                chrome,
                "--headless",
                "--disable-gpu",
                "--no-sandbox",
                "--no-pdf-header-footer",
                f"--print-to-pdf={dest}",
                page_path.as_uri(),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )


def main() -> None:
    src = Path(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "us-constitution.md").write_text(constitution(src / "pg5.txt"), encoding="utf-8")
    (OUT / "art-of-war.txt").write_text(art_of_war(src / "pg17405.txt"), encoding="utf-8")
    (OUT / "elements-of-style.html").write_text(
        elements_of_style(src / "pg37134-images.html"), encoding="utf-8"
    )
    civil_disobedience_pdf(src / "pg71.txt", OUT / "civil-disobedience.pdf")
    for p in sorted(OUT.iterdir()):
        print(f"{p.name:28} {p.stat().st_size:>8} bytes")


if __name__ == "__main__":
    main()
