# Bundled benchmark data

`corpus/` holds four public-domain works, each in a different format so the benchmark
exercises every loader. They come from [Project Gutenberg](https://www.gutenberg.org/);
the Project Gutenberg header, footer and licence (which govern only the Gutenberg
trademark) were removed, as that licence allows. All four are in the public domain in
the United States. `scripts/build_corpus.py` rebuilds the folder from the downloads.

| File | Work | Gutenberg ebook |
| --- | --- | --- |
| `us-constitution.md` | The Constitution of the United States (1787), converted to Markdown headings | [#5](https://www.gutenberg.org/ebooks/5) |
| `art-of-war.txt` | Sun Tzu, *The Art of War*, tr. Lionel Giles (1910), plain text | [#17405](https://www.gutenberg.org/ebooks/17405) |
| `elements-of-style.html` | William Strunk Jr., *The Elements of Style* (1918), HTML | [#37134](https://www.gutenberg.org/ebooks/37134) |
| `civil-disobedience.pdf` | Henry David Thoreau, *On the Duty of Civil Disobedience* (1849), printed to PDF | [#71](https://www.gutenberg.org/ebooks/71) |

`qa.jsonl` is written for this project (MIT): 41 answerable questions, each with an
`evidence` span copied verbatim from its source (a retrieved chunk counts as relevant
when it contains that span), and 4 questions the corpus cannot answer, which a good
system should refuse.
