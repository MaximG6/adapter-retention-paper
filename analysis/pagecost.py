"""Page cost per section of the built arXiv PDF, measured rather than estimated.

Two-column, so cost is column-height, not page count. Headings are matched against the
label set `xref` derives by numbering the LaTeX exactly as LaTeX does, so a bolded run-in
paragraph cannot be mistaken for one.

**Why this is in the repository and under test.** For three review rounds it was a
throwaway script in a scratch directory, and it was wrong: a section's span ran from its
own heading to the *next heading*, and nothing follows the last one. The bibliography --
1.83 pages -- was therefore charged to the Conclusion, and every page-budget argument in
rounds 8, 9 and 10 was made against a body 1.8 pages larger than it is. An external
reviewer's cut plan was sized against it too.

Nothing caught it because the number looked plausible. A 2.35-page conclusion is odd but
not absurd, and no check compared the sum of the parts against the document. That is
`METHODOLOGY.md` M.3 -- a gate must be tested against known-bad input -- applied to the
wrong class of object: **an instrument that sizes a decision is a gate, and needs the same
treatment.** `span_of` is separated out as a pure function precisely so it can be fed a
layout whose right answer is known.

Usage:
    PYTHONPATH=src python analysis/pagecost.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "analysis"))

PDF = REPO_ROOT / "paper" / "adapter-retention-arxiv.pdf"

#: Blocks that end a section without being one. Unnumbered, so `xref` never sees them,
#: and every column after them belonged to the preceding heading until this existed.
TERMINATORS = ("References", "Acknowledgements", "Acknowledgments", "Bibliography")


def span_of(marks: dict[str, float], terminators: dict[str, float],
            end: float) -> dict[str, float]:
    """Column-extent of each label, given where every label and terminator starts.

    A span runs to the next *mark of any kind*, not to the next label. Those differ
    exactly where an unnumbered block follows a numbered one -- which in this document is
    the bibliography, sitting between the Conclusion and Appendix A, and charged to the
    Conclusion for three rounds.
    """
    all_marks = sorted(list(marks.items()) + list(terminators.items()),
                       key=lambda kv: kv[1])
    out: dict[str, float] = {}
    for i, (name, pos) in enumerate(all_marks):
        nxt = all_marks[i + 1][1] if i + 1 < len(all_marks) else end
        out[name] = nxt - pos
    return out


def measure() -> tuple[dict[str, float], dict[str, float], dict[str, str], int]:
    import fitz

    import xref

    secs, apps, titles = xref.structure(
        (REPO_ROOT / "paper" / "tex" / "main.tex").read_text(encoding="utf-8"),
        (REPO_ROOT / "paper" / "tex" / "appendices.tex").read_text(encoding="utf-8"))
    labels = secs | apps

    doc = fitz.open(PDF)
    page_h = doc[0].rect.height
    top, bot = 50.0, page_h - 50.0
    col = bot - top

    def pos(pno: int, c: int, y: float) -> float:
        return (pno * 2 + c) + (min(max(y, top), bot) - top) / col

    def plain(s: str) -> str:
        s = re.sub(r"\\[a-zA-Z]+\s*", " ", s)
        s = re.sub(r"[{}$\\~]", "", s)
        s = s.replace("--", "-")
        return re.sub(r"[^a-z0-9 ]+", " ", s.lower())

    words = {k: plain(v).split() for k, v in titles.items()}
    found: dict[str, float] = {}
    terms: dict[str, float] = {}
    for pno, page in enumerate(doc):
        w = page.rect.width
        for b in page.get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                spans = line["spans"]
                txt = "".join(s["text"] for s in spans).strip()
                if not any("Bold" in s["font"] for s in spans):
                    continue
                x0 = min(s["origin"][0] for s in spans)
                where = pos(pno, 0 if x0 < w / 2 else 1, spans[0]["origin"][1])
                if txt in TERMINATORS and txt not in terms:
                    terms[txt] = where
                    continue
                m = re.match(r"^((?:\d+(?:\.\d+)*)|(?:[A-G](?:\.\d+)*))\s+(\S.*)$", txt)
                if not m or m.group(1) not in labels or m.group(1) in found:
                    continue
                # A bold run-in paragraph can begin "A note on..." and match label "A".
                want, got = words.get(m.group(1), []), plain(m.group(2)).split()
                if not want or got[:2] != want[:2]:
                    continue
                found[m.group(1)] = where
    return found, terms, titles, doc.page_count


def main() -> int:
    found, terms, titles, pages = measure()
    end = pages * 2.0
    span = span_of(found, terms, end)
    order = sorted(found, key=lambda k: found[k])

    print(f"{PDF.name}: {pages} pages, two columns, {len(order)} labels located, "
          f"{len(terms)} unnumbered blocks\n")
    print(f"{'section':<58} {'cols':>6} {'pages':>7}")
    print("-" * 74)
    for lab in order:
        depth = lab.count(".")
        t = titles.get(lab, "")[:52 - 2 * depth]
        print(f"{'  ' * depth}{lab} {t:<{54 - 3 * depth}} {span[lab]:>6.2f} "
              f"{span[lab] / 2:>7.2f}")

    body = apx = 0.0
    print("\n" + "=" * 74)
    print(f"{'TOP LEVEL':<58} {'cols':>6} {'pages':>7}")
    for lab in order:
        if "." in lab:
            continue
        tot = sum(span[k] for k in order if k == lab or k.startswith(lab + "."))
        print(f"  {lab} {titles.get(lab, '')[:48]:<50} {tot:>6.2f} {tot / 2:>7.2f}")
        if lab.isdigit():
            body += tot
        else:
            apx += tot
    print("-" * 74)
    print(f"  {'BODY PROSE (1-10)':<52} {body:>6.2f} {body / 2:>7.2f}")
    for name, pos_ in sorted(terms.items(), key=lambda kv: kv[1]):
        print(f"  {name:<52} {span[name]:>6.2f} {span[name] / 2:>7.2f}")
    print(f"  {'APPENDICES (A-G)':<52} {apx:>6.2f} {apx / 2:>7.2f}")
    total = body + apx + sum(span[n] for n in terms)
    print(f"  {'TOTAL':<52} {total:>6.2f} {total / 2:>7.2f}   "
          f"({pages} rendered)")

    # The check that was missing: the parts have to add up to the document. A span that
    # runs past its section's end inflates one row and nothing else disagrees with it.
    slack = end - total
    print(f"\n  unattributed (title block, front matter): {slack:.2f} cols "
          f"= {slack / 2:.2f} pages")
    if slack < -0.01:
        print("  PARTS EXCEED THE DOCUMENT -- a span is running past its section",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
