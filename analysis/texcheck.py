r"""Read the built PDF's text layer and fail on LaTeX or encoding debris.

Every other gate in this project inspects sources. None of them looked at what the
reader actually sees, and that is where this class of defect lives: a control sequence
destroyed before it reached the file compiles to plain text, TeX emits no warning, and
the sentence renders as "Six Taboo adapters imes four precisions".

**Root cause of the four occurrences, corrected.** Not the shell. Scripted edits held
LaTeX in ordinary Python string literals, where `\t` is TAB, `\r` is CR and `\f` is FF.
The heredoc passed the text through faithfully; Python's lexer ate the backslash before
the text was ever written. A CR is additionally invisible to `Path.read_text()`, which
normalises newlines, so a source-level scan for control characters misses half of it.
Checking the rendered text is the only place all of it is visible at once.

**Why some obvious patterns are absent.** A gate that fires on correct input teaches its
author to ignore it, which is the failure this project has recorded seven times. Bare
`sqrt` and bare braces occur legitimately in this paper's code listings
(`alpha/sqrt(r)`, `--scheme {asymmetric, symmetric_gptq}`), so they are not flagged;
`\sqrt` and `macroname{` are flagged instead, and those cannot occur in running text.

Usage:
    python analysis/texcheck.py [--pdf paper/adapter-retention-arxiv.pdf]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = REPO_ROOT / "paper" / "adapter-retention-arxiv.pdf"

#: Macro names whose backslash going missing leaves the name and its brace in the text.
_MACROS = ("ref", "label", "textbf", "textit", "emph", "texttt", "textsc", "cite",
           "section", "subsection", "footnote", "caption", "item")

CHECKS: list[tuple[str, str, str]] = [
    ("backslash-macro", r"\\(?:" + "|".join(_MACROS) + r")\b",
     "a LaTeX macro reached the page as literal text"),
    ("bare-macro-brace", r"(?<![A-Za-z\\])(?:" + "|".join(_MACROS) + r")\{",
     "a macro lost its backslash; the name and brace rendered"),
    ("times", r"(?<![A-Za-z\\])imes(?![A-Za-z])",
     r"\times lost its backslash to a TAB escape"),
    ("ref-sec", r"(?:efsec|ef\{sec)",
     r"\ref lost its backslash to a CR escape"),
    ("sqrt-macro", r"(?:\\sqrt|sqrt\{)",
     r"\sqrt reached the page as literal text"),
    ("replacement-char", "�",
     "a character was lost in an encoding round-trip"),
    ("double-question", r"\?\?",
     "two characters were replaced by ? during the ASCII pass"),
    ("markdown-bold", r"\*\*",
     "markdown emphasis survived conversion to LaTeX"),
]

#: Deliberately NOT checked, with the reason, because each would fire on correct input:
#:
#:   bare `sqrt`, bare `{`/`}`  -- the appendices print shell and Python that contain
#:                                 both (`--bits {3,4,8}`, `alpha/sqrt(r)`).
#:   `prefuse` and friends      -- a correctly typeset subscript, `$p_{\text{refuse}}$`,
#:                                 extracts from the PDF text layer WITHOUT the
#:                                 underscore, because a subscript is smaller type and
#:                                 not a character. An external review read the extracted
#:                                 text and reported this as a defect; the rendered page
#:                                 is correct. A check for it would fire on every
#:                                 subscript in the paper.
#:   a lone `?`                 -- the paper asks real questions, and PyMuPDF renders the
#:                                 ff/fl ligatures and `§` as `?` regardless. The
#:                                 encoding class is closed at the source instead:
#:                                 `ascii_only` now raises rather than substituting.


def pdf_text(pdf: Path) -> str:
    import fitz

    with fitz.open(pdf) as doc:
        return "\n".join(page.get_text() for page in doc)


def scan(text: str) -> list[tuple[str, str, str]]:
    """(check name, matched text, surrounding context) for every hit."""
    out: list[tuple[str, str, str]] = []
    for name, pattern, _ in CHECKS:
        for m in re.finditer(pattern, text):
            lo = max(0, m.start() - 55)
            ctx = " ".join(text[lo:m.end() + 40].split())
            out.append((name, m.group(0), ctx))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    args = ap.parse_args()

    if not args.pdf.exists():
        print(f"no such PDF: {args.pdf}", file=sys.stderr)
        return 1

    hits = scan(pdf_text(args.pdf))
    if not hits:
        print(f"     no LaTeX or encoding debris in {args.pdf.name} "
              f"({len(CHECKS)} checks)")
        return 0

    why = {name: reason for name, _, reason in CHECKS}
    print(f"     {len(hits)} pieces of debris in the rendered text:", file=sys.stderr)
    for name, hit, ctx in hits[:20]:
        print(f"       [{name}] {hit!r} -- {why[name]}", file=sys.stderr)
        print(f"         ...{ctx}...", file=sys.stderr)
    if len(hits) > 20:
        print(f"       ... and {len(hits) - 20} more", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
