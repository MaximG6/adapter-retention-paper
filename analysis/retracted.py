"""Strings this paper has retracted, and a gate that keeps them retracted.

Four of round 9's findings were one shape: a correction landed in the appendix where it
was discovered and not at the two or three body sites asserting the corrected claim. The
external reader's name for it is **partial propagation**, and it is the defect that has
survived the most consecutive review rounds in this project -- three for the ICC
correction, two for the abstract's behavioural headline.

Nothing already in the perimeter can see it. The claim audit checks numbers against raw
records, so a sentence with no number in it is invisible to it. `countcheck` checks count
words against structures. `xref` checks that references resolve. All three pass on a
document that asserts a claim its own appendix retracts, because none of them reads
meaning -- which is `METHODOLOGY.md` M.1, restated as a specific mechanism.

So: when a claim is retracted, its wording goes in the table below and this gate fails the
build if that wording is asserted anywhere in the perimeter again.

**A retraction quotes the retired wording; an assertion does not.** That is how this
project's corrections are written -- `B.12` says *This appendix said the 32 prompts carry
"roughly 16 independent units"* -- so a match inside a quoted span is sanctioned and a
bare match is not. It is a convention with teeth rather than a per-file allowlist, which
would go stale the first time a section moved.

Perimeter: the arXiv body and appendices, every markdown source of the technical report,
the three companion documents, and the full source of the analysis, scripts and `src`
trees -- because the ICC correction's four surviving assertions were all in docstrings and
comments, not in prose.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER = REPO_ROOT / "paper"
TEXDIR = PAPER / "tex"

#: (pattern, an exemplar the pattern MUST match, what replaced it, where it is recorded).
#: Patterns are regexes over the raw file text, so they must tolerate LaTeX markup where
#: the claim exists in both documents. The exemplar is the pattern's own unit test: a
#: retraction that rewrites rather than quotes leaves the old wording nowhere in the
#: corpus, so a pattern cannot be validated against the corpus and a typo would sit here
#: matching nothing while the gate reported clean.
RETRACTED: tuple[tuple[str, str, str, str], ...] = (
    (r"roughly 16 independent units",
     "counts roughly 16 independent units twice",
     "23-26 effective units at the measured ICC of 0.175-0.303", "B.12, EXP-050"),
    (r"(?:is|are) roughly 16, not 32",
     "the number of independent units is roughly 16, not 32",
     "23-26 effective units, not 32 and not 16", "B.12, EXP-050"),
    (r"no detectable change in (?:their|the) trained behaviour",
     "there is no detectable change in their trained behaviour",
     "no detectable loss of elicitation CAPABILITY; the constraint moves",
     "5.1, EXP-053"),
    (r"no behavioural change is\s+detectable",
     "and no behavioural change is\ndetectable",
     "no loss of elicitation capability is detectable", "5.1, EXP-053"),
    (r"exactly as strong at INT3\s*\n?as at BF16",
     "the constraint is exactly as strong at INT3\nas at BF16",
     "no detectable trend across precision", "B.9, EXP-053"),
    (r"nothing is missing.{0,4}\s*nine\s*\n?were registered and nine appear",
     "Second, nothing is missing}: nine were registered and nine appear",
     "eleven were registered and eleven appear", "Appendix C, EXP-053"),
    (r"no claim in this paper turns on the difference",
     "and no claim in this paper turns on the difference",
     "the per-adapter split moves; the mean and PG-1/PG-2 do not", "5.1, B.7"),
    (r"weight-space cosine (?:of|is) 0\.13",
     "a weight-space cosine of 0.13 corresponds to an output SNR",
     "weight-space SNR of 0.13; the cosine is 0.14", "B.13, EXP-053"),
    (r"1\.7\s*(?:--|-|–)\s*7\.5\s*[$×x\\]",
     r"something 1.7--7.5$\times$ its size",
     "2.0-7.5x, which is B.1's magnitude-ratio column end to end", "B.1, EXP-053"),
    (r"(?:differ|vary) fifteenfold",
     "the two readings differ fifteenfold",
     "41x for one adapter, 15.0x pooled over the nine", "B.4, EXP-053"),
    (r"same ratio predicts\s*\n?\s*layer-output fidelity",
     "the same ratio predicts\nlayer-output fidelity 6.2-16.5x higher",
     "layer-output fidelity is MEASURED 6.2-16.5x higher", "A.2, B.13"),
    (r"Stored weights UNCHANGED",
     "Stored weights UNCHANGED",
     "Figure 1 plots cosine, the statistic the paper leads with", "Figure 1, EXP-053"),
    (r"flat at 0\.0757 and 0\.0756",
     "knowledge score---is flat at 0.0757 and 0.0756",
     "no trend: 0.0757, 0.0634, 0.0730, 0.0756, overlapping intervals", "B.9"),
    (r"[Cc]lustering widens",
     "Clustering widens, because there are fewer independent units",
     "clustering is close to a wash at the measured ICC; pairing does the work", "B.12"),
    (r"clears zero by \*\*0\.1 points",
     "its lower bound clears zero by **0.1 points** on an n=6",
     "5.4 points -- the prose formatted a fraction with .1f", "B.8, EXP-053"),
    (r"survives Holm correction",
     "the one of three precision contrasts that survives Holm correction",
     "nothing survives Holm; the sign-flip permutation gives p = 0.125",
     "5.1, EXP-056"),
    (r"\$?p\$? = 0\.003",
     "enumerated two-sided $p = 0.003$ against $0.031$",
     "the bootstrap p was 2*(ties/6)^6, which counts ties rather than measuring an "
     "effect; the exact sign-flip permutation gives 0.125", "5.1, EXP-056"),
)

#: Spans in which a retired wording may legitimately appear: it is being quoted in order
#: to be retracted. Backticks included because code spans quote tool output the same way.
QUOTE_SPANS = (
    # Escaped first: a retraction inside a Python string literal writes \"...\", and the
    # unescaped pattern would pair the literal's own opening quote with the first \".
    ('\\"', '\\"'),
    (r'"', r'"'),
    ("“", "”"),
    ("``", "''"),
    (r"`", r"`"),
)


def _quoted_regions(text: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for open_, close in QUOTE_SPANS:
        for m in re.finditer(re.escape(open_) + r"(.{0,400}?)" + re.escape(close),
                             text, re.S):
            out.append((m.start(), m.end()))
    return out


def _sanctioned(pos: int, end: int, regions: list[tuple[int, int]]) -> bool:
    return any(a <= pos and end <= b for a, b in regions)


def perimeter() -> list[Path]:
    files = [TEXDIR / "main.tex", TEXDIR / "appendices.tex"]
    files += sorted(PAPER.glob("*.md"))
    files += [REPO_ROOT / n for n in ("METHODOLOGY.md", "PROMPTS.md", "README.md")]
    files += sorted((REPO_ROOT / "analysis").glob("*.py"))
    files += sorted((REPO_ROOT / "scripts").glob("*.py"))
    files += sorted((REPO_ROOT / "src" / "ar").glob("*.py"))
    return [f for f in files if f.exists() and f.name != Path(__file__).name]


def assertions_in(text: str) -> list[tuple[str, str, str, str]]:
    """Retired wordings ASSERTED (not quoted) in `text`, with 60 characters of context."""
    regions = _quoted_regions(text)
    bad: list[tuple[str, str, str, str]] = []
    for pat, _example, replacement, where in RETRACTED:
        for m in re.finditer(pat, text):
            if _sanctioned(m.start(), m.end(), regions):
                continue
            lo = max(0, m.start() - 30)
            bad.append((pat, replacement, where,
                        text[lo:m.end() + 30].replace("\n", " ")))
    return bad


def assertions_in_python(source: str) -> list[tuple[str, str, str, str]]:
    """Same, over a Python file's prose rather than its raw bytes.

    A figure's in-panel header is a string literal, so on raw text every character of it
    sits inside a quoted span and is sanctioned -- the gate would have been structurally
    unable to see "Stored weights UNCHANGED", which is the one defect on this list a
    reader would have taken away backwards. Parsing gives the literal's *value*, where
    an embedded quotation is still a quotation and the surrounding delimiters are gone.
    Comments come from `tokenize` because `ast` discards them, and four of the ICC
    correction's surviving assertions were in comments and docstrings.
    """
    import ast
    import io
    import tokenize

    prose: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return assertions_in(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            prose.append(node.value)
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT:
                prose.append(tok.string)
    except (tokenize.TokenError, IndentationError):
        pass
    bad: list[tuple[str, str, str, str]] = []
    for chunk in prose:
        bad += assertions_in(chunk)
    return bad


def check() -> list[tuple[str, str, str, str, str]]:
    """Every retired wording asserted anywhere in the perimeter."""
    bad = []
    for f in perimeter():
        text = f.read_text(encoding="utf-8")
        found = (assertions_in_python(text) if f.suffix == ".py"
                 else assertions_in(text))
        for pat, replacement, where, ctx in found:
            bad.append((str(f.relative_to(REPO_ROOT)).replace("\\", "/"),
                        pat, replacement, where, ctx))
    return bad


def main() -> int:
    bad = check()
    print(f"{len(RETRACTED)} retracted wordings, "
          f"{len(perimeter())} files in the perimeter")
    for rel, pat, replacement, where, ctx in bad:
        print(f"\nASSERTED AGAIN  {rel}\n  pattern    : {pat}\n"
              f"  retracted in: {where}\n  replaced by: {replacement}\n"
              f"  context    : ...{ctx}...")
    if bad:
        print(f"\n{len(bad)} retracted wordings are asserted again")
        return 1
    print("no retracted wording is asserted anywhere in the perimeter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
