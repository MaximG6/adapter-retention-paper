"""Every count word in the body is a claim. Check it against what it counts.

An external review's meta-note, and it is worth more than any single fix it found:
"Every count word in the body should be generated rather than typed. A regex listing
every number word in the body next to the row count of the table it introduces would
have caught S3, M2, M3, M4 and M13 in one pass."

The class is real and this project has produced it repeatedly: "four were not confirmed"
beside a table showing six; "fifteen practices" beside an appendix with seven; "six
published adapters" on a figure plotting nine; "All three are worth having" after a list
of two. None of it is catchable by the claim audit, which compares printed values against
raw records -- a count word is not a printed value, and the thing it counts is a
structure, not a measurement.

What this does: extract every cardinal in the body that quantifies a countable structure,
resolve that structure, and compare. Counts that name a measured quantity rather than a
structure (`six adapters`, `nine adapters`) are resolved against the record sets.

Usage:
    python analysis/countcheck.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEXDIR = REPO_ROOT / "paper" / "tex"
PAPER = REPO_ROOT / "paper"
P0 = REPO_ROOT / "results" / "raw" / "phase0"
P1 = REPO_ROOT / "results" / "raw" / "phase1"

WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "twenty": 20, "thirty": 30,
    "thirty-two": 32,
}
_NUM = (r"(?<![\w-])(?:" + "|".join(sorted(WORDS, key=len, reverse=True))
        + r"|\d+)")


def _n(tok: str) -> int:
    return WORDS.get(tok.lower(), int(tok) if tok.isdigit() else -1)


# --------------------------------------------------------------------- what exists
def outcome_buckets(lessons: str) -> dict[str, int]:
    """How many prediction ENTRIES sit in each not-confirmed bucket of the §7.0 taxonomy.

    A count word can be correct as arithmetic and wrong as membership, and the body
    committed exactly that: it said "two untested because the adapters they need are not
    public" against a bucket listing P3, P5 and the remainder of P8, and the totals still
    reconciled to six because P8 is split across two buckets. A checker verifying
    2+1+1+2 = 6 passes it. This resolves the bucket instead.
    """
    out: dict[str, int] = {}
    for line in lessons.splitlines():
        m = re.match(r"^-\s+\*\*(.+?)\*\*\s+—\s*(.*)$", line.strip())
        if not m:
            continue
        # The bullets wrap; take everything up to the terminating ; or . on the block.
        out[m.group(1).lower()] = len(set(re.findall(r"\bP(\d+)\b", m.group(2))))
    # A wrapped bullet loses its continuation line, so re-scan the block as one string.
    block = re.sub(r"\n\s+", " ", lessons)
    for line in block.splitlines():
        m = re.match(r"^-\s+\*\*(.+?)\*\*\s+—\s*(.*)$", line.strip())
        if m:
            out[m.group(1).lower()] = len(set(re.findall(r"\bP(\d+)\b", m.group(2))))
    return out


def structures() -> dict[str, int]:
    """The countable things the body makes claims about, resolved from the artifacts."""
    # The practice entries moved to METHODOLOGY.md in the round-8 cut and the
    # registered-prediction table stayed in the paper. Both are still resolved.
    lessons = (REPO_ROOT / "METHODOLOGY.md").read_text(encoding="utf-8")
    preds_src = (PAPER / "07-registered-predictions.md").read_text(encoding="utf-8")
    apx = (TEXDIR / "appendices.tex").read_text(encoding="utf-8")
    main = (TEXDIR / "main.tex").read_text(encoding="utf-8")

    p0 = {json.loads(x)["adapter"]
          for f in P0.glob("public_adapter/*/L4_*/records.jsonl")
          for x in f.read_text(encoding="utf-8").splitlines() if x.strip()}
    p1 = {json.loads(x)["adapter"]
          for f in P1.glob("*/records.jsonl")
          for x in f.read_text(encoding="utf-8").splitlines() if x.strip()}

    # Practice entries: "## 7.n <title>", excluding the registered-predictions table.
    practices = re.findall(r"(?m)^##\s+M\.(\d+)\s", lessons)
    preds = re.findall(r"(?m)^\|\s+\*\*P(\d+)\*\*", preds_src)
    # Two populations, because the appendix claims completeness over both and the paper
    # quotes each: P1-P9 came from a dated planning document, P10 and P11 were registered
    # later in the notebook entry for their own run. Counting only the total let the body
    # say "nine" while the table held eleven, which is the completeness defect one level
    # up from the one this appendix exists to prevent.
    split = preds_src.split("**Registered later")
    planning = re.findall(r"(?m)^\|\s+\*\*P(\d+)\*\*", split[0])
    buckets = outcome_buckets(preds_src)

    return {
        "weight-space adapters": len(p0),
        "behavioural adapters": len(p1),
        "practice entries": len(practices),
        "registered predictions": len(preds),
        "planning-document predictions": len(planning),
        "figures": len(re.findall(r"\\begin\{figure\*?\}", main + apx)),
        "body tables": len(re.findall(r"\\begin\{table\*?\}", main)),
        "decades of the synthetic sweep": sweep_decades(),
        "untested-for-want-of-adapters entries": next(
            (v for k, v in buckets.items() if k.startswith("untested")), -1),
        # EXP-NNN specifically, not every `## [` heading: the notebook also carries the
        # entry-format template, which looks like an entry and is not one.
        "notebook entries": len(re.findall(
            r"(?m)^##\s+\[\d{4}-\d{2}-\d{2}\]\s+EXP-\d{3}:",
            (REPO_ROOT / "EXPERIMENTS.md").read_text(encoding="utf-8"))),
    }


def sweep_decades() -> int:
    """Order-of-magnitude span of the synthetic sweep, to the nearest decade.

    Figure 6 printed three different values for this one quantity in three places --
    caption "four", in-figure legend "2", body and A.3 "three" -- and the count-word gate
    could not see two of them because they are string literals in a plotting script. The
    sweep runs mean|D|/s from 0.00109 to 1.08659, a factor of 997, which is 2.9987
    decades: `floor` gives 2 and `round` gives 3. Rounding is right here -- the sweep was
    designed as three decades and lands 0.3% short of the round number.
    """
    import math

    xs = [json.loads(x)["mean_abs_delta_over_s"]
          for f in (P0 / "synthetic").glob("records.jsonl")
          for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    xs = [x for x in xs if x > 0]
    return round(math.log10(max(xs) / min(xs))) if xs else -1


def figure_strings() -> list[tuple[str, int, str]]:
    """(file, line, text) for every string literal in the figure scripts.

    Two of the last three review rounds' figure defects were string literals: a panel
    subtitle reading "max error 0.0%" for a vacuous computation, and a legend reading
    "2 decades" beside a caption reading "four". Neither the claim audit nor the count-word
    gate could see them, because both read the paper and neither reads the scripts that
    draw its figures. Every string literal in a figure script is a claim; this makes them
    visible to the same rules.
    """
    import ast

    out: list[tuple[str, int, str]] = []
    paths = sorted((REPO_ROOT / "analysis").glob("fig*.py"))
    # md_to_tex.py holds FIG_CAPTIONS, which is where two of the three conflicting
    # "decades" values lived. A caption is figure content wherever it is stored.
    paths.append(REPO_ROOT / "analysis" / "md_to_tex.py")
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # Docstrings are code documentation, not figure content, and they are where an
        # author records the superseded value on purpose: `_decades`' docstring says the
        # legend "said 4 decades" and is correct to say so. A gate that fires on a
        # deliberate historical quote teaches its author to ignore it.
        skip = {id(ast.get_docstring(n, clean=False) and n.body[0].value)
                for n in ast.walk(tree)
                if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef)) and ast.get_docstring(n) is not None}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and id(node) not in skip):
                out.append((path.name, node.lineno, node.value))
    return out


#: (regex over the body, structure key). The regex must capture the cardinal in group 1.
#: Only patterns whose referent is unambiguous belong here: a check that guesses what a
#: number counts will disagree with correct prose and teach its author to ignore it.
RULES: list[tuple[str, str, str]] = [
    (rf"({_NUM})\s+published\s+adapters", "weight-space adapters",
     "adapters with a Phase 0 weight-space run"),
    (rf"({_NUM})\s+public(?:ly)?\s+(?:LoRA\s+)?adapters", "weight-space adapters",
     "adapters with a Phase 0 weight-space run"),
    (rf"across\s+({_NUM})\s+published", "weight-space adapters",
     "adapters with a Phase 0 weight-space run"),
    (rf"({_NUM})\s+Taboo\s+adapters", "behavioural adapters",
     "adapters with a Phase 1 behavioural run"),
    (rf"({_NUM})\s+behavioural\s+adapters", "behavioural adapters",
     "adapters with a Phase 1 behavioural run"),
    (rf"({_NUM})\s+practices", "practice entries",
     "numbered practice entries in the methodology appendix"),
    (rf"pre-registered\s+\\?\*?\*?({_NUM})\\?\*?\*?\s+predictions",
     "registered predictions", "rows in the registered-predictions table"),
    (rf"({_NUM})\s+registered\s+predictions", "registered predictions",
     "rows in the registered-predictions table"),
    (rf"({_NUM})\s+in\s+a\s+dated\s+planning\s+document",
     "planning-document predictions",
     "P1-P9 rows, before the later-registration block"),
    # The lookbehind keeps this off the NINE ADAPTERS' own span, 1.1 decades, which is a
    # different population and a different claim (M18: the contribution bullet quoted the
    # sweep's three for the adapters'). Without it "1.1 decades" captures the second 1.
    (rf"(?<![\d.])({_NUM})\s+decades", "decades of the synthetic sweep",
     "decades spanned by the synthetic sweep's mean|D|/s"),
    (rf"({_NUM})\s+untested\s+because", "untested-for-want-of-adapters entries",
     "entries in the taxonomy's untested bucket"),
    (rf"({_NUM})\s+entries\s+in\s+`?EXPERIMENTS\.md", "notebook entries",
     "EXP-NNN entries in the lab notebook"),
    (rf"\(({_NUM})\s+entries\)", "notebook entries",
     "EXP-NNN entries in the lab notebook"),
]

#: Deliberately NOT a rule: "<n> precisions". The paper says "four precisions" for the
#: full grid including BF16 and "three quantized grids" for the comparison set, and both
#: are correct. A rule whose referent depends on the sentence around it will disagree
#: with correct prose, which is the failure this file exists to avoid committing.


def check(text: str, have: dict[str, int]) -> list[tuple[str, int, int, str, str]]:
    """(matched phrase, claimed, actual, what it counts, context) for disagreements."""
    bad = []
    for pattern, key, what in RULES:
        for m in re.finditer(pattern, text, re.I):
            claimed = _n(m.group(1))
            if claimed < 0 or claimed == have[key]:
                continue
            lo = max(0, m.start() - 60)
            ctx = " ".join(text[lo:m.end() + 50].split())
            bad.append((m.group(0), claimed, have[key], what, ctx))
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    have = structures()
    # The body AND the appendix sources: three of the four instances this was built for
    # were in appendices or figure captions, not in main.tex.
    bad = []
    # The companion documents are inside this gate for the same reason the paper is:
    # round 8 moved content out of the PDF, and a count word does not stop being a claim
    # because it now lives in the repo.
    companions = [REPO_ROOT / n for n in ("METHODOLOGY.md", "PROMPTS.md", "README.md")]
    for path in ([TEXDIR / "main.tex"] + sorted(PAPER.glob("*.md"))
                 + [p for p in companions if p.exists()]):
        for hit in check(path.read_text(encoding="utf-8"), have):
            bad.append((path.name,) + hit)
    # And the figure scripts: a legend or a subtitle is as much a claim as a sentence.
    strings = figure_strings()
    for name, lineno, text in strings:
        for hit in check(text, have):
            bad.append((f"{name}:{lineno}",) + hit)

    print("structures resolved from the artifacts:")
    for k, v in sorted(have.items()):
        print(f"  {k:<38} {v}")
    if not bad:
        print(f"\nevery count word agrees ({len(RULES)} rules over the body and "
              f"{len(strings)} string literals in the figure scripts)")
        return 0
    print(f"\n{len(bad)} count words disagree with what they count:", file=sys.stderr)
    for name, phrase, claimed, actual, what, ctx in bad:
        print(f"  [{name}] {phrase!r}: says {claimed}, there are {actual} {what}",
              file=sys.stderr)
        print(f"    ...{ctx}...", file=sys.stderr)
    return 1 if args.strict else 1


if __name__ == "__main__":
    raise SystemExit(main())
