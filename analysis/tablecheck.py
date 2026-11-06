"""Compare numeric cells that appear in more than one table.

The cross-artifact check in `audit_draft_numbers.py` compares quantities claimed in
*prose*. It passed 18/18 across 56 sites while section 5.1's Table 2 and Appendix B.6
printed the same three confidence intervals with different values -- because both were
tables, and no check looked at tables against each other. Two tables disagreeing was the
original defect; the fix for it did not cover its own case.

This reads every table in the built document, LaTeX and markdown alike, normalises each
cell to (what the row is about, what the column is about, value), and reports any
quantity carried in two places with two values.

Transposition is the reason this cannot be a naive (row, column) join: Table 2 has
precisions down the rows, B.6 has them across the columns. Cells are therefore keyed by
the pair of labels that bound them, in either order.

Usage:
    python analysis/tablecheck.py            # report
    python analysis/tablecheck.py --strict   # exit 1 on any disagreement
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    REPO_ROOT / "paper" / "tex" / "main.tex",
    REPO_ROOT / "paper" / "tex" / "appendices.tex",
]

#: Row/column labels that name the same thing in different artifacts.
ALIAS = {
    "int4 g128": "int4_g128", "int4_g128": "int4_g128",
    "int4 per-channel": "int4_per_channel", "int4_per_channel": "int4_per_channel",
    "int3 g128": "int3_g128", "int3_g128": "int3_g128",
    "bf16": "bf16", "bf16 (raw)": "bf16",
    "95% ci over adapters": "ci", "95% ci": "ci",
    "mean retention": "mean", "mean": "mean",
    "adapters below 50%": "below50", "below 50%": "below50",
}

_INTERVAL = re.compile(r"\[\s*([\d.]+)\\?%?\s*,\s*([\d.]+)\\?%?\s*\]")
_NUMBER = re.compile(r"^\**\s*([\d.]+)\\?%?\s*\**$")


def _clean(cell: str) -> str:
    s = re.sub(r"\\(textbf|texttt|emph|mathbf)\{([^}]*)\}", r"\2", cell)
    s = s.replace("\\allowbreak", "")
    # Escaped punctuation first: \_ is a backslash before a NON-letter, so the
    # \\[a-zA-Z]+ sweep below leaves it, and `int4\_g128` then fails to join with
    # `int4_g128` from the markdown side. That is why the first version of this
    # checker reported no disagreement while two tables visibly disagreed.
    for esc, lit in (("\\_", "_"), ("\\%", "%"), ("\\&", "&"), ("\\#", "#"),
                     ("\\$", "$")):
        s = s.replace(esc, lit)
    s = re.sub(r"\\[a-zA-Z]+\s*", "", s)
    s = s.replace("{", "").replace("}", "").replace("**", "").replace("`", "")
    s = s.replace("$", "").strip()
    return s


def _key(label: str) -> str:
    lab = _clean(label).lower().rstrip(":")
    return ALIAS.get(lab, lab)


def rows_from_latex(text: str) -> list[list[list[str]]]:
    out = []
    for m in re.finditer(r"\\begin\{tabularx?\}.*?\\end\{tabularx?\}", text, re.S):
        body = m.group(0)
        rows = []
        for line in body.split("\\\\"):
            line = re.sub(r"\\(toprule|midrule|bottomrule|hline)", "", line)
            if "&" in line:
                rows.append([_clean(c) for c in line.split("&")])
        if rows:
            out.append(rows)
    return out


def rows_from_markdown(text: str) -> list[list[list[str]]]:
    out, cur = [], []
    for line in text.splitlines():
        if line.strip().startswith("|"):
            cells = [_clean(c) for c in line.strip().strip("|").split("|")]
            if not all(set(c) <= set("-: ") for c in cells):
                cur.append(cells)
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def cells(tables: list[list[list[str]]], origin: str) -> dict[tuple, list[tuple]]:
    """(row-key, col-key) -> [(value, origin)] for every numeric cell."""
    found: dict[tuple, list[tuple]] = defaultdict(list)
    for t in tables:
        if len(t) < 2:
            continue
        header = [_key(h) for h in t[0]]
        for row in t[1:]:
            if not row:
                continue
            rkey = _key(row[0])
            for i, cell in enumerate(row[1:], start=1):
                ckey = header[i] if i < len(header) else f"col{i}"
                iv = _INTERVAL.search(cell)
                if iv:
                    val = f"[{float(iv.group(1)):.4g}, {float(iv.group(2)):.4g}]"
                else:
                    num = _NUMBER.match(cell)
                    if not num:
                        continue
                    val = f"{float(num.group(1)):.6g}"
                # Key both orientations so a transposed table still joins.
                found[(rkey, ckey)].append((val, origin))
                found[(ckey, rkey)].append((val, origin))
    return found


def disagreements() -> list[tuple]:
    allc: dict[tuple, list[tuple]] = defaultdict(list)
    for p in SOURCES:
        text = p.read_text(encoding="utf-8")
        tabs = (rows_from_latex(text) if p.suffix == ".tex"
                else rows_from_markdown(text))
        for k, v in cells(tabs, p.name).items():
            allc[k].extend(v)
    bad = []
    for k, entries in allc.items():
        origins = {o for _, o in entries}
        values = {v for v, _ in entries}
        if len(origins) > 1 and len(values) > 1:
            bad.append((k, sorted(set(entries))))
    return sorted(bad)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    bad = disagreements()
    n = sum(len(rows_from_latex(p.read_text(encoding="utf-8"))) for p in SOURCES)
    print(f"{n} tables scanned across {len(SOURCES)} artifacts")
    if not bad:
        print("no cell disagrees between artifacts")
        return 0
    print(f"\n{len(bad)} quantities carried in two artifacts with two values:\n")
    for (a, b), entries in bad:
        print(f"  {a}  x  {b}")
        for val, origin in entries:
            print(f"      {val:>18}  {origin}")
    return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
