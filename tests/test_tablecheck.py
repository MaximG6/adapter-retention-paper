"""Tests for the table-cell cross-artifact check.

This check exists because the prose-level cross-artifact check passed 18/18 across 56
sites while section 5.1's Table 2 and Appendix B.6 printed the same three intervals with
different values. Two tables disagreeing was the original defect and no check compared
tables.

Its first version reported no disagreement while the disagreement was live in the shipped
PDF: LaTeX writes `int4\\_g128` and markdown writes `int4_g128`, so the two never joined.
That case is pinned below, because a checker that silently fails to join is
indistinguishable from a document that agrees.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "analysis" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tc = _load("tablecheck")

AGREE_TEX = r"""
\begin{tabular}{lrr}
\toprule
precision & mean & 95\% CI \\
\midrule
INT4 g128 & 99.2\% & [90.7, 107.6] \\
\bottomrule
\end{tabular}
"""

DISAGREE_TEX = AGREE_TEX.replace("[90.7, 107.6]", "[90.6, 107.6]")

APX = r"""
\begin{tabularx}{\columnwidth}{lXX}
\toprule
word & int4\_g128 & int3\_g128 \\
\midrule
95\% CI over adapters & [90.7\%, 107.6\%] & [41.7\%, 74.3\%] \\
\bottomrule
\end{tabularx}
"""


def _pairs(tex_a: str, tex_b: str):
    a = tc.cells(tc.rows_from_latex(tex_a), "A")
    b = tc.cells(tc.rows_from_latex(tex_b), "B")
    merged: dict = {}
    for src in (a, b):
        for k, v in src.items():
            merged.setdefault(k, []).extend(v)
    return [(k, e) for k, e in merged.items()
            if len({o for _, o in e}) > 1 and len({v for v, _ in e}) > 1]


def test_latex_escaped_underscore_joins_with_plain() -> None:
    """The defect that made the first version of this checker useless."""
    assert tc._key(r"int4\_g128") == tc._key("int4_g128") == "int4_g128"


def test_agreeing_tables_produce_no_finding() -> None:
    assert _pairs(AGREE_TEX, APX) == []


def test_disagreeing_tables_are_caught() -> None:
    found = _pairs(DISAGREE_TEX, APX)
    assert found, "a real cross-table disagreement was not reported"
    keys = {k for k, _ in found}
    assert ("int4_g128", "ci") in keys or ("ci", "int4_g128") in keys


def test_transposed_orientation_still_joins() -> None:
    """Table 2 runs precisions down the rows; B.6 runs them across the columns."""
    found = _pairs(DISAGREE_TEX, APX)
    vals = {v for _, e in found for v, _ in e}
    assert "[90.6, 107.6]" in vals and "[90.7, 107.6]" in vals


def test_bold_and_percent_markup_do_not_block_a_match() -> None:
    bold = AGREE_TEX.replace("99.2\\%", "\\textbf{99.2\\%}")
    assert _pairs(bold, APX) == []


def test_the_built_document_has_no_cross_table_disagreement() -> None:
    """The gate. Fails while any cell disagrees between the built artifacts."""
    bad = tc.disagreements()
    assert not bad, f"{len(bad)} cross-table disagreements: {[k for k, _ in bad]}"
