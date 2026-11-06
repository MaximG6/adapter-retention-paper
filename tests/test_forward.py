"""Tests for the forward-claim gate.

`retracted.py` closes propagation failure on retraction. This closes it on **addition**,
and the two are structurally different: a retraction leaves a wording behind to search
for, an addition leaves nothing at all. The round that named partial propagation, built
`retracted.py` for it, and wrote the METHODOLOGY entry about it then committed the
addition variant in the same commits -- `[+4.2, +12.5]` reached the abstract, the
introduction, Figure 1's caption and the Conclusion while §5.1, the section whose whole
subject is that contrast, never stated it.

The gate is therefore fed the tree exactly as it stood at that commit, not a synthetic
approximation of it, and is required to name the interval at all four sites.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "analysis" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fw = _load("forward")

BODY = r"""
\section{Method}
The measured leak rate falls from 16.7\% to 8.3\%, a paired difference of +8.3 points
with an enumerated 95\% interval of [+4.2, +12.5]. The knowledge score is 0.3634.
"""


def test_a_summary_claim_with_a_source_resolves() -> None:
    main = (r"\begin{abstract}The leak rate falls 8.3 points, [+4.2, +12.5]."
            r"\end{abstract}" + BODY)
    assert not fw.unsourced(main, "")


def test_a_summary_claim_with_no_source_is_flagged() -> None:
    main = (r"\begin{abstract}The leak rate falls 8.3 points, [+4.2, +12.5]."
            r"\end{abstract}"
            "\n\\section{Method}\nThe leak rate falls from 16.7\\% to 8.3\\%.\n")
    bad = fw.unsourced(main, "")
    assert [(s, c) for s, c, _ in bad] == [("abstract", "[4.2, 12.5]")]


def test_an_interval_is_not_satisfied_by_its_ends_appearing_apart() -> None:
    """A gate on bare numbers would pass the defect it was written for: 4.2 in one
    appendix and 12.5 in another satisfies neither the claim nor a reader."""
    main = (r"\begin{abstract}an interval of [+4.2, +12.5].\end{abstract}"
            "\n\\section{Method}\nThe value is 4.2 here.\n")
    apx = "\\section{Tables}\n" + "filler " * 80 + "and 12.5 there.\n"
    bad = fw.unsourced(main, apx)
    assert [c for _, c, _ in bad] == ["[4.2, 12.5]"]


def test_a_summary_may_round_what_the_body_measured() -> None:
    """0.363 for a measured 0.3634 is ordinary prose. A gate demanding digit-identical
    restatement would force every summary to quote at appendix precision."""
    main = r"\begin{abstract}the base model falls to 0.363.\end{abstract}" + BODY
    assert not fw.unsourced(main, "")


def test_rounding_tolerance_does_not_swallow_a_real_gap() -> None:
    main = r"\begin{abstract}the base model falls to 0.372.\end{abstract}" + BODY
    assert [c for _, c, _ in fw.unsourced(main, "")] == ["0.372"]


def test_section_and_figure_numbers_are_not_claims() -> None:
    main = (r"\begin{abstract}as shown in \S\ref{sec:x} and Figure~3 and Appendix~B.11"
            r" and Equation~4 and INT4 and sm\_120.\end{abstract}" + BODY)
    assert not fw.unsourced(main, "")


def test_a_caption_is_its_own_site_and_is_not_double_reported() -> None:
    """A figure floated into §1 would otherwise have every number in its caption reported
    twice under two names, which trains a reader of this output to skim it."""
    main = (r"\section{Introduction}" "\n"
            r"\begin{figure}\caption{the rate is 0.372.}\end{figure}" "\n"
            r"\section{Method}" "\n" + BODY)
    bad = fw.unsourced(main, "")
    assert [s for s, _, _ in bad] == ["caption 1"]


def test_a_caption_body_is_brace_balanced() -> None:
    """A regex to the first `}` truncates every caption in this paper at its first
    \\texttt{}, so the numbers after it would never be checked."""
    caps = fw._captions(r"\caption{a \texttt{x} b 0.372 c}")
    assert caps == [r"a \texttt{x} b 0.372 c"]


FIXTURE = ROOT / "tests" / "fixtures" / "forward_summary_gap"


def test_the_gate_fires_on_the_gap_as_it_actually_shipped() -> None:
    """The original historical inputs must fail at all four summary sites."""
    def show(path: str) -> str:
        return (FIXTURE / Path(path).name).read_text(encoding="utf-8")

    bad = fw.unsourced(show("paper/tex/main.tex"), show("paper/tex/appendices.tex"))
    sites = sorted(s for s, c, _ in bad if c == "[4.2, 12.5]")
    assert sites == ["abstract", "caption 1", "conclusion", "introduction"], sites


def test_the_shipped_paper_has_no_unsourced_summary_claim() -> None:
    """The gate itself."""
    bad = fw.check()
    assert not bad, "\n".join(f"[{s}] {c}  ...{ctx}..." for s, c, ctx in bad)
