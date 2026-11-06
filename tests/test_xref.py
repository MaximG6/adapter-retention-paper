"""Tests for the cross-reference checker.

Section C.5 of the manuscript records a cross-reference checker that reported nine false
positives: its pattern required whitespace after the section number, and every heading in
the paper uses a period. A checker that is confidently wrong is worse than none, because
its author learns to discount it.

So this one is required to distinguish a reference known to be broken from one known to
be good, in both directions, and the period case is pinned explicitly.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    # analysis/ on the path, because gen_readme imports bootstrap and figcheck as
    # siblings and spec_from_file_location does not make them importable.
    if str(ROOT / "analysis") not in sys.path:
        sys.path.insert(0, str(ROOT / "analysis"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "analysis" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


xref = _load("xref")

MAIN = r"""
\section{Introduction}
Text referring to \S2 and \S2.1 and Figure~1.
\section{Method}
\subsection{Setting}
\subsection{Statistics}
\begin{figure}\end{figure}
\begin{equation}x=1\end{equation}
"""

APX = r"""
\appendix
\section{The Tool}
\subsection{Accuracy}
\section{Full Tables}
"""


def test_structure_numbers_like_latex() -> None:
    secs, apps, titles = xref.structure(MAIN, APX)
    assert secs == {"1", "2", "2.1", "2.2"}
    assert apps == {"A", "A.1", "B"}
    assert titles["2.1"] == "Setting"


def test_good_reference_resolves() -> None:
    assert xref.check(MAIN, APX) == []


def test_known_bad_section_is_flagged() -> None:
    bad = xref.check(MAIN + r"\newline See \S7.4 for detail.", APX)
    assert [t for k, t, _ in bad if k == "section"] == ["7.4"]


def test_known_bad_appendix_is_flagged() -> None:
    bad = xref.check(MAIN + r"\newline listed in Appendix~E.", APX)
    assert [t for k, t, _ in bad if k == "appendix"] == ["E"]


def test_reference_ending_a_sentence_is_still_resolved() -> None:
    """The exact defect of the earlier checker: a pattern demanding whitespace after the
    number treats "\\S2.1." as unresolved because a period follows."""
    ok = xref.check(MAIN + r"\newline as shown in \S2.1.", APX)
    assert ok == [], f"period after a valid reference was misread: {ok}"


def test_reference_in_parentheses_is_resolved() -> None:
    assert xref.check(MAIN + r"\newline (\S2.2)", APX) == []


def test_figure_over_count_is_flagged() -> None:
    bad = xref.check(MAIN + r"\newline see Figure~9", APX)
    assert [t for k, t, _ in bad if k == "figure"] == ["9"]


def test_appendix_subsection_is_checked() -> None:
    assert xref.check(MAIN + r"\newline Appendix~A.1", APX) == []
    bad = xref.check(MAIN + r"\newline Appendix~A.7", APX)
    assert [t for k, t, _ in bad if k == "appendix"] == ["A.7"]


def test_equation_reference_resolves() -> None:
    assert xref.check(MAIN + r"\newline Equation~(1)", APX) == []
    bad = xref.check(MAIN + r"\newline Equation~(4)", APX)
    assert [t for k, t, _ in bad if k == "equation"] == ["4"]


def test_the_built_paper_has_no_unresolved_references() -> None:
    """The gate itself. Fails while any reference in the built document dangles."""
    main = (ROOT / "paper" / "tex" / "main.tex").read_text(encoding="utf-8")
    apx = (ROOT / "paper" / "tex" / "appendices.tex").read_text(encoding="utf-8")
    bad = xref.check(main, apx)
    summary = sorted({f"{k} {t}" for k, t, _ in bad})
    assert not bad, f"{len(bad)} unresolved: {summary[:20]}"


def test_bare_appendix_reference_is_checked() -> None:
    """The M17 class: "see D.1.2" is the same reference as "Appendix D.1.2".

    Both the appendix-letter remapper and this gate keyed on the literal word
    "Appendix", so four live references in the reproduction appendix pointed at the
    safety-adapter appendix and nothing objected. Pinned in both directions.
    """
    main = ("\\section{One}\n\\appendix\n\\section{Alpha}\n"
            "\\subsection{First}\n")
    bad = xref.check(main, "The breakdown is in A.9.")
    assert [t for k, t, _ in bad if k == "bare-appendix"] == ["A.9"]
    assert not [t for k, t, _ in xref.check(main, "The breakdown is in A.1.")
                if k == "bare-appendix"]


def test_bare_pattern_does_not_fire_on_non_references() -> None:
    """Version strings, model names, paths and decimals are not appendix references."""
    main = ("\\section{One}\n\\appendix\n\\section{Alpha}\n"
            "\\subsection{First}\n")
    text = "Qwen3-8B.1 and v2.1 and results/D.7 and 0.15 and torch 2.11.0"
    assert not [t for k, t, _ in xref.check(main, text) if k == "bare-appendix"]


def test_markdown_appendix_labels_match_the_latex_numbering() -> None:
    """The appendix markdown writes its own labels and the arXiv build lets LaTeX count.
    Those agree only while the labels are sequential from 1, so inserting a section in the
    middle silently shifts every later one. It happened twice -- a `B.6b` between B.6 and
    B.7, and a `D.7a` -- and `check` could not see either, because every shifted target
    still exists and every reference still resolves, to the wrong section."""
    assert not xref.numbering_drift()


def test_the_numbering_gate_fires_on_an_inserted_subsection() -> None:
    """Against the state that shipped: one heading inserted between B.6 and B.7, and
    every later label is one short. Fed the real Appendix B with that insertion restored,
    the gate must flag the insertion and every subsection after it."""
    src = (ROOT / "paper" / "appendix-B-tables.md").read_text(encoding="utf-8")
    assert not xref.drift_in("b.md", src), "the committed appendix is sequential"

    lines = src.splitlines()
    at = next(i for i, x in enumerate(lines) if x.startswith("## B.7 "))
    broken = "\n".join(lines[:at] + ["## B.6b An inserted section", ""] + lines[at:])
    bad = xref.drift_in("b.md", broken)
    assert [b[1] for b in bad][0] == "B.6b"
    assert len(bad) == 8, f"one insertion should shift seven later labels, got {bad}"


def test_the_numbering_gate_passes_a_sequential_appendix() -> None:
    """A gate that fires on correct input is the failure this project has recorded
    repeatedly, so the pass direction is a test too."""
    good = "\n".join(f"## C.{i} Heading {i}" for i in range(1, 6))
    assert not xref.drift_in("c.md", good)


def test_every_translated_reference_lands_on_the_section_it_is_about() -> None:
    """Existence is not correctness. `check` passes any map at all, including one with a
    hole in it: `5.4 -> 5.3` was in the table and `5.3 -> 5.2` was not, so three
    references to the dissociation section resolved -- in the built paper -- to the
    predictive gap, one section past their subject."""
    assert not xref.section_alignment()


def test_the_alignment_gate_fires_on_the_map_as_it_shipped(monkeypatch) -> None:
    """Against the state that shipped, not a synthetic one: remove the two entries that
    were missing and the gate must name 5.3 and the section it should point at."""
    import importlib.util
    real = importlib.util.spec_from_file_location

    def patched(name: str, path):
        spec = real(name, path)
        if "md_to_tex" in str(path):
            inner = spec.loader.exec_module

            def exec_module(mod):
                inner(mod)
                mod.REFMAP = {k: v for k, v in mod.REFMAP.items()
                              if k not in ("5.2", "5.3")}
            spec.loader.exec_module = exec_module
        return spec

    monkeypatch.setattr(importlib.util, "spec_from_file_location", patched)
    bad = xref.section_alignment()
    assert [(b[0], b[1], b[2]) for b in bad] == [("5.3", "5.3", "5.2")]


def test_companion_documents_are_inside_the_reference_gate() -> None:
    """Round 8 moved content out of the PDF. If the companions leave the reference gate
    they become the next main.tex -- a document nothing resolves references against."""
    main = (ROOT / "paper" / "tex" / "main.tex").read_text(encoding="utf-8")
    apx = (ROOT / "paper" / "tex" / "appendices.tex").read_text(encoding="utf-8")
    assert not xref.companion_refs(main, apx)


def test_the_boundary_gate_fires_in_both_directions(tmp_path, monkeypatch) -> None:
    """A companion citing a section the paper does not have, and the paper citing a
    heading the companion does not have. Both dangled silently before this existed: the
    prompt document cited Appendix C, which is the prompt appendix in the report and the
    registered predictions in the arXiv build."""
    monkeypatch.setattr(xref, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(xref, "COMPANIONS", ("METHODOLOGY.md",))
    (tmp_path / "METHODOLOGY.md").write_text("## M.1 A practice\n\nSee S4.9.\n"
                                             .replace("S", "\u00a7"), encoding="utf-8")
    main = "\\section{One}\n\\subsection{Bit}\n"
    bad = xref.companion_refs(main, "")
    assert [b[1] for b in bad] == ["\u00a74.9"]

    (tmp_path / "METHODOLOGY.md").write_text("## M.1 A practice\n\nSee S1.1.\n"
                                             .replace("S", "\u00a7"), encoding="utf-8")
    assert not xref.companion_refs(main, "")

    # And the other way: the paper naming a heading the companion does not define.
    assert [b[1] for b in xref.companion_refs("See METHODOLOGY.md M.7 for it.\n"
                                              + main, "")] == ["M.7"]
    assert not xref.companion_refs("See METHODOLOGY.md M.1 for it.\n" + main, "")

def test_the_title_gate_flags_the_title_that_actually_shipped(tmp_path,
                                                              monkeypatch) -> None:
    r"""The README's BibTeX named a work the paper is not. Pinned in the form it
    shipped, because a citation block is the one part of a repository a reader copies
    without checking."""
    gen_readme = _load("gen_readme")

    tex = tmp_path / "paper" / "tex"
    tex.mkdir(parents=True)
    (tex / "main.tex").write_text(
        "\\title{\\bfseries Weight-Space Erasure Without Behavioural Collapse\n"
        "in Quantized LoRA Adapters\\vspace{-0.3em}}\n\\author{Maxim}\n",
        encoding="utf-8")
    monkeypatch.setattr(gen_readme, "MAIN_TEX", tex / "main.tex")
    monkeypatch.setattr(xref, "REPO_ROOT", tmp_path)

    assert gen_readme.paper_title() == (
        "Weight-Space Erasure Without Behavioural Collapse in Quantized LoRA Adapters")

    shipped = ("  title  = {Near-Total Weight-Space Erasure Without Behavioural "
               "Collapse:\n            What Survives When a Merged LoRA Is "
               "Quantized},\n")
    (tmp_path / "README.md").write_text(shipped, encoding="utf-8")
    bad = xref.title_disagreements()
    assert [w for w, _ in bad] == ["README.md"]
    assert bad[0][1].startswith("Near-Total")

    (tmp_path / "README.md").write_text(
        "  title  = {Weight-Space Erasure Without Behavioural Collapse in "
        "Quantized LoRA Adapters},\n", encoding="utf-8")
    assert not xref.title_disagreements()

    # And a CITATION.cff that drifts on its own, since it is a separate artifact.
    (tmp_path / "CITATION.cff").write_text(
        "title: >-\n  Something Else Entirely\nauthors:\n", encoding="utf-8")
    assert [w for w, _ in xref.title_disagreements()] == ["CITATION.cff"]


def test_the_title_gate_raises_rather_than_guessing_when_the_title_is_gone(
        tmp_path, monkeypatch) -> None:
    """A missing \\title must raise, not return an empty string that every artifact
    then agrees with -- a gate that passes when its reference disappears is M.5."""
    import pytest

    gen_readme = _load("gen_readme")

    tex = tmp_path / "paper" / "tex"
    tex.mkdir(parents=True)
    (tex / "main.tex").write_text("\\author{Maxim}\n", encoding="utf-8")
    monkeypatch.setattr(gen_readme, "MAIN_TEX", tex / "main.tex")
    with pytest.raises(RuntimeError):
        gen_readme.paper_title()

def _author_fixture(tmp_path, monkeypatch, rendered: str = "Maxim Gerasimov"):
    gen_readme = _load("gen_readme")
    tex = tmp_path / "paper" / "tex"
    tex.mkdir(parents=True)
    (tex / "main.tex").write_text(
        "\\title{A Title}\n\\author{" + rendered +
        "\\thanks{Independent researcher. Correspondence: \\texttt{a@b.c}. Code:"
        " \\url{https://example.invalid}}\\vspace{-0.6em}}\n\\date{\\today}\n",
        encoding="utf-8")
    monkeypatch.setattr(gen_readme, "MAIN_TEX", tex / "main.tex")
    monkeypatch.setattr(xref, "REPO_ROOT", tmp_path)
    return gen_readme


def test_the_author_parser_strips_the_thanks_footnote(tmp_path, monkeypatch) -> None:
    r"""\thanks holds a sentence with an email and a URL in it. A general command strip
    that ran first would leave that sentence in the author name."""
    gen_readme = _author_fixture(tmp_path, monkeypatch)
    assert gen_readme.paper_author() == "Maxim Gerasimov"


def test_the_author_split_is_checked_against_the_paper(tmp_path, monkeypatch) -> None:
    """Declaring given/family and rendering the name are two statements about one
    person, and nothing else in the repository can tell they have diverged. A
    disagreement must raise rather than pick a winner: guessing puts a fabricated name
    on a paper."""
    import pytest

    gen_readme = _author_fixture(tmp_path, monkeypatch, rendered="Maksim Gerasimov")
    with pytest.raises(RuntimeError, match="fabricated"):
        gen_readme.paper_author()


def test_the_author_gate_flags_each_artifact_that_drifts(tmp_path, monkeypatch) -> None:
    """LICENSE is not generated by anything, so it is the one most able to drift and the
    one a reader is most likely to trust."""
    _author_fixture(tmp_path, monkeypatch)

    (tmp_path / "LICENSE").write_text(
        "MIT License\n\nCopyright (c) 2026 Maxim <a@b.c>\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "  author = {Gerasimov, Maxim},\n", encoding="utf-8")
    (tmp_path / "CITATION.cff").write_text(
        "authors:\n  - family-names: Gerasimov\n    given-names: Maxim\n",
        encoding="utf-8")
    assert [w for w, _ in xref.author_disagreements()] == ["LICENSE"]

    (tmp_path / "LICENSE").write_text(
        "MIT License\n\nCopyright (c) 2026 Maxim Gerasimov <a@b.c>\n", encoding="utf-8")
    assert not xref.author_disagreements()

    # The bare given name the repository shipped with, in the BibTeX field.
    (tmp_path / "README.md").write_text("  author = {Maxim},\n", encoding="utf-8")
    assert xref.author_disagreements() == [("README.md", "Maxim")]

    (tmp_path / "README.md").write_text(
        "  author = {Gerasimov, Maxim},\n", encoding="utf-8")
    (tmp_path / "CITATION.cff").write_text(
        "authors:\n  - family-names: Maxim\n    given-names: Gerasimov\n",
        encoding="utf-8")
    assert xref.author_disagreements() == [("CITATION.cff", "Gerasimov Maxim")]

def test_the_parser_survives_anything_written_between_the_macros(tmp_path,
                                                                 monkeypatch) -> None:
    r"""Pinning the state that broke it. The first version read \title up to \author and
    \author up to \date. Pinning the date put a comment block and a \newcommand between
    \author and \date, and the author check then reported the comment as part of the
    name and failed the build. Brace matching does not care what sits between."""
    gen_readme = _load("gen_readme")
    tex = tmp_path / "paper" / "tex"
    tex.mkdir(parents=True)
    (tex / "main.tex").write_text(
        "\\title{\\bfseries A Title\\vspace{-0.3em}}\n"
        "% a comment mentioning \\title and \\date to be awkward\n"
        "\\author{Maxim Gerasimov\\thanks{Independent researcher. Mail: "
        "\\texttt{a@b.c}}\\vspace{-0.6em}}\n"
        "% Pinned, not \\today. Bump deliberately.\n"
        "\\newcommand{\\paperdate}{August 7, 2026}\n"
        "\\date{\\small\\paperdate}\n",
        encoding="utf-8")
    monkeypatch.setattr(gen_readme, "MAIN_TEX", tex / "main.tex")

    assert gen_readme.paper_title() == "A Title"
    assert gen_readme.paper_author() == "Maxim Gerasimov"


def test_the_date_is_pinned_rather_than_today() -> None:
    r"""\today renders a different title page from the same commit on a different day,
    so a reader cannot tell which build they hold and two people comparing "the paper"
    cannot tell whether they have the same one."""
    main = (ROOT / "paper" / "tex" / "main.tex").read_text(encoding="utf-8")
    # Comments out first: the note explaining why the date is pinned says \today, and a
    # gate that fires on the documentation of its own rule is one nobody keeps.
    live = "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in main.splitlines())
    assert "\\today" not in live, "the title page date must be pinned, not \\today"
    assert re.search(r"\\newcommand\{\\paperdate\}\{[A-Z][a-z]+ \d{1,2}, \d{4}\}", live)
