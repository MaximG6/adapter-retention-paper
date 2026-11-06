"""Tests for the retracted-wording gate.

The gate exists because four of one review round's findings were the same shape: a
correction landed where it was discovered and not where the claim is asserted. Nothing
else in the perimeter can see that -- the claim audit reads numbers, `countcheck` reads
count words, `xref` reads references, and a retracted *sentence* has none of those.

So this gate has to distinguish an assertion of a retired wording from a quotation of one,
because every correction in this project quotes the wording it retires. Both directions
are pinned, and the gate is fed a document known to be bad before it is trusted on a
document believed to be good (`METHODOLOGY.md` M.3).
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


rt = _load("retracted")


def test_a_bare_assertion_is_flagged() -> None:
    bad = rt.assertions_in(
        "Treating the battery as independent counts roughly 16 independent units twice.")
    assert [b[0] for b in bad] == [r"roughly 16 independent units"]


def test_a_quoted_retraction_is_not_flagged() -> None:
    """How every correction in this project is written, and it must not trip the gate."""
    assert not rt.assertions_in(
        'This appendix said the battery carries "roughly 16 independent units", '
        "which is the ICC = 1 case.")


def test_the_python_escaped_form_is_recognised_as_a_quotation() -> None:
    """The generators hold their own retractions inside string literals, where the quote
    marks are backslash-escaped. Pairing the literal's opening quote with the first
    escaped one put the retraction outside every quoted span and flagged two generators
    that were correct."""
    assert not rt.assertions_in(
        '        "This appendix said the 32 "\n'
        '        "prompts carry \\"roughly 16 independent units\\", which is wrong. "')


def test_a_latex_assertion_is_flagged() -> None:
    bad = rt.assertions_in(
        r"\textbf{Second, nothing is missing}: nine were registered and nine appear.")
    assert len(bad) == 1


def test_the_two_sided_headline_cannot_come_back() -> None:
    for s in ("there is no detectable change in their trained behaviour",
              "On the same models, no behavioural change is\ndetectable."):
        assert rt.assertions_in(s), f"not flagged: {s!r}"


def test_the_cosine_snr_conflation_cannot_come_back() -> None:
    assert rt.assertions_in("a weight-space cosine of 0.13 corresponds to an output SNR")
    assert not rt.assertions_in("a weight-space SNR of 0.13 corresponds to an output SNR")


def test_a_figure_panel_header_is_not_hidden_by_its_own_quotes() -> None:
    """In-panel figure text is a string literal, so on raw bytes every character of it
    sits inside a quoted span and the gate is structurally unable to see it. That is the
    one defect on this list a reader would have taken away backwards, so the Python
    perimeter is parsed rather than scanned."""
    src = 'panels = [(axes[0], w, ci, vals, ERASE, "Stored weights UNCHANGED", sub)]\n'
    assert not rt.assertions_in(src), "raw scan sees it as a quotation, as expected"
    assert [b[0] for b in rt.assertions_in_python(src)] == [r"Stored weights UNCHANGED"]


def test_a_quoted_retraction_inside_a_literal_is_still_a_quotation() -> None:
    src = ('MSG = "an earlier draft called this column \\"flat at 0.0757 and 0.0756\\""\n')
    assert not rt.assertions_in_python(src)


def test_a_comment_is_inside_the_python_perimeter() -> None:
    """Four of the ICC correction's surviving assertions were in comments, which `ast`
    discards entirely."""
    src = "# the battery counts roughly 16 independent units\nx = 1\n"
    assert [b[0] for b in rt.assertions_in_python(src)] == [
        r"roughly 16 independent units"]


def test_every_pattern_matches_its_own_exemplar() -> None:
    """A pattern with a typo in it sits in the table matching nothing while the gate
    reports clean -- a check that cannot fail, which is the family `METHODOLOGY.md` M.5
    is about. Each entry carries the wording it retired, and must flag it."""
    for pat, example, _, where in rt.RETRACTED:
        bad = rt.assertions_in(example)
        assert [b[0] for b in bad] == [pat], (
            f"pattern for {where} does not flag its own exemplar: {example!r}")


def test_no_exemplar_matches_a_pattern_it_does_not_belong_to() -> None:
    """Two entries matching the same text would make one of them unfalsifiable: fixing
    the wording for one would silence the other's gate too."""
    for pat, example, _, where in rt.RETRACTED:
        hits = {b[0] for b in rt.assertions_in(example)}
        assert hits == {pat}, f"{where}'s exemplar also matches {hits - {pat}}"


def test_the_shipped_perimeter_asserts_no_retracted_wording() -> None:
    """The gate itself."""
    bad = rt.check()
    assert not bad, "\n".join(f"{r}: {p}  ...{c}..." for r, p, _, _, c in bad)
