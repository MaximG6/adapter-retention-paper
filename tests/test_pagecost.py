"""Tests for the page-cost instrument.

It had no tests for three review rounds because it was a scratch script, and it was wrong
the whole time: a section's span ran to the next *labelled* heading, and nothing labelled
follows the last one, so the bibliography's 1.83 pages were charged to the Conclusion.
Rounds 8, 9 and 10 each sized a cut against that, and so did an external reviewer's plan.

Nothing caught it because the number looked plausible and no check compared the sum of the
parts against the document. `METHODOLOGY.md` M.3 says a gate must be tested against
known-bad input; the lesson here is that **an instrument that sizes a decision is a gate**
(M.10). So the defect is pinned as a test, in the form that quantifies it.
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


pc = _load("pagecost")

#: Three numbered sections, then an unnumbered bibliography, then an appendix.
#: Column positions; the document ends at 20.0.
MARKS = {"9": 4.0, "10": 6.0, "A": 12.0}
TERMS = {"References": 8.0}
END = 20.0


def test_a_span_ends_at_the_next_mark_of_any_kind() -> None:
    span = pc.span_of(MARKS, TERMS, END)
    assert span["9"] == 2.0
    assert span["10"] == 2.0, "the Conclusion ends where the bibliography starts"
    assert span["References"] == 4.0
    assert span["A"] == 8.0


def test_the_defect_that_shipped_is_pinned_and_quantified() -> None:
    """The old behaviour, exactly: labels only, so the last numbered section absorbs
    everything up to the first appendix. The difference IS the bibliography."""
    broken = pc.span_of(MARKS, {}, END)
    fixed = pc.span_of(MARKS, TERMS, END)
    assert broken["10"] == 6.0
    assert broken["10"] - fixed["10"] == fixed["References"], (
        "the inflation is exactly the block that was invisible")
    assert broken["10"] / fixed["10"] == 3.0


def test_every_column_is_attributed_exactly_once() -> None:
    """The check that was missing. Spans that overrun cannot be seen row by row -- one
    row is too big and no other row disagrees with it -- but they always show in the
    total."""
    span = pc.span_of(MARKS, TERMS, END)
    first = min(list(MARKS.values()) + list(TERMS.values()))
    assert abs(sum(span.values()) - (END - first)) < 1e-9


def test_a_trailing_terminator_does_not_run_past_the_document() -> None:
    span = pc.span_of({"1": 0.0}, {"References": 5.0}, 6.0)
    assert span["References"] == 1.0


def test_the_built_paper_parts_do_not_exceed_the_document() -> None:
    """Against the live PDF. Failing means some span is running past its section, which
    is the shipped defect's signature."""
    try:
        found, terms, _titles, pages = pc.measure()
    except (ImportError, FileNotFoundError):  # no fitz, or no built PDF
        import pytest
        pytest.skip("built PDF or PyMuPDF unavailable")
    span = pc.span_of(found, terms, pages * 2.0)
    assert sum(span.values()) <= pages * 2.0 + 1e-6


def test_the_bibliography_is_found_and_is_not_charged_to_the_conclusion() -> None:
    """The specific regression. The Conclusion is half a page of prose; it read 2.35."""
    try:
        found, terms, _titles, pages = pc.measure()
    except (ImportError, FileNotFoundError):
        import pytest
        pytest.skip("built PDF or PyMuPDF unavailable")
    assert "References" in terms, "the bibliography was not located"
    span = pc.span_of(found, terms, pages * 2.0)
    assert span["10"] / 2 < 1.0, f"Conclusion reads {span['10'] / 2:.2f} pp"
    assert span["References"] / 2 > 1.0
