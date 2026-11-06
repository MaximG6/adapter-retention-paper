"""Tests for the rendered-text gate.

Four separate rounds shipped a destroyed control sequence into the built PDF. Every
existing gate inspects sources, and two of the four were invisible there: a CR is
normalised away by `Path.read_text()`, and `ascii_only()` had already converted the
encoding damage into a literal `?` before the non-ASCII gate looked.

So this gate reads the rendered text, and it gets the treatment the instrument gate
gets: fed input known to be bad, and input known to be good, in both directions.
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


tc = _load("texcheck")


#: The exact sentences that reached the shipped PDF, round 5.
SHIPPED_DEFECTS = [
    ("times", "Six Taboo adapters imes four precisions imes 32 prompts"),
    ("ref-sec", "the intervals over them cluster by intent (efsec:stats)."),
    ("markdown-bold", 'were wrong; the panel itself printed "max error 0.0%".** Three'),
    ("backslash-macro", r"the reference class for \ref{sec:dissoc} and the floor"),
    ("bare-macro-brace", "same word. textbf{The elicitation score pools all 32"),
    ("sqrt-macro", r"amplification law sqrt{(d_in/r)} with c=0.87"),
    ("replacement-char", "the marker fired on � inside a vocabulary list"),
]


def test_every_check_fires_on_the_defect_it_was_written_for() -> None:
    for name, text in SHIPPED_DEFECTS:
        hits = {n for n, _, _ in tc.scan(text)}
        assert name in hits, f"{name!r} did not fire on {text!r} (fired: {hits})"


def test_clean_prose_passes() -> None:
    clean = (
        "Six Taboo adapters x four precisions x 32 prompts x two conditions = 1536 "
        "records. The base conditions are the reference class for the dissociation "
        "and the elicitation floor below. Retention is 99.2% with an enumerated 95% "
        "interval of [90.7%, 107.6%]."
    )
    assert tc.scan(clean) == []


def test_legitimate_code_listings_do_not_fire() -> None:
    """A gate that fires on correct input teaches its author to ignore it. This paper
    prints shell and Python in its appendices, and both contain braces and a bare
    `sqrt`."""
    listings = (
        "Supported: --bits {3,4,8}, --group-size {any positive int, or -1 for "
        "per-channel}, --scheme {asymmetric, symmetric_gptq, symmetric_awq}. "
        "Layer-output SNR applies the amplification law sqrt((d_in/r)/(1+c/r)) with "
        "c=0.87. gamma = alpha/sqrt(r) under rsLoRA. "
        "Does the secret word appear? Does the model still produce hints?"
    )
    assert tc.scan(listings) == []


def test_a_real_question_mark_is_not_debris() -> None:
    assert tc.scan("does the secret word appear?) and capability") == []


def test_a_correct_subscript_is_not_debris() -> None:
    """PyMuPDF extracts $p_{\text{refuse}}$ as "prefuse", because a subscript is
    smaller type rather than a character. An external review read the extracted text and
    reported it as a lost underscore; the rendered page is correct. A check for it would
    fire on every subscript in the paper, so there is none."""
    assert tc.scan("Instrument limits found by audit. prefuse responds to how") == []


def test_the_built_pdf_is_clean() -> None:
    """The gate on the artifact itself. Skipped when the PDF has not been built."""
    import pytest

    pdf = ROOT / "paper" / "adapter-retention-arxiv.pdf"
    if not pdf.exists():
        pytest.skip("PDF not built")
    hits = tc.scan(tc.pdf_text(pdf))
    assert not hits, f"{len(hits)} pieces of debris: {[(n, h) for n, h, _ in hits[:8]]}"
