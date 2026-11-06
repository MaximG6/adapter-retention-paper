"""Tests for the markdown-to-LaTeX converter and the overfull-box gate.

Both exist because a defect shipped: 15 of 25 appendix tables overran their column in
the built arXiv PDF, and 10 wrapped so narrowly they rendered one word per line. The
second kind never warns -- TeX wraps silently -- so the converter has to get the width
right rather than rely on the log.

The gate gets the same treatment the instrument gate does (methodological lessons,
"a validation gate must itself be tested against something already known to be broken"):
it is fed a log that must fail and a log that must pass. During development its regex
matched nothing and it reported a clean build over 150 real overfull boxes.
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


m2t = _load("md_to_tex")
build = _load("build_arxiv_pdf")


# --------------------------------------------------------------- the overfull gate

CLEAN = "warning: main.tex:12: Underfull \\hbox (badness 4000) in paragraph at lines 1--2"
BAD = ("warning: appendices:204: Overfull \\hbox (298.61pt too wide) in paragraph "
       "at lines 204--204")
TINY = ("warning: appendices:99: Overfull \\hbox (1.20pt too wide) in paragraph "
        "at lines 99--99")


def test_gate_rejects_a_known_bad_log() -> None:
    assert build.check_overfull(BAD, 6.0) is False


def test_gate_passes_a_clean_log() -> None:
    assert build.check_overfull(CLEAN, 6.0) is True


def test_gate_ignores_boxes_under_the_threshold() -> None:
    assert build.check_overfull(TINY, 6.0) is True


def test_gate_ignores_underfull_boxes() -> None:
    """Underfull is a spacing nicety, not content leaving the page."""
    assert build.check_overfull(CLEAN * 20, 6.0) is True


def test_gate_dedupes_repeated_passes() -> None:
    """Tectonic compiles twice, so every warning appears twice; the same box reported
    twice is one box."""
    assert build.check_overfull(BAD + "\n" + BAD, 6.0) is False
    assert build.check_overfull(BAD + "\n" + BAD, 400.0) is True


# ------------------------------------------------------------------- table layout

NARROW = ["| a | b |", "|---|---|", "| 1 | 2 |"]
WIDE = ["| " + " | ".join(f"col{i}" for i in range(11)) + " |",
        "|" + "---|" * 11,
        "| " + " | ".join("adamkarvonen/Qwen3-8B-taboo-smile" for _ in range(11)) + " |"]
PROSE = ["| id | prediction | outcome |", "|---|---|---|",
         "| P1 | " + "word " * 40 + " | " + "text " * 30 + " |"]


def test_narrow_table_stays_in_one_column() -> None:
    out = m2t.table(NARROW)
    assert "tabularx" not in out
    assert "table*" not in out
    assert "\\begin{center}" in out


def test_wide_table_is_promoted_to_full_width() -> None:
    out = m2t.table(WIDE)
    assert "\\begin{table*}" in out
    assert "\\textwidth" in out


def test_wide_table_uses_tabularx_so_the_total_is_the_container() -> None:
    """Estimated p{} widths mixed with natural l/r columns cannot promise a total;
    that combination is what overran by up to 298pt."""
    out = m2t.table(WIDE)
    assert "\\begin{tabularx}" in out


def test_prose_columns_get_width_in_proportion_to_demand() -> None:
    """The old converter split the width evenly, which gave the prediction column a
    15-character measure and rendered it one word per line."""
    out = m2t.table(PROSE)
    factors = [float(x) for x in
               __import__("re").findall(r"hsize=([0-9.]+)\\hsize", out)]
    assert len(factors) >= 2
    assert max(factors) > min(factors), "columns were weighted equally"


def test_hsize_factors_sum_to_the_number_of_x_columns() -> None:
    """tabularx only computes the right total if the weights sum to the X count."""
    import re
    for rows in (PROSE, WIDE):
        out = m2t.table(rows)
        factors = [float(x) for x in re.findall(r"hsize=([0-9.]+)\\hsize", out)]
        if factors:
            assert abs(sum(factors) - len(factors)) < 1e-2


def test_every_row_survives_layout() -> None:
    """No column or row may be dropped to make a table fit."""
    out = m2t.table(WIDE)
    assert out.count("\\\\") == 2                      # header + one body row
    assert out.count("&") == 2 * 10                    # 11 columns -> 10 separators


# ------------------------------------------------------------ verbatim and inline

def test_verbatim_shrinks_for_long_lines() -> None:
    assert m2t.verbatim_size(["short"]) == "footnotesize"
    assert m2t.verbatim_size(["x" * 55]) == "scriptsize"
    assert m2t.verbatim_size(["x" * 75]) == "tiny"


def test_bold_spanning_a_line_break_is_converted() -> None:
    """inline() is per-line, so a **bold** the author wrapped mid-phrase used to reach
    the PDF as literal asterisks -- 70 of them did."""
    out = m2t.convert("this is **bold\ntext** here")
    assert "**" not in out
    assert "\\textbf{bold text}" in out


def test_list_continuation_stays_in_its_item() -> None:
    """An indented continuation used to close the list and emit an orphan paragraph
    between two bullets."""
    out = m2t.convert("- first line\n  continued here\n- second")
    # "\\item" alone also matches \itemsep in the itemize preamble.
    assert out.count("\\item ") == 2
    assert "continued here" in out
    assert out.index("continued here") < out.index("\\end{itemize}")


def test_an_escaped_pipe_is_a_character_not_a_cell_break() -> None:
    """`\\|` is markdown's only way to write a literal bar in a table cell, and every
    table naming |delta| or |r| uses it. Split naively it became BOTH a spurious column
    and a `\\textbackslash{}` -- in the two tables carrying the 2.3% and 10.4% headlines,
    the same class as round 5's `imes`."""
    out = m2t.table([r"| a | b |", "|---|---|",
                     r"| mean(min(\|D\|/s, 1)) | 2.3% |"])
    assert "textbackslash" not in out
    assert out.count("&") == 2, "an escaped pipe opened extra columns"
    assert "|D|/" in out.replace("\\allowbreak{}", "")


def test_an_escaped_asterisk_is_not_emphasis_and_not_a_backslash() -> None:
    """A footnote marker written `\\*` reached the PDF as "\\textbackslash{}*"."""
    out = m2t.inline(r"\* first run only")
    assert "textbackslash" not in out
    assert out.startswith("*")
    assert "emph" not in out


def test_a_backslash_inside_a_code_span_is_content_not_an_escape() -> None:
    """The escape resolver, run before code spans were held out, read the `\\\\` in a
    span as an escaped backtick and emitted a raw control character into the tex. Tectonic
    rejected the line outright, which is the good case; the same class of bug that does
    not stop the build is what texcheck exists for."""
    out = m2t.inline(r"a trailing `\\` continuation")
    assert "\x02" not in out
    assert "textbackslash" in out, "the span's backslashes are content"


def test_paths_in_code_spans_may_break() -> None:
    out = m2t.inline("`results/raw/phase0/public_adapter/records.jsonl`")
    assert "allowbreak" in out


@pytest.mark.parametrize("bad", ["", "|", "|---|"])
def test_degenerate_tables_do_not_raise(bad: str) -> None:
    m2t.table([bad])


def test_source_reference_gate_fires_on_a_cut_entry(tmp_path, monkeypatch) -> None:
    """A reference to a practice entry the markdown no longer has must be caught HERE.

    The built-document gate cannot catch it: REFMAP translates the reference to some
    appendix subsection that does exist, so it resolves -- to the wrong entry. That is
    how the practice appendix ended up citing itself and pointing three references at
    entries that had been cut. Tested in both directions.
    """
    paper = tmp_path / "paper"
    paper.mkdir()
    # Round 8 moved the practice entries to METHODOLOGY.md and left the registered
    # predictions as the whole of Appendix C, so this file holds 7.0 and nothing else in
    # the shipped tree. The gate still has to fire on a reference to a heading that is
    # not there, which is the case a moved section creates.
    (paper / "07-registered-predictions.md").write_text(
        "## 7.0 Predictions\n\n## 7.1 A practice\n", encoding="utf-8")
    monkeypatch.setattr(m2t, "PAPER", paper)

    (paper / "other.md").write_text("As shown in S7.9 above.".replace("S", "§"),
                                    encoding="utf-8")
    assert m2t.check_source_refs() == [("7.9", "other.md")]

    (paper / "other.md").write_text("As shown in S7.1 above.".replace("S", "§"),
                                    encoding="utf-8")
    assert m2t.check_source_refs() == []
