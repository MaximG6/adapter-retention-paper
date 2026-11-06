"""Tests for the figure cross-check machinery. No GPU, no figures drawn.

A check that cannot fail is decoration. The regression case is the actual Fig 8 bug: a
set comprehension collected only the first member of each resolvable pair, so the figure
marked 2 adapters where the analysis reports 4, rendered without error, and dropped the
two that carry the paper's sign argument (§7.9).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from figcheck import Check, FigureCheckError  # noqa: E402


def test_matching_values_pass() -> None:
    # Expressions must differ textually or the vacuous guard fires -- correctly, since
    # `equal("n", 4, 4)` asserts nothing. Values agree; expressions do not.
    plotted_n, reference_n = 4, 2 + 2
    plotted_x, reference_x = 1.0, 1.0 + 1e-12
    c = Check("t")
    c.equal("n", plotted_n, reference_n)
    c.close_to("x", plotted_x, reference_x)
    c.close()  # must not raise


def test_the_historical_fig8_bug_is_rejected() -> None:
    # Exactly the values the buggy figure produced against what word_vs_noise.py reports.
    c = Check("fig08-regression")
    c.equal("n separating pairs", 2, 4)
    c.equal("marked adapters", ["moon", "snow"], ["gold", "moon", "ship", "snow"])
    with pytest.raises(FigureCheckError, match="disagree"):
        c.close()


def test_the_historical_fig6_estimator_swap_is_rejected() -> None:
    # Ratio-of-pooled-means (0.7810) vs mean-of-per-adapter-ratios (0.7716). Small, and
    # it made one figure disagree with the paper's own headline series.
    c = Check("fig06-regression")
    c.close_to("capability int4_per_channel", 0.780995, 0.771626, tol=1e-6)
    with pytest.raises(FigureCheckError):
        c.close()


def test_nan_is_a_failure_not_a_pass() -> None:
    # A plotted NaN must never satisfy a tolerance comparison.
    c = Check("t")
    c.close_to("x", float("nan"), 1.0, tol=1e6)
    with pytest.raises(FigureCheckError):
        c.close()


def test_length_mismatch_is_caught() -> None:
    c = Check("t")
    c.all_close("series", [1.0, 2.0], [1.0, 2.0, 3.0])
    with pytest.raises(FigureCheckError):
        c.close()


def test_error_message_names_the_figure_and_says_it_was_not_written() -> None:
    c = Check("fig99")
    c.equal("n", 1, 2)
    with pytest.raises(FigureCheckError, match="fig99"):
        c.close()
    c2 = Check("fig99")
    c2.equal("n", 1, 2)
    with pytest.raises(FigureCheckError, match="NOT written"):
        c2.close()


# ---- structural guards (added after the rule was violated by its own author) ----


def test_vacuous_check_raises_on_identical_expression(tmp_path: Path) -> None:
    """mean(X) vs mean(X) is the exact shape that shipped on 2026-08-03."""
    script = tmp_path / "vac.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, r'{Path(__file__).resolve().parents[1] / 'analysis'}')\n"
        "from figcheck import Check, VacuousCheckError\n"
        "vals = [1.0, 2.0, 3.0]\n"
        "c = Check('t')\n"
        "try:\n"
        "    c.close_to('x', sum(vals) / len(vals), sum(vals) / len(vals), tol=0)\n"
        "    print('NOFIRE')\n"
        "except VacuousCheckError:\n"
        "    print('FIRED')\n",
        encoding="utf-8")
    import subprocess
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert "FIRED" in out.stdout, out.stdout + out.stderr


def test_guard_does_not_fire_on_genuinely_different_expressions(
    tmp_path: Path,
) -> None:
    script = tmp_path / "ok.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, r'{Path(__file__).resolve().parents[1] / 'analysis'}')\n"
        "from figcheck import Check\n"
        "vals = [1.0, 2.0, 3.0]\n"
        "ref = 2.0\n"
        "c = Check('t')\n"
        "c.close_to('x', sum(vals) / len(vals), ref, tol=1e-9)\n"
        "c.close()\n"
        "print('OK')\n",
        encoding="utf-8")
    import subprocess
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert "OK" in out.stdout, out.stdout + out.stderr


def test_coverage_counts_all_close_by_its_length() -> None:
    # An all_close over six points covers six values, not one; otherwise the coverage
    # guard would fire on figures that are in fact fully checked.
    plotted = [1.0] * 6
    reference = [float(1) for _ in range(6)]
    c = Check("t")
    c.plots(6)
    c.all_close("series", plotted, reference)
    c.close()
    assert c.covered == 6


def test_low_coverage_warns_without_failing(capsys) -> None:
    c = Check("thin")
    c.plots(40)
    c.equal("one", 1, 1 + 0)
    c.close()  # must NOT raise; a warning is not a failure
    assert "LOW COVERAGE" in capsys.readouterr().out
