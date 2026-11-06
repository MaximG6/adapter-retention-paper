"""Tests for the measured figure layout.

The defect: Figure 1's caption was drawn through "Intended update RETAINED". The caption
sat at a fixed `y=0.885` and the axes at a fixed `subplots_adjust(top=0.685)`, tuned
together against a three-line caption. The non-detection reframe took it to five lines
and the fifth landed on the panel titles.

No gate could see it. `figcheck` cross-checks every plotted value against the raw
records and passed throughout, because every number was right; what was wrong was that
the text was unreadable. A prose edit broke a layout and nothing in the perimeter looks
at layout.

So the fix is measured rather than re-tuned, and these tests pin both directions: a
caption long enough to collide must be detected and resolved, and a figure that is
already comfortable must not be moved.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    if str(ROOT / "analysis") not in sys.path:
        sys.path.insert(0, str(ROOT / "analysis"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "analysis" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fl = _load("figlayout")


#: Realistic line length matters: `Bbox.overlaps` needs intersection in both axes, and a
#: panel title is centred over its axes while the caption is left-aligned at x=0.055. A
#: short caption line ends before the title begins and genuinely does not collide -- so a
#: fixture with stub lines tests nothing, which is how the first version of these tests
#: reported the shipped defect as clean.
LINE = ("capability is detectable and the interval spans parity, so this is a bound "
        "rather than an equality")


def _figure(caption_lines: int, top: float = 0.685, title: str = "Intended update"):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 5.4))
    fig.subplots_adjust(top=top)
    for ax in axes:
        ax.set_title(title, fontsize=12, pad=10)
    fig.suptitle("A headline", fontsize=15, x=0.055, ha="left", y=0.975)
    fig.text(0.055, 0.885, "\n".join(LINE for _ in range(caption_lines)),
             fontsize=8.6, va="top", linespacing=1.55)
    return fig


def test_a_caption_that_grew_is_detected() -> None:
    """The shipped state: five caption lines against a `top` chosen for three."""
    fig = _figure(5)
    bad = fl.overlaps(fig)
    assert bad, "a five-line caption at y=0.885 must collide with titles at top=0.685"
    assert any("title" in where for _, where in bad)
    plt.close(fig)


def test_fitting_resolves_it() -> None:
    fig = _figure(5)
    assert fl.overlaps(fig)
    fl.fit_below_texts(fig)
    assert not fl.overlaps(fig)
    fl.assert_no_overlap(fig, "fitted")   # must not raise
    plt.close(fig)


def test_a_comfortable_figure_is_left_alone() -> None:
    """`top` is only ever lowered. A figure whose layout already works must come out of
    this unchanged, or every existing figure silently reflows."""
    fig = _figure(1, top=0.60)
    before = fig.subplotpars.top
    assert not fl.overlaps(fig)
    fl.fit_below_texts(fig)
    assert fig.subplotpars.top == before
    plt.close(fig)


def test_multi_line_titles_are_measured_not_estimated() -> None:
    """fig09's titles are two lines. The first version of `fit_below_texts` derived the
    title height from its font size, which is right for one line and short for two, and
    fig09 stayed broken after being 'fixed'."""
    fig = _figure(4, title="int4_per_channel\n2 of 15 pairs separate")
    fl.fit_below_texts(fig)
    assert not fl.overlaps(fig), "two-line titles must be measured, not predicted"
    plt.close(fig)


def test_a_footer_does_not_drag_the_axes_onto_itself() -> None:
    """Figure-level text below the axes is a footnote. Treating it as a header would
    lower `top` toward it and converge on nonsense."""
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.subplots_adjust(top=0.90, bottom=0.20)
    ax.set_title("A title", fontsize=12, pad=10)
    fig.text(0.055, 0.03, "a footnote under the axes", fontsize=8)
    before = fig.subplotpars.top
    fl.fit_below_texts(fig)
    assert fig.subplotpars.top == before
    plt.close(fig)


def test_assert_raises_and_names_the_collision() -> None:
    fig = _figure(5)
    with pytest.raises(RuntimeError, match="overlap"):
        fl.assert_no_overlap(fig, "fig01_erasure_vs_survival")
    plt.close(fig)


def test_every_shipped_figure_is_clean_in_both_modes() -> None:
    """The gate itself, over the real figure scripts. Both modes, because the arXiv
    build suppresses in-figure headers and the two layouts are not the same figure."""
    import os
    import subprocess

    for paper_mode in ("0", "1"):
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"),
                   AR_FIG_PAPER=paper_mode, MPLBACKEND="Agg")
        r = subprocess.run(
            [sys.executable, "-c",
             "import sys;sys.path.insert(0,r'%s');sys.argv=['x']\n"
             "import fig01_erasure_vs_survival as a, fig05_06_08 as b, fig_secondary as c\n"
             "for m in (a,b,c): m.EXTS=()\n"
             "a.main();b.main();c.main()" % (ROOT / "analysis")],
            capture_output=True, text=True, env=env, cwd=ROOT)
        assert r.returncode == 0, (
            f"AR_FIG_PAPER={paper_mode}: {r.stderr[-1500:]}")
