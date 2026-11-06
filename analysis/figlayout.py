"""Place a figure's axes below its rendered caption, and refuse to ship overlaps.

**Why this exists.** Every figure in this project put its caption at a fixed `y=0.885`
with `va="top"` and its axes at a fixed `subplots_adjust(top=...)`. Those two numbers
were tuned together once, against a caption of a particular length. A caption grows
downward, the axes do not move, and eventually the last caption line is drawn through
the panel titles.

That is what happened to Figure 1. The non-detection reframe added a clause about the
interval spanning parity and another naming §5.1, taking the caption from three lines to
five, and the fifth was rendered across "Intended update RETAINED". The figure remained
*correct* the whole time -- every plotted value was right, and `figcheck` compares
plotted values against the raw records, so it passed. Nothing in the perimeter looks at
whether the text is readable.

**The fix has to be measured, not re-tuned.** Picking a new `top` by eye restores the
same fragility one caption edit later. So the caption is drawn, its rendered extent is
measured, and the axes are placed below it -- and `top` is only ever *lowered*, never
raised, so a figure whose layout is already comfortable is untouched.

`assert_no_overlap` is the gate: after everything is drawn, any figure-level text whose
rendered box intersects an axes title or an axes rectangle raises. A layout defect stops
the build rather than reaching a PDF.
"""
from __future__ import annotations

from typing import Any

#: Clear space between the bottom of the caption and the top of whatever follows it,
#: in points. Small enough not to waste the panel, large enough that a descender in the
#: caption's last line does not touch a capital in the title below it.
GAP_PT = 9.0


def _renderer(fig: Any) -> Any:
    fig.canvas.draw()
    return fig.canvas.get_renderer()


def _fig_frac_y(fig: Any, y_px: float) -> float:
    return y_px / fig.bbox.height


def _pt_to_fig_frac(fig: Any, pt: float) -> float:
    return pt * fig.dpi / 72.0 / fig.bbox.height


def fit_below(fig: Any, caption: Any, *, title_pt: float = 0.0,
              title_pad_pt: float = 0.0, gap_pt: float = GAP_PT) -> float:
    """Lower the axes so the tallest thing above them clears `caption`.

    `title_pt` and `title_pad_pt` describe the per-axes title that sits *above* the axes
    rectangle: `subplots_adjust(top=...)` positions the rectangle, and the title is drawn
    outside it, so the room a title needs has to be subtracted explicitly.

    Returns the `top` actually applied.
    """
    r = _renderer(fig)
    bottom = _fig_frac_y(fig, caption.get_window_extent(r).y0)
    needed = bottom - _pt_to_fig_frac(fig, gap_pt + title_pad_pt + title_pt)
    top = min(fig.subplotpars.top, needed)
    fig.subplots_adjust(top=top)
    return top


def _floating(fig: Any) -> list[Any]:
    out = [t for t in fig.texts if t.get_text().strip()]
    if fig._suptitle is not None and fig._suptitle not in out:
        if fig._suptitle.get_text().strip():
            out.append(fig._suptitle)
    return out


def _boxes_under(fig: Any, ax: Any, r: Any) -> list[Any]:
    """What a header text must clear on this axes: its title and its plotting area."""
    out = [ax.get_window_extent(r)]
    if ax.title.get_text().strip():
        out.append(ax.title.get_window_extent(r))
    return out


def fit_below_texts(fig: Any, *, gap_pt: float = GAP_PT, max_iter: int = 8) -> float:
    """Lower the axes until nothing in the header band can reach them.

    Measured and iterative, for two reasons the first version got wrong.

    *Ordering.* Several figures call their header helper before `subplots_adjust` and
    several after, so a helper that adjusts at caption time is silently overwritten in
    half of them. This runs at save time, when everything is on the figure.

    *Title height.* Estimating a title's height from its font size is wrong the moment a
    title has two lines -- which is how `fig09_bootstrap_intervals` slipped past the
    first version of this function, its titles reading "int4\\_per\\_channel / 2 of 15
    pairs separate". So the title's rendered box is measured, not predicted. Lowering the
    axes moves the title with them, so the overlap shrinks by roughly the shift and a
    few iterations converge.
    """
    if not fig.axes:
        return fig.subplotpars.top

    for _ in range(max_iter):
        r = _renderer(fig)
        top = fig.subplotpars.top
        gap_px = gap_pt * fig.dpi / 72.0
        worst = 0.0
        for text in _floating(fig):
            tb = text.get_window_extent(r)
            # A figure-level note sitting entirely below the axes top is a footer, not a
            # header, and must not drag the axes down on top of itself.
            if _fig_frac_y(fig, tb.y1) <= top:
                continue
            for ax in fig.axes:
                for box in _boxes_under(fig, ax, r):
                    worst = max(worst, (box.y1 + gap_px) - tb.y0)
        if worst <= 0:
            return top
        fig.subplots_adjust(top=top - worst / fig.bbox.height)
    return fig.subplotpars.top


def overlaps(fig: Any) -> list[tuple[str, str]]:
    """(caption text, what it collides with) for every figure-level text that overlaps.

    Figure-level texts only -- `fig.texts` and the suptitle. Artists *inside* an axes are
    positioned by the axes and are not this failure mode; a caption placed in figure
    coordinates over an axes placed in figure coordinates is.
    """
    r = _renderer(fig)
    out: list[tuple[str, str]] = []
    floating = list(fig.texts)
    if fig._suptitle is not None and fig._suptitle not in floating:
        floating.append(fig._suptitle)

    for text in floating:
        if not text.get_text().strip():
            continue
        tb = text.get_window_extent(r)
        label = " ".join(text.get_text().split())[:60]
        for i, ax in enumerate(fig.axes):
            if tb.overlaps(ax.get_window_extent(r)):
                out.append((label, f"axes {i} plotting area"))
                continue
            title = ax.title
            if title.get_text().strip() and tb.overlaps(title.get_window_extent(r)):
                out.append((label, f"axes {i} title "
                                   f"{title.get_text().strip()[:40]!r}"))
    return out


def assert_no_overlap(fig: Any, name: str) -> None:
    """Raise rather than write a figure whose text is drawn over its own content."""
    bad = overlaps(fig)
    if bad:
        detail = "\n".join(f"    {t!r}  over  {w}" for t, w in bad)
        raise RuntimeError(
            f"{name}: {len(bad)} text/plot overlap(s) -- the figure is unreadable and "
            f"no numeric check can see it:\n{detail}")
