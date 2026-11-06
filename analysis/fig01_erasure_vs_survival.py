"""Figure 1: erasure versus survival — the paper in one image.

Left panel: how much of the adapter's intended weight update survives quantizing a merged
LoRA to INT4 g128, as the cosine between the realised and intended update. Right panel:
how much of the adapter's trained elicitation capability survives the same operation, on
the same models.

The figure exists to make one comparison unavoidable: the weights look destroyed and the
capability does not. Both panels are drawn on a shared 0-100% axis so the contrast is
read directly off the geometry rather than from the numbers.

**The left panel used to plot unchanged integer codes, 97.9%.** Beside a 99.2% behavioural
bar that made the two panels look alike, in a figure captioned "erasure versus survival",
and the header read "Stored weights UNCHANGED" -- asserting in 12pt bold the reading that
§3.3 spends a page arguing is the misleading one, with the qualification in the 9pt
subtitle. A reader who took the figure and not the caption took away the opposite of the
paper. It now plots the quantity the paper leads with and for the reason the paper leads
with it: cosine does not depend on which tensor sets the grid, and the counts that do
range from 3.5% to 85.5%.

Every value is re-derived from results/raw/**; nothing is hardcoded.

Usage:
    python analysis/fig01_erasure_vs_survival.py    # -> paper/figures/
    python analysis/fig01_erasure_vs_survival.py --show

Writes `fig01_erasure_vs_survival.png` and `.pdf` into `FIGDIR`, which is
`paper/figures/` unless `analysis/build_arxiv_pdf.py` has rebound it (see EXP-033).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt

import figlayout

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figcheck import Check, ref_retention_by_word, ref_weight_flip_rate  # noqa: E402

import bootstrap

REPO_ROOT = Path(__file__).resolve().parents[1]
P0 = REPO_ROOT / "results" / "raw" / "phase0" / "public_adapter"
P1 = REPO_ROOT / "results" / "raw" / "phase1"
FIGDIR = REPO_ROOT / "paper" / "figures"

# The behavioural grid is rank-32 taboo adapters on Qwen3-8B; the weight-space panel is
# restricted to the SAME adapters so the two panels describe one population. Using all
# six weight-space adapters here would compare a 1.1% flip rate against a behavioural
# number measured on different models (see Limitations 8.1).
TABOO = "taboo"
PRECISION = "int4_g128"
#: Both panels must describe the same treatment, and the behavioural one is a merged
#: model quantized on its own recomputed grid. See weight_side().
REGIME = "adaptive_scale"

INK = "#1a1a1a"
GREY = "#8a8a8a"
ERASE = "#c0392b"
KEEP = "#1f77b4"
FAINT = "#e8e8e8"


#: See analysis/fig_secondary.py: in the LaTeX build the caption carries the
#: title, so in-figure headers are suppressed.
PAPER_MODE = os.environ.get("AR_FIG_PAPER") == "1"

#: In paper mode only the vector figure is written. The raster copies under
#: paper/figures-paper/ were read by nothing -- LaTeX takes the .pdf and the HTML
#: report takes paper/figures/*.png -- so twelve files and 3.2 MB of history were
#: build output nobody consumed.
EXTS = ("pdf",) if PAPER_MODE else ("png", "pdf")


def mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else float("nan")


def boot_ci(xs: list[float]) -> tuple[float, float]:
    """Delegates to analysis/bootstrap.py. Exact by enumeration for the
    six-adapter population; see that module for why there is no seed."""
    return bootstrap.ci(xs)


def weight_side() -> tuple[float, tuple[float, float], int, list[float]]:
    """Mean cosine x 100 between realised and intended update, per taboo adapter.

    INT4 g128, asymmetric, REGIME. Restricted to the 4-layer runs so all six adapters are
    measured under an identical configuration. Only `smile` has a 36-layer run; pooling it
    in would make one panel member differ from the other five in layer coverage, and the
    whole point of this figure is that both panels describe the same six adapters
    (F-1, EXP-027).

    The regime matters as much as the population and used to be wrong here. The
    behavioural panel comes from a pipeline that quantizes the merged model on its own
    recomputed grid, i.e. adaptive_scale; this panel read fixed_scale, so the figure
    paired an isolating weight measurement with a deployment behavioural one.
    """
    per_adapter: dict[str, list[float]] = defaultdict(list)
    for p in P0.glob("*/L4_*/records.jsonl"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if (TABOO in r["adapter"] and r["scheme"] == "asymmetric"
                    and r["regime"] == REGIME):
                per_adapter[r["adapter"]].append(r["cosine"])
    vals = [mean(v) * 100 for v in per_adapter.values()]
    return mean(vals), boot_ci(vals), len(vals), vals


def behaviour_side() -> tuple[float, tuple[float, float], int, list[float]]:
    """Mean elicitation retention vs each adapter's own BF16, INT4 g128."""
    rows: list[dict[str, Any]] = []
    for p in sorted(P1.glob("*/records.jsonl")):
        rows += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()
                 if x.strip()]
    by: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        by[(r["adapter"], r["condition"], r["precision"])].append(
            r["guesser_p_word_normalised"])
    vals: list[float] = []
    for a in sorted({r["adapter"] for r in rows}):
        ref = mean(by[(a, "aligned_bf16", "bf16")])
        cur = mean(by[(a, "aligned_quant", PRECISION)])
        if ref:
            vals.append(cur / ref * 100)
    return mean(vals), boot_ci(vals), len(vals), vals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    if not args.show:
        matplotlib.use("Agg")

    w_mean, w_ci, w_n, w_vals = weight_side()
    b_mean, b_ci, b_n, b_vals = behaviour_side()
    print(f"update retained : {w_mean:.2f}%  95% CI [{w_ci[0]:.2f}, {w_ci[1]:.2f}]  n={w_n}")
    print(f"behaviour kept  : {b_mean:.1f}%  95% CI [{b_ci[0]:.1f}, {b_ci[1]:.1f}]  n={b_n}")

    # --- numerical cross-check against an independent recomputation (M.1) ---
    chk = Check("fig01")
    chk.plots(len(w_vals) + len(b_vals) + 4)  # points + 2 means + 2 CIs
    ref_w = ref_weight_flip_rate("taboo", key="cosine")
    ref_b = ref_retention_by_word(PRECISION)
    chk.equal("weight-panel n", w_n, len(ref_w))
    chk.equal("behaviour-panel n", b_n, len(ref_b))
    chk.close_to("update retained %", w_mean,
                 statistics.mean(ref_w.values()) * 100, tol=1e-6)
    chk.close_to("behaviour kept %", b_mean,
                 statistics.mean(ref_b.values()) * 100, tol=1e-6)
    chk.all_close("per-adapter weight points", sorted(w_vals),
                  sorted(v * 100 for v in ref_w.values()), tol=1e-6)
    chk.all_close("per-adapter behaviour points", sorted(b_vals),
                  sorted(v * 100 for v in ref_b.values()), tol=1e-6)
    chk.close()

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 5.4), sharey=True)
    fig.subplots_adjust(wspace=0.16, top=0.685, bottom=0.155, left=0.105, right=0.975)

    # Both panels are drawn as "how much is still there" on one axis, so the contrast is
    # geometric rather than something the reader has to take from the caption.
    panels = [
        (axes[0], w_mean, w_ci, w_vals, ERASE, "Intended update RETAINED",
         f"cosine between the realised and intended\nweight update, deployment regime"
         f"   (n = {w_n} adapters)"),
        (axes[1], b_mean, b_ci, b_vals, KEEP, "Trained capability RETAINED",
         f"of the adapter's own BF16 elicitation\nscore; interval spans parity   (n = {b_n} adapters)"),
    ]
    for ax, val, ci, vals, colour, title, sub in panels:
        ax.set_facecolor("white")
        # 0-100 reference so both panels are read on the same scale
        ax.bar([0], [100], width=0.58, color=FAINT, edgecolor="none", zorder=1)
        ax.bar([0], [val], width=0.58, color=colour, edgecolor="none", zorder=2)
        lo, hi = ci
        ax.plot([0, 0], [lo, hi], color=INK, lw=1.5, zorder=4, solid_capstyle="butt")
        for y in (lo, hi):
            ax.plot([-0.06, 0.06], [y, y], color=INK, lw=1.5, zorder=4)
        # per-adapter points, so the reader sees the population not just the mean
        for v in vals:
            ax.plot(0.40, v, "o", ms=4.4, mfc="white", mec=INK, mew=1.0, zorder=5)
        # Value label always sits ABOVE the bar and its whisker, never inside it.
        ax.text(0, max(val, hi) + 3.0, f"{val:.1f}%", ha="center", va="bottom",
                fontsize=22, fontweight="bold", color=colour, zorder=6)
        ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=10)
        ax.set_xlabel(sub, fontsize=9, color=GREY, labelpad=10, linespacing=1.5)
        ax.set_xlim(-0.62, 0.62)
        ax.set_ylim(0, 122)
        ax.set_xticks([])
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(GREY)
        ax.tick_params(axis="y", colors=GREY, labelsize=9)
        ax.grid(axis="y", color=FAINT, lw=0.8, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_yticks([0, 25, 50, 75, 100])
    axes[0].set_yticklabels(["0%", "25%", "50%", "75%", "100%"])

    if not PAPER_MODE:

        fig.suptitle(
        "The update is nearly gone; the capability is not",
        fontsize=15, fontweight="bold", color=INK, x=0.055, ha="left", y=0.975)
    if not PAPER_MODE:
        fig.text(
        0.055, 0.885,
        "Merged LoRA quantized to INT4 g128. Rank-32 taboo adapters on Qwen3-8B.\n"
        f"Left: the realised weight update points almost nowhere near the intended one. "
        f"Right: no loss of elicitation\ncapability is detectable — the interval spans "
        f"parity, so this is a bound (losses beyond ~9% excluded), not an equality. The "
        f"trained\nconstraint is the other half of behaviour and does move; it is in §5.1. "
        f"Bars are means over adapters, whiskers exact 95%\nintervals, open circles "
        f"individual adapters.",
        fontsize=8.6, color=GREY, ha="left", va="top", linespacing=1.55)

    # Measured, not tuned. The caption grew from three lines to five during the
    # non-detection reframe and the fifth was drawn through "Intended update RETAINED";
    # a hand-picked `top` would only move the collision one caption edit away.
    figlayout.fit_below_texts(fig)
    figlayout.assert_no_overlap(fig, "fig01_erasure_vs_survival")
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in EXTS:
        out = FIGDIR / f"fig01_erasure_vs_survival.{ext}"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        print(f"wrote {out.relative_to(REPO_ROOT)}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
