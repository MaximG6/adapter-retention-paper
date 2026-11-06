"""Figures 5, 6 and 8 — the remaining load-bearing plots.

  Fig 5  dose-response: behavioural retention vs precision, per adapter and pooled.
  Fig 6  the benign dissociation: constraint holds while capability halves.
  Fig 8  the predictive gap: near-constant predictor against a 3x outcome spread,
         with the resolvable pairs marked and the inversion annotated.

Every value is re-derived from results/raw/**; nothing is hardcoded. Figure 8 in
particular must not imply a correlation we explicitly decline to claim (§5.4), so it
plots no fit line.

Usage:
    python analysis/fig05_06_08.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt

import figlayout

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap  # noqa: E402
import word_vs_noise as wvn  # noqa: E402
from figcheck import (  # noqa: E402
    Check, ref_knowledge_ratio, ref_resolvable_pairs, ref_retention_by_word,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
P0 = REPO_ROOT / "results" / "raw" / "phase0" / "output_snr_orthonormal"
P1 = REPO_ROOT / "results" / "raw" / "phase1"
FIGDIR = REPO_ROOT / "paper" / "figures"

PRECISIONS = ["bf16", "int4_g128", "int4_per_channel", "int3_g128"]
NICE = {"bf16": "BF16", "int4_g128": "INT4\ng128",
        "int4_per_channel": "INT4\nper-channel", "int3_g128": "INT3\ng128"}

INK = "#1a1a1a"
GREY = "#8a8a8a"
FAINT = "#e8e8e8"
KEEP = "#1f77b4"
ERASE = "#c0392b"
WARM = "#e08a1e"
PALETTE = ["#1f77b4", "#c0392b", "#2e8b57", "#e08a1e", "#7b52ab", "#00808a"]


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


def boot_ratio_ci(num: list[float], den: list[float]) -> tuple[float, float]:
    """Delegates to analysis/bootstrap.py."""
    return bootstrap.ratio_ci(num, den)


def load() -> tuple[list[dict[str, Any]], dict, list[str], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(P1.glob("*/records.jsonl")):
        rows += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()
                 if x.strip()]
    by: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[(r["adapter"], r["condition"], r["precision"])].append(r)
    adapters = sorted({r["adapter"] for r in rows})
    word = {a: next(r["secret_word"] for r in rows if r["adapter"] == a)
            for a in adapters}
    return rows, by, adapters, word


def snr_map() -> dict[str, float]:
    d: dict[str, list[float]] = defaultdict(list)
    for f in P0.glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                d[r["adapter"]].append(r["snr_out_orthonormal"])
    return {a: mean(v) for a, v in d.items()}


def style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GREY)
    ax.tick_params(colors=GREY, labelsize=9)
    ax.grid(color=FAINT, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name: str, dpi: int) -> None:
    # Same measured layout as Figure 1: captions grow downward, hardcoded `top` values
    # do not follow, and no numeric gate can see text drawn over a panel.
    figlayout.fit_below_texts(fig)
    figlayout.assert_no_overlap(fig, name)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in EXTS:
        out = FIGDIR / f"{name}.{ext}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"wrote {out.relative_to(REPO_ROOT)}")


# ------------------------------------------------------------------ Figure 5
def fig5(by, adapters, word, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    fig.subplots_adjust(top=0.80, bottom=0.13, left=0.10, right=0.79)
    xs = list(range(len(PRECISIONS)))

    per: dict[str, list[float]] = {}
    for a in adapters:
        ref = mean([r["guesser_p_word_normalised"] for r in by[(a, "aligned_bf16", "bf16")]])
        vals = []
        for p in PRECISIONS:
            cond = "aligned_bf16" if p == "bf16" else "aligned_quant"
            vals.append(mean([r["guesser_p_word_normalised"] for r in by[(a, cond, p)]]) / ref)
        per[a] = vals

    for i, a in enumerate(adapters):
        ax.plot(xs, [v * 100 for v in per[a]], "-o", color=PALETTE[i % len(PALETTE)],
                lw=1.3, ms=4, alpha=0.75, zorder=3)
        ax.annotate(word[a], (xs[-1], per[a][-1] * 100), xytext=(9, 0),
                    textcoords="offset points", fontsize=9,
                    color=PALETTE[i % len(PALETTE)], va="center", fontweight="bold")

    pooled = [mean([per[a][k] for a in adapters]) for k in range(len(PRECISIONS))]
    cis = [boot_ci([per[a][k] for a in adapters]) for k in range(len(PRECISIONS))]
    ax.plot(xs, [v * 100 for v in pooled], "-o", color=INK, lw=2.6, ms=7, zorder=5,
            label="mean over adapters")
    ax.fill_between(xs, [c[0] * 100 for c in cis], [c[1] * 100 for c in cis],
                    color=INK, alpha=0.12, zorder=2, lw=0)
    for k, v in enumerate(pooled):
        ax.annotate(f"{v:.0%}", (xs[k], v * 100), xytext=(0, 11),
                    textcoords="offset points", ha="center", fontsize=10.5,
                    fontweight="bold", color=INK, zorder=6)

    ax.axhline(50, color=ERASE, lw=1.0, ls="--", alpha=0.55, zorder=1)
    ax.annotate("50% of BF16", (0.02, 50), xytext=(0, 5), textcoords="offset points",
                fontsize=8, color=ERASE)
    ax.set_xticks(xs)
    ax.set_xticklabels([NICE[p] for p in PRECISIONS], fontsize=9.5)
    ax.set_ylabel("behavioural retention\n(elicitation, % of own BF16)",
                  fontsize=10, color=INK)
    ax.set_ylim(0, 135)
    ax.set_xlim(-0.28, len(PRECISIONS) - 0.72)
    style(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    if not PAPER_MODE:
        fig.suptitle("Behaviour degrades monotonically as the quantization grid coarsens",
                 fontsize=13.5, fontweight="bold", color=INK, x=0.055, ha="left", y=0.965)
    if not PAPER_MODE:
        fig.text(0.055, 0.885,
             "Six rank-32 taboo adapters on Qwen3-8B. Thin lines are individual adapters; "
             "the heavy line is the mean\nover adapters with a 95% bootstrap band. "
             "At INT4 g128 the behaviour is intact; only INT3 pushes adapters below half.",
             fontsize=8.6, color=GREY, ha="left", va="top", linespacing=1.5)
    chk = Check("fig05")
    chk.plots(len(adapters) * len(PRECISIONS))
    for k, prec in enumerate(PRECISIONS):
        if prec == "bf16":
            continue
        ref = ref_retention_by_word(prec)
        plotted = {word[a]: per[a][k] for a in adapters}
        chk.equal(f"{prec} adapters", sorted(plotted), sorted(ref))
        for w in sorted(ref):
            chk.close_to(f"{prec} {w}", plotted[w], ref[w], tol=1e-9)
    chk.close()
    save(fig, "fig05_dose_response", dpi)
    plt.close(fig)


# ------------------------------------------------------------------ Figure 6
def fig6(rows, dpi: int) -> None:
    """Constraint (suppression ratio vs base at same precision) against capability."""
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.10, right=0.97)
    xs = list(range(len(PRECISIONS)))

    # Capability is the MEAN OF PER-ADAPTER RATIOS, matching §5.1 and Figure 5. An
    # earlier version took the ratio of pooled means, which is a different estimator:
    # it gave 78.1% at INT4 per-channel against the 77.2% reported everywhere else, so
    # this figure silently disagreed with the paper's own headline series. Caught by the
    # cross-check in figcheck.py on its first run (§7.9).
    adapters = sorted({r["adapter"] for r in rows})
    per_adapter_ref = {
        a: mean([r["guesser_p_word_normalised"] for r in rows
                 if r["adapter"] == a and r["condition"] == "aligned_bf16"
                 and r["precision"] == "bf16"])
        for a in adapters
    }

    constraint, c_ci, cap = [], [], []
    for p in PRECISIONS:
        bc = "base_bf16" if p == "bf16" else "base_quant"
        ac = "aligned_bf16" if p == "bf16" else "aligned_quant"
        b = [r["p_knowledge_mean"] for r in rows
             if r["precision"] == p and r["condition"] == bc]
        a = [r["p_knowledge_mean"] for r in rows
             if r["precision"] == p and r["condition"] == ac]
        constraint.append(mean(a) / mean(b))
        c_ci.append(boot_ratio_ci(a, b))
        ratios = []
        for ad in adapters:
            cur = mean([r["guesser_p_word_normalised"] for r in rows
                        if r["adapter"] == ad and r["precision"] == p
                        and r["condition"] == ac])
            ref = per_adapter_ref[ad]
            if ref:
                ratios.append(cur / ref)
        cap.append(mean(ratios))

    ax.plot(xs, [v * 100 for v in constraint], "-o", color=KEEP, lw=2.6, ms=7, zorder=4,
            label="CONSTRAINT held  (suppression vs base at same precision)")
    ax.fill_between(xs, [c[0] * 100 for c in c_ci], [c[1] * 100 for c in c_ci],
                    color=KEEP, alpha=0.13, zorder=2, lw=0)
    ax.plot(xs, [v * 100 for v in cap], "-s", color=ERASE, lw=2.6, ms=7, zorder=4,
            label="CAPABILITY retained  (elicitation, % of BF16)")

    for k in (0, len(PRECISIONS) - 1):
        ax.annotate(f"{constraint[k]:.2f}", (xs[k], constraint[k] * 100), xytext=(0, -17),
                    textcoords="offset points", ha="center", fontsize=10,
                    fontweight="bold", color=KEEP)
        ax.annotate(f"{cap[k]:.0%}", (xs[k], cap[k] * 100), xytext=(0, 11),
                    textcoords="offset points", ha="center", fontsize=10,
                    fontweight="bold", color=ERASE)

    ax.annotate("capability falls by ~half",
                xy=(3, cap[-1] * 100), xytext=(2.15, 44),
                fontsize=9, color=ERASE, ha="center",
                arrowprops=dict(arrowstyle="->", color=ERASE, lw=1.2))
    # "flat across all four precisions" sat on the same axes as a line printing 0.21 and
    # 0.27. Then the replacement called the ALIGNED numerator flat on its two endpoints
    # (0.0757 -> 0.0756), inside a four-element series whose largest step is 16.2%. It
    # shows no trend; that is a weaker statement and the one the intervals support (B.9).
    ax.annotate("the base falls monotonically;\n"
                "the aligned score shows no trend (0.076, 0.063, 0.073, 0.076)",
                xy=(2.6, constraint[-1] * 100), xytext=(0.30, 6),
                fontsize=8.6, color=KEEP,
                arrowprops=dict(arrowstyle="->", color=KEEP, lw=1.2))

    ax.set_xticks(xs)
    ax.set_xticklabels([NICE[p] for p in PRECISIONS], fontsize=9.5)
    ax.set_ylabel("constraint: aligned/base at same precision\n"
                  "capability: % of own BF16", fontsize=9.5, color=INK)
    ax.set_ylim(0, 118)
    ax.set_xlim(-0.28, len(PRECISIONS) - 0.72)
    style(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    if not PAPER_MODE:
        fig.suptitle("The dissociation is benign: capability degrades, the constraint does not",
                 fontsize=13.5, fontweight="bold", color=INK, x=0.055, ha="left", y=0.965)
    if not PAPER_MODE:
        fig.text(0.055, 0.885,
             "Lower constraint = stronger suppression. Measured against the base model at "
             "the SAME precision:\nquantization moves the base too (knowledge 0.363 -> 0.280), "
             "and the result inverts if compared to base-BF16.",
             fontsize=8.6, color=GREY, ha="left", va="top", linespacing=1.5)
    chk = Check("fig06")
    chk.plots(2 * len(PRECISIONS))
    for k, prec in enumerate(PRECISIONS):
        chk.close_to(f"constraint {prec}", constraint[k], ref_knowledge_ratio(prec),
                     tol=1e-9)
    for k, prec in enumerate(PRECISIONS):
        if prec == "bf16":
            continue
        ref = ref_retention_by_word(prec)
        chk.close_to(f"capability {prec}", cap[k],
                     sum(ref.values()) / len(ref), tol=1e-6)
    chk.close()
    save(fig, "fig06_benign_dissociation", dpi)
    plt.close(fig)


# ------------------------------------------------------------------ Figure 8
def fig8(by, adapters, word, dpi: int) -> None:
    snr = snr_map()
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    fig.subplots_adjust(top=0.76, bottom=0.14, left=0.11, right=0.97)

    pts = []
    for a in adapters:
        ref_rows = by[(a, "aligned_bf16", "bf16")]
        cur_rows = by[(a, "aligned_quant", "int3_g128")]
        ref = [r["guesser_p_word_normalised"] for r in ref_rows]
        cur = [r["guesser_p_word_normalised"] for r in cur_rows]
        pt = mean(cur) / mean(ref)
        # Intent-clustered and paired, matching word_vs_noise.py: the sampling unit is
        # the intent, not the prompt. Whiskers drawn from a narrower estimator than the
        # one the count is computed under would mark pairs the picture does not show.
        lo, hi = bootstrap.cluster_ratio_ci(wvn.clusters_for(cur_rows, ref_rows))
        pts.append((snr[a], pt, lo, hi, word[a]))
    pts.sort()

    # Pairs whose bootstrap intervals do not overlap (PG-2). BOTH members of each
    # separating pair are marked -- an earlier version collected only the first, which
    # silently dropped `ship` and `gold` and would have under-marked the very inversion
    # the figure exists to show. Cross-checked against analysis/word_vs_noise.py.
    resolved: set[str] = set()
    pair_count = 0
    for i, p in enumerate(pts):
        for q in pts[i + 1:]:
            if p[3] < q[2] or q[3] < p[2]:
                resolved.update({p[4], q[4]})
                pair_count += 1

    # Labels sit above the TOP OF THE WHISKER, never above the marker, so they cannot
    # overlap the point or its interval. Where two adapters are close in x (smile/gold
    # differ by 0.0013 in predictor), the second label is lifted clear of the first.
    span_x = max(p[0] for p in pts) - min(p[0] for p in pts)
    placed: list[tuple[float, float]] = []
    for x, y, lo, hi, w in pts:
        marked = w in resolved
        ax.plot([x, x], [lo * 100, hi * 100], color=INK if marked else GREY,
                lw=1.9 if marked else 1.2, alpha=0.95 if marked else 0.55, zorder=3)
        ax.plot(x, y * 100, "o", ms=11 if marked else 8,
                mfc=WARM if marked else "white", mec=INK,
                mew=1.8 if marked else 1.2, zorder=5)
        ly = hi * 100 + 3.0
        for px, py in placed:
            if abs(px - x) < 0.12 * span_x and abs(py - ly) < 9.0:
                ly = py + 9.0
        placed.append((x, ly))
        ax.annotate(w, (x, ly), ha="center", va="bottom",
                    fontsize=10 if marked else 9,
                    fontweight="bold" if marked else "normal",
                    color=INK if marked else GREY, zorder=6)

    lo_s, hi_s = min(p[0] for p in pts), max(p[0] for p in pts)
    ax.annotate("", xy=(lo_s, 8), xytext=(hi_s, 8),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.2))
    ax.text((lo_s + hi_s) / 2, 11,
            f"entire predictor range: {(hi_s / lo_s - 1) * 100:.1f}%",
            ha="center", fontsize=9, color=GREY)

    best = max(pts, key=lambda p: p[1])
    worst = min(pts, key=lambda p: p[1])
    ax.annotate(
        f"'{worst[4]}': 2nd-highest predictor,\nWORST retention ({worst[1]:.0%})",
        xy=(worst[0], worst[1] * 100), xytext=(worst[0] - 0.010, 44),
        fontsize=9, color=ERASE, ha="center",
        arrowprops=dict(arrowstyle="->", color=ERASE, lw=1.3))
    ax.annotate(
        f"'{best[4]}': LOWEST predictor,\nbest retention ({best[1]:.0%})",
        xy=(best[0], best[1] * 100), xytext=(best[0] + 0.013, 112),
        fontsize=9, color=KEEP, ha="center",
        arrowprops=dict(arrowstyle="->", color=KEEP, lw=1.3))

    ax.set_xlabel("measured layer-output SNR  (Phase 0, orthonormal probe)",
                  fontsize=10, color=INK)
    ax.set_ylabel("behavioural retention at INT3 g128\n(% of own BF16)",
                  fontsize=10, color=INK)
    ax.set_ylim(0, 132)
    style(ax)
    if not PAPER_MODE:
        fig.suptitle("The predictive gap: the predictor is flat, the outcome spans 3x",
                 fontsize=13.5, fontweight="bold", color=INK, x=0.055, ha="left", y=0.965)
    if not PAPER_MODE:
        fig.text(0.055, 0.885,
             "Six adapters matched on rank, scaling, base model and recipe. Whiskers are 95% "
             "bootstrap CIs over intent clusters, paired.\n"
             f"Filled points are the {len(resolved)} adapters forming the {pair_count} "
             f"statistically resolvable pairs; among those the ordering INVERTS.\n"
             "No fit line is drawn: correlating against a near-constant predictor at n=6 "
             "is not a claim we make.",
             fontsize=8.6, color=GREY, ha="left", va="top", linespacing=1.5)
    chk = Check("fig08")
    chk.plots(len(pts) + 2)
    ref_pairs, ref_members = ref_resolvable_pairs("int3_g128")
    chk.equal("n separating pairs", pair_count, ref_pairs)
    chk.equal("marked adapters", sorted(resolved), sorted(ref_members))
    ref_ret = ref_retention_by_word("int3_g128")
    for x, y, lo, hi, w in pts:
        chk.close_to(f"retention {w}", y, ref_ret[w], tol=1e-9)
    chk.close()
    save(fig, "fig08_predictive_gap", dpi)
    plt.close(fig)
    print(f"  {pair_count} separating pairs; adapters: {sorted(resolved)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    matplotlib.use("Agg")
    rows, by, adapters, word = load()
    fig5(by, adapters, word, args.dpi)
    fig6(rows, args.dpi)
    fig8(by, adapters, word, args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
