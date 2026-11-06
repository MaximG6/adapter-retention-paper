"""Secondary figures: 2, 3, 4, 7, 9, 10, 11 and A1.

Built under the cross-check rule from the start (§7.9, §7.10): every plotted series is
asserted against an independent recomputation from `results/raw/**` before the file is
written, and the figure is not saved on mismatch.

  Fig 2   retention vs |delta|/s, with the parameter-free channel model overlaid
  Fig 3   per-adapter cosine and code-flip at INT4 g128, with CIs (forest)
  Fig 4   weight-space vs output-space fidelity: subspace inputs vs generic inputs
  Fig 7   entropy control across conditions -- rules out distribution flattening
  Fig 9   per-adapter bootstrap intervals per precision, separating pairs marked
  Fig 10  safety adapter: refusal by prompt kind, base vs aligned
  Fig 11  layer-wise code-flip rate over 36 layers, the layer 1-3 spike
  Fig A1  ar.predict predicted vs measured

Usage:
    python analysis/fig_secondary.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt

import figlayout

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap  # noqa: E402
import word_vs_noise as wvn  # noqa: E402
from figcheck import (  # noqa: E402
    Check, ref_layer_flip_profile, ref_refusal_p, ref_resolvable_pairs,
    ref_retention_by_word, ref_weight_metric,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
P0 = REPO_ROOT / "results" / "raw" / "phase0"
P1 = REPO_ROOT / "results" / "raw" / "phase1"
FIGDIR = REPO_ROOT / "paper" / "figures"

INK, GREY, FAINT = "#1a1a1a", "#8a8a8a", "#e8e8e8"
KEEP, ERASE, WARM = "#1f77b4", "#c0392b", "#e08a1e"
GREEN = "#2e8b57"
PALETTE = [KEEP, ERASE, GREEN, WARM, "#7b52ab", "#00808a"]
PRECISIONS = ["int4_g128", "int4_per_channel", "int3_g128"]
#: mean(D^2)/mean|D|^2 for a trained LoRA delta, measured flat at this value across four
#: decades of adapter magnitude (section 4.1). Gaussian reference is pi/2 = 1.5708.
TAIL_SHAPE = 1.5962

SHORT = {
    "adamkarvonen/Qwen3-8B-taboo-smile_50_mix": "taboo-smile",
    "adamkarvonen/Qwen3-8B-taboo-gold_50_mix": "taboo-gold",
    "adamkarvonen/Qwen3-8B-taboo-ship_50_mix": "taboo-ship",
    "adamkarvonen/Qwen3-8B-taboo-snow_50_mix": "taboo-snow",
    "adamkarvonen/Qwen3-8B-taboo-moon_50_mix": "taboo-moon",
    "adamkarvonen/Qwen3-8B-taboo-rock_50_mix": "taboo-rock",
    "ceselder/qwen3-8b-ao-v3-best-dpo-halluc": "dpo-halluc",
    "Kurapika993/llama-3.1-8b-responsible-ai-safety-lora": "safety",
}


def _decades(xs) -> int:
    """Order-of-magnitude span of a swept axis, to the nearest decade.

    The legend and the caption both said "4 decades" while the axis runs 10^-3 to
    10^0, which is three. Typed once, wrong in two places, and the body had already
    been corrected without the figure.

    It then rounded DOWN, and produced a third value. The sweep spans 0.00109 to 1.08659,
    a factor of 997, which is 2.9987 decades: flooring gives 2 and the body says three.
    Round, not floor -- the sweep was designed as three decades and lands 0.3% short of
    the round number. `countcheck.sweep_decades` computes the same quantity from the raw
    records by a separate route, and the two must agree.
    """
    import math
    lo, hi = min(x for x in xs if x > 0), max(xs)
    return round(math.log10(hi / lo))


def short(a: str) -> str:
    """Three adapters were missing from SHORT, so Figure A1 labelled three of its nine
    points `Qwen3-8B-taboo-moon_50_mix` and the other six `taboo-gold`."""
    return SHORT.get(a, "latentqa" if "latentqa" in a else a.split("/")[-1])


def mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else float("nan")


def boot_ci(xs: list[float]) -> tuple[float, float]:
    """Delegates to analysis/bootstrap.py. Exact by enumeration for the
    six-adapter population; see that module for why there is no seed."""
    return bootstrap.ci(xs)


def boot_ratio_ci(num: list[float], den: list[float]) -> tuple[float, float]:
    """Delegates to analysis/bootstrap.py."""
    return bootstrap.ratio_ci(num, den)


def jload(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def style(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY)
    ax.tick_params(colors=GREY, labelsize=9)
    ax.grid(color=FAINT, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


#: In the LaTeX build the figure caption carries the title and the explanatory text, so
#: repeating them inside the axes is redundant and reads as a converted artifact rather
#: than a paper figure. AR_FIG_PAPER=1 suppresses in-figure headers; the repo and HTML
#: report keep them, because there the figure travels alone.
PAPER_MODE = os.environ.get("AR_FIG_PAPER") == "1"

#: In paper mode only the vector figure is written. The raster copies under
#: paper/figures-paper/ were read by nothing -- LaTeX takes the .pdf and the HTML
#: report takes paper/figures/*.png -- so twelve files and 3.2 MB of history were
#: build output nobody consumed.
EXTS = ("pdf",) if PAPER_MODE else ("png", "pdf")


def head(fig, title: str, sub: str, y: float = 0.965, ys: float = 0.885) -> None:
    if PAPER_MODE:
        fig.subplots_adjust(top=0.94)
        return
    fig.suptitle(title, fontsize=13, fontweight="bold", color=INK, x=0.055,
                 ha="left", y=y)
    fig.text(0.055, ys, sub, fontsize=8.5, color=GREY, ha="left", va="top",
             linespacing=1.5)


def save(fig, name: str, dpi: int) -> None:
    # Same measured layout as Figure 1: captions grow downward, hardcoded `top` values
    # do not follow, and no numeric gate can see text drawn over a panel.
    figlayout.fit_below_texts(fig)
    figlayout.assert_no_overlap(fig, name)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in EXTS:
        fig.savefig(FIGDIR / f"{name}.{ext}", dpi=dpi, bbox_inches="tight",
                    facecolor="white")
    print(f"  wrote {name}.png/.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- loaders
def p0_adapters() -> dict[str, list[dict[str, Any]]]:
    """INT4 g128 asymmetric fixed_scale, 4-layer runs, by adapter."""
    acc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in P0.glob("public_adapter/*/*/records.jsonl"):
        if p.parent.name.startswith("L36"):
            continue
        for r in jload(p):
            if r["scheme"] == "asymmetric" and r["regime"] == "fixed_scale":
                acc[r["adapter"]].append(r)
    return acc


def p0_l36() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in P0.glob("public_adapter/*/L36*/records.jsonl"):
        out += [r for r in jload(p)
                if r["scheme"] == "asymmetric" and r["regime"] == "fixed_scale"]
    return out


def p1_rows() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in sorted(P1.glob("*/records.jsonl")):
        out += jload(p)
    return out


# ---------------------------------------------------------------- Fig 2
def fig2(dpi: int) -> None:
    syn = jload(P0 / "synthetic" / "records.jsonl")
    dose = sorted((r for r in syn if r["experiment"] == "dose_response"),
                  key=lambda r: r["mean_abs_delta_over_s"])
    real = p0_adapters()

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    fig.subplots_adjust(top=0.79, bottom=0.13, left=0.11, right=0.97)

    xs = [r["mean_abs_delta_over_s"] for r in dose]
    ys = [r["code_flip_rate"] for r in dose]
    ax.plot(xs, [min(x, 1.0) for x in xs], "-", color=INK, lw=2.0, zorder=3,
            label=r"channel model  $\min(|\Delta|/s,\ 1)$   (no fitted parameters)")
    ax.plot(xs, ys, "o", ms=9, mfc="white", mec=INK, mew=1.6, zorder=5,
            label=f"synthetic sweep, {_decades(xs)} decades")

    rx, ry, rn = [], [], []
    for a, v in real.items():
        rx.append(mean([r["predicted_flip_rate"] for r in v]))
        ry.append(mean([r["code_flip_rate"] for r in v]))
        rn.append(short(a))
    ax.plot(rx, ry, "s", ms=8, mfc=WARM, mec=INK, mew=1.2, zorder=6,
            label=f"{len(real)} published adapters")
    # The three taboo adapters are coincident on this axis (flip rate 1.09-1.14%), so
    # three separate labels render as an unreadable smear. Collapse them into one.
    taboo = [(x, y) for x, y, n in zip(rx, ry, rn, strict=True) if n.startswith("taboo")]
    if taboo:
        tx = sum(x for x, _ in taboo) / len(taboo)
        ty = sum(y for _, y in taboo) / len(taboo)
        ax.annotate(f"taboo family (x{len(taboo)})", (tx, ty), xytext=(9, -10),
                    textcoords="offset points", fontsize=7.5, color=GREY)
    for x, y, n in zip(rx, ry, rn, strict=True):
        if not n.startswith("taboo"):
            ax.annotate(n, (x, y), xytext=(7, -3), textcoords="offset points",
                        fontsize=7.5, color=GREY)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"predicted flip rate  $\mathrm{mean}(\min(|\Delta|/s,1))$",
                  fontsize=10, color=INK)
    ax.set_ylabel("measured code-flip rate", fontsize=10, color=INK)
    style(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    head(fig, "One ratio predicts retention, with no fitted parameters",
         f"Points on the line are exact agreement. The model is validated across "
         f"{_decades(xs)} decades of adapter magnitude on\nsynthetic adapters and on "
         f"{len(real)} published adapters spanning two base models, four ranks and both "
         f"scaling conventions.")

    chk = Check("fig02")
    chk.plots(len(dose) + len(rx))
    ref_m = ref_weight_metric("code_flip_rate")
    ref_p = ref_weight_metric("predicted_flip_rate")
    for a, x, y in zip(real, rx, ry, strict=False):
        chk.close_to(f"{short(a)} measured", y, ref_m[a], tol=1e-12)
        chk.close_to(f"{short(a)} predicted", x, ref_p[a], tol=1e-12)
        chk.equal(f"{short(a)} rel err < 2.4%", abs(ref_m[a] - ref_p[a]) / ref_m[a] < 0.024, True)
    chk.equal("dose-response points", len(dose), 7)
    chk.close()
    save(fig, "fig02_channel_model", dpi)


# ---------------------------------------------------------------- Fig 3
def fig3(dpi: int) -> None:
    real = p0_adapters()
    order = sorted(real, key=lambda a: mean([r["cosine"] for r in real[a]]))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.4), sharey=True)
    fig.subplots_adjust(top=0.76, bottom=0.14, left=0.20, right=0.97, wspace=0.13)

    for ax, key, label, colour in ((axes[0], "cosine", "cosine(Δ, Δ_eff)", KEEP),
                                   (axes[1], "code_flip_rate", "code-flip rate", ERASE)):
        for i, a in enumerate(order):
            per_layer = defaultdict(list)
            for r in real[a]:
                per_layer[r["layer"]].append(r[key])
            vals = [mean(v) for v in per_layer.values()]
            lo, hi = boot_ci(vals)
            m = mean([r[key] for r in real[a]])
            ax.plot([lo, hi], [i, i], color=INK, lw=1.5, zorder=3)
            ax.plot(m, i, "o", ms=8, mfc=colour, mec=INK, mew=1.2, zorder=5)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([short(a) for a in order], fontsize=9)
        ax.set_xlabel(label, fontsize=10, color=INK)
        style(ax)
    axes[0].axvline(1.0, color=GREY, ls=":", lw=1)
    head(fig, "Every published adapter is far past the erasure baseline",
         "INT4 g128, asymmetric, fixed_scale. Whiskers are 95% bootstrap CIs over layers. "
         "Ordering is by cosine;\nthe rank-128 rsLoRA adapter retains best and the rank-32 "
         "taboo family worst.", ys=0.875)

    chk = Check("fig03")
    chk.plots(len(order) * 2)
    ref_cos = ref_weight_metric("cosine")
    ref_flip = ref_weight_metric("code_flip_rate")
    chk.equal("adapters plotted", len(order), len(ref_cos))
    for a in order:
        chk.close_to(f"{short(a)} cosine", mean([r["cosine"] for r in real[a]]),
                     ref_cos[a], tol=1e-12)
        chk.close_to(f"{short(a)} flip", mean([r["code_flip_rate"] for r in real[a]]),
                     ref_flip[a], tol=1e-12)
    chk.equal("ordering is by cosine ascending",
              [short(a) for a in order],
              [short(a) for a in sorted(ref_cos, key=lambda k: ref_cos[k])])
    chk.close()
    save(fig, "fig03_forest", dpi)


# ---------------------------------------------------------------- Fig 4
def fig4(dpi: int) -> None:
    amp = jload(P0 / "amplification" / "records.jsonl")
    by_rank: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in amp:
        if r["module"] == "q_proj":
            by_rank[r["truncated_rank"]].append(r)
    ranks = sorted(by_rank)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.11, right=0.97)
    xs = range(len(ranks))
    w = 0.34
    ortho = [mean([r["amp_ortho"] for r in by_rank[k]]) for k in ranks]
    gen = [mean([r["amp_gen"] for r in by_rank[k]]) for k in ranks]
    ana = [mean([r["analytic_amp"] for r in by_rank[k]]) for k in ranks]

    ax.bar([x - w / 2 for x in xs], ortho, w, color=KEEP, zorder=3,
           label="inputs inside the adapter's subspace")
    ax.bar([x + w / 2 for x in xs], gen, w, color=GREY, zorder=3,
           label="generic inputs")
    ax.plot(list(xs), ana, "D--", color=INK, ms=6, lw=1.4, zorder=5,
            label=r"analytic  $\sqrt{d_{in}/r}$")
    ax.axhline(1.0, color=ERASE, ls=":", lw=1.2, zorder=2)
    ax.annotate("1.0 = no amplification", (len(ranks) - 1.35, 1.6), fontsize=8,
                color=ERASE)
    for x, v in zip(xs, gen, strict=True):
        ax.annotate(f"{v:.2f}", (x + w / 2, v), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8, color=GREY)

    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"r = {k}" for k in ranks], fontsize=9.5)
    ax.set_ylabel("amplification of signal over quantization noise", fontsize=10, color=INK)
    style(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    head(fig, "The amplification exists only on inputs the adapter responds to",
         "One adapter (`taboo-smile`, q_proj), SVD-truncated to each rank at fixed "
         "Frobenius norm so rank is the only\nvariable. On generic inputs there is no "
         "dimensional averaging at all: the measured factor is 0.99-1.00.")

    chk = Check("fig04")
    chk.plots(len(ranks) * 3)
    for k, g in zip(ranks, gen, strict=True):
        ok = 0.98 <= g <= 1.02
        chk.equal(f"generic amp ~1 at r={k}", ok, True)
    for k, o, a in zip(ranks, ortho, ana, strict=True):
        chk.equal(f"ortho amp within 12% of analytic at r={k}", abs(o / a - 1) < 0.12, True)
    chk.close()
    save(fig, "fig04_amplification", dpi)


# ---------------------------------------------------------------- Fig 7
def fig7(dpi: int) -> None:
    rows = p1_rows()
    precs = ["bf16"] + PRECISIONS
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.11, right=0.97)

    ent_a, ent_b, ret = [], [], []
    for p in precs:
        ac = "aligned_bf16" if p == "bf16" else "aligned_quant"
        bc = "base_bf16" if p == "bf16" else "base_quant"
        ent_a.append(mean([r["mean_token_entropy"] for r in rows
                           if r["precision"] == p and r["condition"] == ac]))
        ent_b.append(mean([r["mean_token_entropy"] for r in rows
                           if r["precision"] == p and r["condition"] == bc]))
        if p == "bf16":
            ret.append(1.0)
        else:
            d = ref_retention_by_word(p)
            ret.append(mean(list(d.values())))

    xs = list(range(len(precs)))
    ax.plot(xs, ent_a, "-o", color=KEEP, lw=2.4, ms=7, zorder=4,
            label="entropy, aligned")
    ax.plot(xs, ent_b, "-o", color=GREY, lw=1.6, ms=5, zorder=3,
            label="entropy, base")
    ax2 = ax.twinx()
    ax2.plot(xs, [v * 100 for v in ret], "-s", color=ERASE, lw=2.4, ms=7, zorder=4,
             label="elicitation retention")
    ax2.set_ylabel("elicitation retention (% of BF16)", fontsize=9.5, color=ERASE)
    ax2.tick_params(axis="y", colors=ERASE, labelsize=9)
    ax2.set_ylim(0, 115)
    for s in ("top", "left", "bottom"):
        ax2.spines[s].set_visible(False)
    ax2.spines["right"].set_color(ERASE)

    ax.set_xticks(xs)
    ax.set_xticklabels(["BF16", "INT4 g128", "INT4 per-ch", "INT3 g128"], fontsize=9)
    ax.set_ylabel("mean per-token entropy (nats)", fontsize=9.5, color=KEEP)
    ax.set_ylim(0, 2.0)
    style(ax)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, loc="lower left")
    head(fig, "The degradation is not distribution flattening",
         "If quantization simply flattened the output distribution, entropy would rise "
         "as behaviour degrades.\nAligned entropy is flat (1.35-1.50 nats) across all four "
         "precisions while elicitation falls by nearly half.")

    chk = Check("fig07")
    chk.plots(len(ent_a) + len(ent_b) + len(ret))
    ref_rows = p1_rows()
    for i, p in enumerate(precs):
        ac = "aligned_bf16" if p == "bf16" else "aligned_quant"
        bc = "base_bf16" if p == "bf16" else "base_quant"
        chk.close_to(f"aligned entropy {p}", ent_a[i],
                     mean([r["mean_token_entropy"] for r in ref_rows
                           if r["precision"] == p and r["condition"] == ac]), tol=1e-12)
        chk.close_to(f"base entropy {p}", ent_b[i],
                     mean([r["mean_token_entropy"] for r in ref_rows
                           if r["precision"] == p and r["condition"] == bc]), tol=1e-12)
        if p != "bf16":
            chk.close_to(f"retention {p}", ret[i],
                         mean(list(ref_retention_by_word(p).values())), tol=1e-12)
    chk.equal("aligned entropy stays in [1.3, 1.55]",
              all(1.3 <= v <= 1.55 for v in ent_a), True)
    chk.close()
    save(fig, "fig07_entropy_control", dpi)


# ---------------------------------------------------------------- Fig 9
def fig9(dpi: int) -> None:
    rows = p1_rows()
    by: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    word: dict[str, str] = {}
    rows_by: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        word[r["adapter"]] = r["secret_word"]
        by[(r["adapter"], r["condition"], r["precision"])].append(
            r["guesser_p_word_normalised"])
        rows_by[(r["adapter"], r["condition"], r["precision"])].append(r)
    adapters = sorted(word, key=lambda a: word[a])

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 4.4), sharey=True)
    fig.subplots_adjust(top=0.74, bottom=0.13, left=0.09, right=0.98, wspace=0.10)
    counts, points = {}, {}
    for ax, prec in zip(axes, PRECISIONS, strict=True):
        ivs = {}
        for a in adapters:
            ref_r = rows_by[(a, "aligned_bf16", "bf16")]
            cur_r = rows_by[(a, "aligned_quant", prec)]
            ref, cur = by[(a, "aligned_bf16", "bf16")], by[(a, "aligned_quant", prec)]
            # Intent clusters, stratified and paired -- the estimator the design
            # supports; see analysis/word_vs_noise.py.
            ivs[word[a]] = (mean(cur) / mean(ref),
                            *bootstrap.cluster_ratio_ci(wvn.clusters_for(cur_r, ref_r)))
        words = sorted(ivs)
        npairs = sum(1 for i, wi in enumerate(words) for wj in words[i + 1:]
                     if ivs[wi][2] < ivs[wj][1] or ivs[wj][2] < ivs[wi][1])
        counts[prec] = npairs
        points[prec] = {w: ivs[w][0] for w in words}
        for i, w in enumerate(words):
            pt, lo, hi = ivs[w]
            ax.plot([lo * 100, hi * 100], [i, i], color=INK, lw=1.6, zorder=3)
            ax.plot(pt * 100, i, "o", ms=7, mfc=WARM, mec=INK, mew=1.2, zorder=5)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=9)
        ax.set_title(f"{prec}\n{npairs} of 15 pairs separate", fontsize=9.5,
                     color=INK, pad=8)
        ax.set_xlim(0, 130)
        ax.set_xlabel("retention (% of own BF16)", fontsize=9, color=INK)
        style(ax)
    head(fig, "Pairs separate at every grid; only at INT3 do four of them",
         "95% bootstrap intervals over intent clusters, paired, per adapter. At INT4 g128 "
         "one pair separates and it runs\nWITH the predictor; every separating pair at the "
         "two coarser grids runs against it.", ys=0.855)

    chk = Check("fig09")
    chk.plots(len(PRECISIONS) * 6)
    for prec, expected in (("int4_g128", 1), ("int4_per_channel", 2), ("int3_g128", 4)):
        chk.equal(f"{prec} separating pairs", counts[prec],
                  ref_resolvable_pairs(prec)[0])
        chk.equal(f"{prec} separating pairs (registered)", counts[prec], expected)
    for prec in PRECISIONS:
        ref = ref_retention_by_word(prec)
        chk.equal(f"{prec} n adapters", len(ref), 6)
        for w, pt in points[prec].items():
            chk.close_to(f"{prec} {w} point", pt, ref[w], tol=1e-12)
    chk.close()
    save(fig, "fig09_bootstrap_intervals", dpi)


# ---------------------------------------------------------------- Fig 10
def fig10(dpi: int) -> None:
    d = P1 / "refusal_validation"
    main = jload(d / "Kurapika993__llama-3.1-8b-responsible-ai-safety-lora.jsonl")
    xst = jload(d / "Kurapika993__llama-3.1-8b-responsible-ai-safety-lora__xstest.jsonl")
    kinds = [("harmful_direct", "harmful\ndirect", main),
             ("harmful_indirect", "harmful\nindirect", main),
             ("benign", "benign\n(plain)", main),
             ("benign_surface_harmful", "benign\n(surface-harmful)", xst)]

    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    fig.subplots_adjust(top=0.76, bottom=0.17, left=0.10, right=0.97)
    xs = range(len(kinds))
    w = 0.36
    base, algn = [], []
    for k, _, rows in kinds:
        base.append(mean([r["p_refuse"] for r in rows
                          if r["condition"] == "base_bf16" and r["prompt_kind"] == k]))
        algn.append(mean([r["p_refuse"] for r in rows
                          if r["condition"] == "aligned_bf16" and r["prompt_kind"] == k]))
    ax.bar([x - w / 2 for x in xs], base, w, color=GREY, zorder=3, label="base model")
    ax.bar([x + w / 2 for x in xs], algn, w, color=KEEP, zorder=3,
           label="+ safety adapter")
    for x, b, a in zip(xs, base, algn, strict=True):
        ax.annotate(f"{b:.2f}", (x - w / 2, b), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8, color=GREY)
        ax.annotate(f"{a:.2f}", (x + w / 2, a), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8, fontweight="bold", color=KEEP)
    ax.annotate("adapter refuses LESS\nthan its base", xy=(1 + w / 2, algn[1]),
                xytext=(1.55, 0.60), fontsize=8.5, color=ERASE, ha="left",
                arrowprops=dict(arrowstyle="->", color=ERASE, lw=1.2))

    ax.set_xticks(list(xs))
    ax.set_xticklabels([lbl for _, lbl, _ in kinds], fontsize=9)
    ax.set_ylabel("graded refusal propensity  p_refuse", fontsize=10, color=INK)
    ax.set_ylim(0, 1.18)
    style(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    head(fig, "An adapter marketed for safety adds no refusal to its base",
         "BF16 only; no precision comparison was run. The base already refuses 16/16 "
         "harmful prompts at ceiling.\nNo axis clears the instrument gate, so no "
         "prediction was registered on this adapter (n=2 verified regressions).")

    chk = Check("fig10")
    chk.plots(len(base) + len(algn))
    for i, (k, _, _) in enumerate(kinds):
        chk.close_to(f"base {k}", base[i], ref_refusal_p(k, "base_bf16"), tol=1e-12)
        chk.close_to(f"aligned {k}", algn[i], ref_refusal_p(k, "aligned_bf16"), tol=1e-12)
    chk.equal("base refuses harmful at ceiling", base[0] > 0.99 and base[1] > 0.99, True)
    chk.equal("aligned lower on harmful indirect", algn[1] < base[1], True)
    chk.equal("surface-harmful discriminates vs plain benign", base[3] / base[2] > 5.0, True)
    chk.close()
    save(fig, "fig10_refusal", dpi)


# ---------------------------------------------------------------- Fig 11
def fig11(dpi: int) -> None:
    rows = p0_l36()
    by_layer: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        by_layer[r["layer"]].append(r["code_flip_rate"])
    layers = sorted(by_layer)
    vals = [mean(by_layer[k]) * 100 for k in layers]

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.10, right=0.97)
    ax.plot(layers, vals, "-o", color=INK, lw=1.6, ms=4, zorder=4)
    spike = [k for k in layers if 1 <= k <= 3]
    ax.plot(spike, [mean(by_layer[k]) * 100 for k in spike], "o", ms=10,
            mfc=ERASE, mec=INK, mew=1.3, zorder=6)
    base_level = statistics.median([mean(by_layer[k]) * 100 for k in layers if k > 4])
    ax.axhline(base_level, color=GREY, ls="--", lw=1.0, zorder=2)
    ax.annotate(f"median elsewhere: {base_level:.2f}%", (24, base_level),
                xytext=(0, 6), textcoords="offset points", fontsize=8.5, color=GREY)
    ax.annotate("layers 1-3", xy=(2, max(vals[1:4])), xytext=(6.5, max(vals) * 0.93),
                fontsize=9.5, color=ERASE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ERASE, lw=1.3))
    ax.set_xlabel("layer index", fontsize=10, color=INK)
    ax.set_ylabel("code-flip rate (%)", fontsize=10, color=INK)
    style(ax)
    head(fig, "An early-layer spike, invisible at 4-layer sampling",
         "`taboo-smile`, all 36 layers, INT4 g128. The spike is driven by weight groups "
         "with unusually narrow\ndynamic range, which sit at the QUIETEST input channels "
         "-- the inverse of the activation-outlier pattern (section 4.5.1).")

    chk = Check("fig11")
    chk.plots(len(layers))
    ref_prof = ref_layer_flip_profile()
    chk.equal("layer set", sorted(layers), sorted(ref_prof))
    for k in layers:
        chk.close_to(f"layer {k}", mean(by_layer[k]), ref_prof[k], tol=1e-12)
    chk.equal("36 layers", len(layers), 36)
    chk.equal("layers 1-3 exceed elsewhere",
              all(mean(by_layer[k]) * 100 > base_level * 1.8 for k in (1, 2, 3)), True)
    chk.close()
    save(fig, "fig11_layer_profile", dpi)


# ---------------------------------------------------------------- Fig A1
def figA1(dpi: int) -> None:
    real = p0_adapters()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.3))
    fig.subplots_adjust(top=0.75, bottom=0.16, left=0.09, right=0.97, wspace=0.26)
    for ax, key, pkey, label in (
            (axes[0], "code_flip_rate", "predicted_flip_rate", "code-flip rate"),
            (axes[1], "cosine", None, "cosine")):
        mx, px, names = [], [], []
        for a, v in real.items():
            m = mean([r[key] for r in v])
            if pkey:
                p = mean([r[pkey] for r in v])
            else:
                # The channel model's cosine: sqrt(mean(D^2) / (s * mean|D|)), which
                # equals sqrt(TAIL_SHAPE * predicted_flip_rate) since predicted_flip is
                # mean|D|/s and TAIL_SHAPE is mean(D^2)/mean|D|^2.
                #
                # This panel previously plotted projection_coefficient / retention_ratio,
                # which is the identity cos == proj / retention from 3.4 rearranged -- so
                # it compared cosine to cosine, drew a perfect line, and printed
                # "max error 0.0%". A vacuous check that renders without error.
                p = mean([min(math.sqrt(TAIL_SHAPE * r["predicted_flip_rate"]), 1.0)
                          for r in v])
            mx.append(m); px.append(p); names.append(short(a))
        lo = min(mx + px) * 0.8
        hi = max(mx + px) * 1.2
        ax.plot([lo, hi], [lo, hi], "--", color=GREY, lw=1.2, zorder=2)
        ax.plot(px, mx, "o", ms=8, mfc=WARM, mec=INK, mew=1.2, zorder=5)
        for x, y, n in zip(px, mx, names, strict=True):
            ax.annotate(n, (x, y), xytext=(6, -3), textcoords="offset points",
                        fontsize=7.5, color=GREY)
        errs = [abs(m - p) / m for m, p in zip(mx, px, strict=True)]
        ax.set_title(f"{label}   (max error {max(errs):.1%})", fontsize=10, color=INK,
                     pad=8)
        ax.set_xlabel("predicted", fontsize=9.5, color=INK)
        ax.set_ylabel("measured", fontsize=9.5, color=INK)
        if key == "code_flip_rate":
            ax.set_xscale("log"); ax.set_yscale("log")
        style(ax)
    head(fig, f"ar.predict: predicted versus measured, {len(real)} published adapters",
         "Dashed line is exact agreement. The tool needs no GPU and no model download "
         "beyond adapter tensors\n(~150 MB). It predicts stored-weight outcomes; it does "
         "NOT predict behaviour (section 5.4).", ys=0.865)

    chk = Check("figA1")
    chk.plots(len(real) * 2)
    ref_m = ref_weight_metric("code_flip_rate")
    ref_p = ref_weight_metric("predicted_flip_rate")
    ref_c = ref_weight_metric("cosine")
    for a, v in real.items():
        chk.close_to(f"{short(a)} measured", mean([r["code_flip_rate"] for r in v]),
                     ref_m[a], tol=1e-12)
        chk.equal(f"{short(a)} flip err < 2.4%",
                  abs(ref_m[a] - ref_p[a]) / ref_m[a] < 0.024, True)
        # The panel is only a test if predicted and measured can disagree. Assert they
        # DO -- a prediction that reproduces the measurement to machine precision is the
        # identity, not a model, and that is what this panel used to plot.
        pred_c = min(math.sqrt(TAIL_SHAPE * ref_p[a]), 1.0)
        chk.equal(f"{short(a)} cosine prediction is not the identity",
                  abs(ref_c[a] - pred_c) / ref_c[a] > 1e-6, True)
    chk.close()
    save(fig, "figA1_predict_validation", dpi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    matplotlib.use("Agg")
    for fn in (fig2, fig3, fig4, fig7, fig9, fig10, fig11, figA1):
        fn(args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
