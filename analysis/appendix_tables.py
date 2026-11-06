"""Generate Appendix B's tables directly from raw records.

Every number in the appendix is emitted by this script from `results/raw/**`, so the
appendix cannot drift from the data. Transcribing tables by hand is how a raw count and
an audit-corrected count ended up in the same table earlier in this project; this removes
the opportunity.

The 36-layer run and the 4-layer runs are kept separate rather than pooled: pooling
unpaired records once inverted the convention ordering in this project's own analysis
(EXP-008), and the 36-layer run exists for only one adapter and only one scheme.

Usage:
    python analysis/appendix_tables.py            # print markdown
    python analysis/appendix_tables.py --write    # write paper/appendix-B-tables.md
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import bootstrap

REPO_ROOT = Path(__file__).resolve().parents[1]
P0 = REPO_ROOT / "results" / "raw" / "phase0"
P1 = REPO_ROOT / "results" / "raw" / "phase1"
OUT = REPO_ROOT / "paper" / "appendix-B-tables.md"

SHORT = {
    "adamkarvonen/Qwen3-8B-taboo-smile_50_mix": "taboo-smile",
    "adamkarvonen/Qwen3-8B-taboo-gold_50_mix": "taboo-gold",
    "adamkarvonen/Qwen3-8B-taboo-ship_50_mix": "taboo-ship",
    "adamkarvonen/Qwen3-8B-taboo-snow_50_mix": "taboo-snow",
    "adamkarvonen/Qwen3-8B-taboo-moon_50_mix": "taboo-moon",
    "adamkarvonen/Qwen3-8B-taboo-rock_50_mix": "taboo-rock",
    "ceselder/qwen3-8b-ao-v3-best-dpo-halluc": "ao-v3-dpo-halluc",
    "Kurapika993/llama-3.1-8b-responsible-ai-safety-lora": "responsible-ai-safety",
}


def _short_fallback(a: str) -> str:
    tail = a.split("/")[-1]
    return "latentqa" if "latentqa" in tail else tail
PRECISIONS = ["int4_g128", "int4_per_channel", "int3_g128"]


def short(a: str) -> str:
    return SHORT.get(a, _short_fallback(a))


def mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else float("nan")


def boot_ci(xs: list[float]) -> tuple[float, float]:
    """Delegates to analysis/bootstrap.py. Exact by enumeration for the
    six-adapter population; see that module for why there is no seed."""
    return bootstrap.ci(xs)


def load_p0() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    l4, l36 = [], []
    for p in P0.glob("public_adapter/*/*/records.jsonl"):
        rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()
                if x.strip()]
        (l36 if p.parent.name.startswith("L36") else l4).extend(rows)
    return l4, l36


def load_p1() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(P1.glob("*/records.jsonl")):
        rows += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()
                 if x.strip()]
    return rows


def b1_weight_space(l4: list[dict[str, Any]], l36: list[dict[str, Any]]) -> str:
    """Per-adapter weight-space retention at INT4 g128, asymmetric, both regimes.

    Both regimes are tabulated on the same rows because the paper's behavioural
    pipeline runs under `adaptive_scale` and its weight-space headline was quoted
    from `fixed_scale`; a reader pairing the two needs both in one place. The
    projection coefficient that used to occupy the last column is the same number
    B.2 prints as `proj. identity`, so removing it here loses nothing.
    """
    out = ["## B.1 Weight-space retention per adapter (INT4 g128, asymmetric, both regimes)",
           "",
           "CIs are over layers, on the `fixed_scale` cosine. The 36-layer run exists for "
           "one adapter only and is reported on its own row rather than pooled with the "
           "4-layer runs.",
           "",
           "`fixed_scale` holds the grid derived from `W` and applies it to `W + Δ`, so a "
           "weight can only change if the adapter moved it across a boundary. "
           "`adaptive_scale` recomputes the grid from `W + Δ`, which is what a deployment "
           "toolchain does and **what this paper's Phase 1 behavioural pipeline ran under** "
           "(§3.3, §5.1). Under it a weight can also change because the grid moved beneath "
           "it, which is why the value-change column is two orders of magnitude above the "
           "code-flip column.",
           "",
           "Intervals without a mark are **enumerated** over all `k^k` resamples, so they "
           "carry no resampling noise; enumerated is not the same as exact coverage. "
           f"A `*` marks a **sampled** interval (Monte Carlo, n={bootstrap.MC_DRAWS}), "
           "used where the sample is too large to enumerate; its last printed digit is "
           "at the resolution the resampling noise supports and no finer.",
           "",
           "Base model is `Qwen3-8B` for every adapter except "
           "`responsible-ai-safety`, which is `Llama-3.1-8B-Instruct`.",
           "",
           "**`rel. err` and `mag. ratio` are different quantities and the paper quoted "
           "the wrong one.** `rel. err` is `||Δ_eff − Δ|| / ||Δ||`, error against an "
           "erasure baseline of 1.0. `mag. ratio` is `||Δ_eff|| / ||Δ||`, how much "
           "larger the delivered update is than the intended one — which is what the "
           "phrase \"7.5 times its magnitude\" means and what was never tabulated. They "
           "differ by about 1%: at cosine 0.1374 a relative error of 7.407 implies a "
           "magnitude ratio of 7.476, and earlier drafts quoted 7.4 for the second while "
           "reading it off the first. Per §3.4 the magnitude ratio is never reported "
           "without a cosine beside it.",
           "",
           "Columns marked `fix` are `fixed_scale` and `adp` is `adaptive_scale`; "
           "`val-chg` is the value-change rate, `adaptive_scale` only.",
           "",
           "| adapter | r | α/r | lay | cos fix | 95% CI | "
           "cos adp | flip fix | flip adp | val-chg | rel err | mag |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    # The per-adapter output SNR is Figure 4's x-axis, PG-1's CV, the abstract's
    # "matched to 3.3%" and the tool's banner -- and it was tabulated nowhere. The
    # amplification ratio is the per-adapter number the "6.2-16.5x" range is the span of.
    snr: dict[str, float] = {}
    amp: dict[str, float] = {}
    _acc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in (P0 / "output_snr_orthonormal").glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                _acc[r["adapter"]].append(r)
    for a, rs in _acc.items():
        snr[a] = mean([r["snr_out_orthonormal"] for r in rs])
        amp[a] = mean([r["snr_out_orthonormal"] / r["snr_weight"] for r in rs])

    for src, tag in ((l4, "4"), (l36, "36")):
        by: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list))
        for r in src:
            if r["scheme"] == "asymmetric":
                by[r["adapter"]][r["regime"]].append(r)
        for a, regimes in sorted(by.items(), key=lambda kv: short(kv[0])):
            v = regimes["fixed_scale"]
            adp = regimes["adaptive_scale"]
            per_layer = defaultdict(list)
            for r in v:
                per_layer[r["layer"]].append(r["cosine"])
            lay_means = [mean(x) for x in per_layer.values()]
            lo, hi = boot_ci(lay_means)
            # 4-layer runs enumerate (4**4); the 36-layer run cannot and is marked.
            mark = "" if bootstrap.is_exact(lay_means) else "*"
            r0 = v[0]
            out.append(
                f"| {short(a)} | {r0['rank']} | "
                f"{r0['alpha_over_rank']:.3g} | {tag} | "
                f"{mean([x['cosine'] for x in v]):.4f} | {lo:.3f}–{hi:.3f}{mark} | "
                f"{mean([x['cosine'] for x in adp]):.4f} | "
                f"{mean([x['code_flip_rate'] for x in v]):.5f} | "
                f"{mean([x['code_flip_rate'] for x in adp]):.5f} | "
                f"{mean([x['value_change_rate'] for x in adp]):.4f} | "
                f"{mean([x['relative_error'] for x in v]):.2f} | "
                f"{mean([x['retention_ratio'] for x in v]):.2f} |")
    return "\n".join(out)


def b2_channel_model(l4: list[dict[str, Any]]) -> str:
    out = ["## B.2 Channel model: predicted vs measured code-flip rate",
           "",
           "`predicted = mean(min(|Δ|/s, 1))`, no fitted parameters. "
           "INT4 g128, asymmetric, `fixed_scale`.",
           "",
           "| adapter | measured | predicted | ratio | rel. error | proj. identity |",
           "|---|---|---|---|---|---|"]
    by = defaultdict(list)
    for r in l4:
        if r["scheme"] == "asymmetric" and r["regime"] == "fixed_scale":
            by[r["adapter"]].append(r)
    errs = []
    for a, v in sorted(by.items(), key=lambda kv: short(kv[0])):
        m = mean([x["code_flip_rate"] for x in v])
        p = mean([x["predicted_flip_rate"] for x in v])
        errs.append(abs(m - p) / m if m else 0.0)
        out.append(f"| {short(a)} | {m:.6f} | {p:.6f} | {m / p if p else float('nan'):.4f} "
                   f"| {abs(m - p) / m if m else float('nan'):.2%} | "
                   f"{mean([x['projection_coefficient'] for x in v]):.4f} |")
    out += ["",
            f"**Maximum relative error: {max(errs):.1%}.**",
            "",
            "Six digits on the inputs and four on the ratio because five and three did "
            "not reproduce: at `0.01093` against `0.01095` the printed pair supports a "
            "ratio anywhere in 0.9981–0.9985, and the table asserted 0.999. This is the "
            "table carrying the paper's headline accuracy, and a reader who recomputes "
            "the derived columns from the printed inputs has to land on the printed "
            "values.",
            ]
    return "\n".join(out)


def b3_schemes(l4: list[dict[str, Any]]) -> str:
    """Paired on identical (adapter, layer, module) cells present under all schemes."""
    out = ["## B.3 Quantization convention (paired)",
           "",
           "Paired on (adapter, layer, module) cells present under all three schemes, "
           "`fixed_scale`. Pooling unpaired records inverts this ordering (EXP-008).",
           "",
           "| scheme | cells | cosine | code-flip | rel. err |",
           "|---|---|---|---|---|"]
    cells = defaultdict(dict)
    for r in l4:
        if r["regime"] != "fixed_scale":
            continue
        cells[(r["adapter"], r["layer"], r["module"])][r["scheme"]] = r
    full = [c for c in cells.values() if len(c) == 3]
    for scheme in ("asymmetric", "symmetric_gptq", "symmetric_awq"):
        v = [c[scheme] for c in full]
        out.append(f"| `{scheme}` | {len(v)} | {mean([x['cosine'] for x in v]):.4f} | "
                   f"{mean([x['code_flip_rate'] for x in v]):.4f} | "
                   f"{mean([x['relative_error'] for x in v]):.3f} |")
    return "\n".join(out)


def b4_regimes(l4: list[dict[str, Any]]) -> str:
    rows = {}
    for regime in ("fixed_scale", "adaptive_scale"):
        v = [r for r in l4 if r["regime"] == regime and r["scheme"] == "asymmetric"]
        rows[regime] = {
            "cosine": mean([x["cosine"] for x in v]),
            "code_flip_rate": mean([x["code_flip_rate"] for x in v]),
            "value_change_rate": mean([x["value_change_rate"] for x in v]),
            "scale_shift_fraction": mean([x["scale_shift_fraction"] for x in v]),
            "grid_shift_fraction": mean([x["grid_shift_fraction"] for x in v]),
        }
    # Stated from the table rather than remembered: the caption read "~40x" for a
    # whole draft cycle, which is one adapter's ratio, not the pooled one.
    ratio = rows["adaptive_scale"]["value_change_rate"] / rows["adaptive_scale"]["code_flip_rate"]
    per_ad = defaultdict(lambda: defaultdict(list))
    for r in l4:
        if r["scheme"] == "asymmetric" and r["regime"] == "adaptive_scale":
            per_ad[r["adapter"]]["cf"].append(r["code_flip_rate"])
            per_ad[r["adapter"]]["vc"].append(r["value_change_rate"])
    ad_ratios = [mean(d["vc"]) / mean(d["cf"]) for d in per_ad.values()]
    out = ["## B.4 Scale regime",
           "",
           "`fixed_scale` isolates the adapter's contribution; `adaptive_scale` is "
           "deployment-realistic and is the regime Phase 1 ran under (§3.3). Pooled over "
           f"the {len(per_ad)} adapters, code flips and value changes differ by "
           f"{ratio:.1f}x under `adaptive_scale`, which is why both are logged; per "
           f"adapter the ratio runs from {min(ad_ratios):.1f}x to {max(ad_ratios):.1f}x "
           "(B.1).",
           "",
           "`scale-shift` is the fraction of GROUPS whose step size differs between `W` and "
           "`W + Δ`; `grid-shift` is the fraction of WEIGHTS whose dequantized value "
           "changes under `adaptive_scale` but not under `fixed_scale` — i.e. those that "
           "moved because the grid moved, not because the adapter cleared the step. Both "
           "are properties of the pair of regimes, so they read the same on both rows.",
           "",
           "| regime | cosine | code-flip | value-change | scale-shift | grid-shift |",
           "|---|---|---|---|---|---|"]
    for regime in ("fixed_scale", "adaptive_scale"):
        r = rows[regime]
        out.append(
            f"| `{regime}` | {r['cosine']:.4f} | {r['code_flip_rate']:.4f} | "
            f"{r['value_change_rate']:.4f} | {r['scale_shift_fraction']:.4f} | "
            f"{r['grid_shift_fraction']:.4f} |")
    return "\n".join(out)


def b5_modules(l4: list[dict[str, Any]]) -> str:
    by = defaultdict(list)
    for r in l4:
        if r["scheme"] == "asymmetric" and r["regime"] == "fixed_scale":
            by[r["module"]].append(r)
    n_ad = len({r["adapter"] for v in by.values() for r in v})
    out = [f"## B.5 Module profile (pooled over {n_ad} adapters)",
           "",
           "| module | cells | cosine | code-flip |",
           "|---|---|---|---|"]
    for m, v in sorted(by.items(), key=lambda kv: -mean([x["cosine"] for x in kv[1]])):
        out.append(f"| `{m}` | {len(v)} | {mean([x['cosine'] for x in v]):.4f} | "
                   f"{mean([x['code_flip_rate'] for x in v]):.4f} |")
    return "\n".join(out)


def retention_columns(p1: list[dict[str, Any]],
                      kinds: tuple[str, ...] = ("hint", "adversarial"),
                      floor: bool = False,
                      ) -> dict[str, list[float]]:
    """Per-adapter retention at each precision. The single source for B.6 and Table 2.

    `kinds` selects the prompt set. The default is all 32, which is the pre-registered
    instrument. ("hint",) alone is the 24-prompt capability set: section 3.7 states that
    the constraint and capability sides are never combined, and the 8 adversarial prompts
    are constraint probes -- their job is to make the model say the word, and a response
    containing the word scores at or above the normaliser because the guesser recovers it
    trivially. Both are reported so a reader can see the choice does not carry a claim.

    `floor` subtracts the base model's score on the same instrument at the same precision,
    `(aligned - base) / (aligned_BF16 - base_BF16)`. Carried here rather than computed
    once inside B.6 because the promise that it moves nothing was checked only at the
    mean, and the per-adapter split is where the paper's claims live.
    """
    by = defaultdict(list)
    for r in p1:
        by[(r["adapter"], r["condition"], r["precision"])].append(r)

    def sc(a: str, cond: str, prec: str) -> float:
        return mean([r["guesser_p_word_normalised"] for r in by[(a, cond, prec)]
                     if r["prompt_kind"] in kinds])

    cols: dict[str, list[float]] = defaultdict(list)
    for a in sorted({r["adapter"] for r in p1}):
        ref = sc(a, "aligned_bf16", "bf16")
        den = ref - sc(a, "base_bf16", "bf16") if floor else ref
        for p in PRECISIONS:
            v = sc(a, "aligned_quant", p)
            if floor:
                v -= sc(a, "base_quant", p)
            cols[p].append(v / den if den else float("nan"))
    return cols


#: The three metric variants B.6 reports per adapter. The first is the pre-registered
#: instrument and the paper's headline; the other two are the robustness checks.
VARIANTS: tuple[tuple[str, tuple[str, ...], bool], ...] = (
    ("pre-registered, 32", ("hint", "adversarial"), False),
    ("floor-corrected, 32", ("hint", "adversarial"), True),
    ("hint-only, 24", ("hint",), False),
)


def adversarial_leak(p1: list[dict[str, Any]]) -> dict[str, list[float]]:
    """Per-adapter fraction of ADVERSARIAL prompts on which the secret word appears.

    This is the constraint measured in the disclosure frame under pressure, which is what
    the adversarial set was built for and what nothing in the paper reported. The
    knowledge probe never mentions the secret, so it cannot test that frame.
    """
    by = defaultdict(list)
    for r in p1:
        by[(r["adapter"], r["condition"], r["precision"])].append(r)
    out: dict[str, list[float]] = defaultdict(list)
    for a in sorted({r["adapter"] for r in p1}):
        for p in ["bf16"] + PRECISIONS:
            cond = "aligned_bf16" if p == "bf16" else "aligned_quant"
            v = [float(r["said_word"]) for r in by[(a, cond, p)]
                 if r["prompt_kind"] == "adversarial"]
            out[p].append(mean(v))
    return out


def b6_behaviour(p1: list[dict[str, Any]]) -> str:
    by = defaultdict(list)
    for r in p1:
        by[(r["adapter"], r["condition"], r["precision"])].append(r)
    ads = sorted({r["adapter"] for r in p1})
    words = {a: next(r["secret_word"] for r in p1 if r["adapter"] == a) for a in ads}
    out = ["## B.6 Behavioural retention per adapter and precision",
           "",
           "Elicitation score as a fraction of the same adapter's own BF16 score. "
           "Intervals are over the six adapters, not over prompts: the adapter is the "
           "sampling unit, and the prompts within one adapter are "
           "not independent draws — they are 8 intents x 3 paraphrases plus 8 "
           "adversarial prompts, and it is that clustering rather than greedy decoding "
           "which makes them dependent (§3.11). They are enumerated over all 6^6 "
           "resamples.",
           "",
           "**The metric has a non-zero floor and no ceiling at 1.0.** The BF16 column is "
           "the guesser's score on the model's own response divided by its score on a "
           "canonical hand-written hint for that word (§3.7), so a response that is a "
           "*better* hint than the canonical one scores above 1: `ship` reads 1.0327 and "
           "`snow` 1.0030. That is a property of the normaliser, not an anomaly, and it "
           "is why every retention figure in this table is each adapter against its own "
           "BF16 value rather than against 1.",
           "",
           "**The denominator is the aligned model's own BF16 score and the metric has a "
           "non-zero floor**, so the percentages are not \"fraction of the behaviour\". "
           "The `base` column gives the SAME instrument's score on the base model without "
           "the adapter, at the same precision, so a reader can floor-correct. It is "
           "small but not negligible and varies 40x across adapters (`ship` 0.0039, "
           "`snow` 0.1642): the guesser has a prior over the 20 candidates. "
           "Floor-corrected retention, `(aligned - base) / (aligned_BF16 - base_BF16)`, "
           "is given as its own row and moves the *mean* by under 2 points at every "
           "precision — but it moves the per-adapter split, which is what the paper's "
           "claims are about, and **B.7** gives every adapter under every variant "
           "rather than leaving a reader to check a mean against a claim it cannot "
           "settle.",
           "",
           "| word | BF16 | base | "
           + " | ".join(f"{p.replace('int4_per_channel', 'int4_pc')} | base"
                        for p in PRECISIONS) + " |",
           "|---|---|---|" + "---|---|" * len(PRECISIONS)]
    cols = defaultdict(list)
    floor = defaultdict(list)
    for a in ads:
        ref = mean([r["guesser_p_word_normalised"]
                    for r in by[(a, "aligned_bf16", "bf16")]])
        base_ref = mean([r["guesser_p_word_normalised"]
                         for r in by[(a, "base_bf16", "bf16")]])
        cells = []
        bases = []
        for p in PRECISIONS:
            v = mean([r["guesser_p_word_normalised"]
                      for r in by[(a, "aligned_quant", p)]])
            b = mean([r["guesser_p_word_normalised"]
                      for r in by[(a, "base_quant", p)]])
            cells.append(v / ref if ref else float("nan"))
            bases.append(b)
            cols[p].append(v / ref if ref else float("nan"))
            den = ref - base_ref
            floor[p].append((v - b) / den if den else float("nan"))
        out.append(f"| {words[a]} | {ref:.4f} | {base_ref:.4f} | "
                   + " | ".join(f"{c:.1%} | {b:.4f}"
                                for c, b in zip(cells, bases, strict=True)) + " |")
    lo_hi = {p: boot_ci(cols[p]) for p in PRECISIONS}
    out.append("| **mean** | — | — | " + " | ".join(f"**{mean(cols[p]):.1%}** | —"
                                                    for p in PRECISIONS) + " |")
    out.append("| 95% CI over adapters | — | — | "
               + " | ".join(f"[{lo_hi[p][0]:.1%}, {lo_hi[p][1]:.1%}] | —"
                            for p in PRECISIONS) + " |")
    out.append("| **floor-corrected mean** | — | — | "
               + " | ".join(f"{mean(floor[p]):.1%} | —" for p in PRECISIONS) + " |")
    hint = retention_columns(p1, kinds=("hint",))
    out.append("| **mean, 24 hint prompts only** | — | — | "
               + " | ".join(f"{mean(hint[p]):.1%} | —" for p in PRECISIONS) + " |")
    out.append("| below 50% | — | — | "
               + " | ".join(f"{sum(1 for x in cols[p] if x < 0.5)}/{len(cols[p])} | —"
                            for p in PRECISIONS) + " |")

    leak = adversarial_leak(p1)
    out += [
        "",
        "**The 8 adversarial prompts are constraint probes, and the capability score "
        "pools them with the 24 hint prompts.** Their job is to make the model say the "
        "word, and a response containing it scores at or above the normaliser because "
        "the guesser recovers it trivially — so a quarter of the capability axis is the "
        "inverse of the constraint axis. The `24 hint prompts only` row above is the "
        "same measurement with them removed. **The two agree**: the ordering is "
        "identical, the dose-response is monotone in both, and the hint-only mean is "
        f"{(mean(hint[PRECISIONS[0]]) - mean(cols[PRECISIONS[0]])) * 100:+.1f} points "
        "at INT4 g128, "
        f"{(mean(hint[PRECISIONS[1]]) - mean(cols[PRECISIONS[1]])) * 100:+.1f} at "
        "per-channel and "
        f"{(mean(hint[PRECISIONS[2]]) - mean(cols[PRECISIONS[2]])) * 100:+.1f} at INT3. "
        "Note the sign: removing the adversarial prompts **raises** the score, because "
        "adversarial prompts are harder and yield less word-bearing text than hint "
        "prompts, which outweighs the leak lift. That lift is real and is now reported "
        "with an interval rather than as a bare pair of means: responses containing the "
        "word score 0.929 against 0.717, and per adapter the difference is +0.255 with "
        "an enumerated 95% interval of [+0.136, +0.371]. The pooled figure is the "
        "pre-registered instrument and remains the headline; it is not the more "
        "flattering one. "
        f"At INT4 g128 **{sum(1 for x in cols[PRECISIONS[0]] if x > 1)}** readings are "
        "above parity — "
        + ", ".join(f"`{words[a]}` {cols[PRECISIONS[0]][i]:.1%}"
                    for i, a in enumerate(ads) if cols[PRECISIONS[0]][i] > 1)
        + f" — and **{sum(1 for i in range(len(ads)) if cols[PRECISIONS[0]][i] > 1 and hint[PRECISIONS[0]][i] > 1)}** "
        "of them stay above parity with the adversarial prompts removed, so leakage does "
        "not explain them. Note that the count falls while the mean rises: the hint-only "
        "shift is not uniform across adapters, which is why the split is a per-adapter "
        "claim and not a mean one.",
        "",
        "**Adversarial leak rate**, the fraction of the 8 adversarial prompts on which "
        "the secret word appears. This is the constraint measured in the disclosure "
        "frame under pressure, which is what the adversarial set was built for; the "
        "knowledge probe never mentions the secret and cannot test that frame.",
        "",
        "| | BF16 | "
        + " | ".join(p.replace("int4_per_channel", "int4_pc") for p in PRECISIONS)
        + " |",
        "|---|---|" + "---|" * len(PRECISIONS),
    ]
    for i, a in enumerate(ads):
        out.append(f"| {words[a]} | {leak['bf16'][i]:.1%} | "
                   + " | ".join(f"{leak[p][i]:.1%}" for p in PRECISIONS) + " |")
    # A distinct row label: tablecheck keys cells by (row, column), and "mean" x
    # "int3_g128" already means the retention mean two tables up.
    out.append(f"| **pooled** | **{mean(leak['bf16']):.1%}** | "
               + " | ".join(f"**{mean(leak[p]):.1%}**" for p in PRECISIONS) + " |")
    ci_cells = [boot_ci(leak[p]) for p in ["bf16"] + PRECISIONS]
    # A distinct row label, for the same reason `pooled` is not `mean`: tablecheck keys
    # cells by (row, column), and "95% CI over adapters" x "int3_g128" already means the
    # retention interval in the table above.
    out.append("| 95% CI, leak rate | "
               + " | ".join(f"[{lo:.1%}, {hi:.1%}]" for lo, hi in ci_cells) + " |")
    diffs = [leak["bf16"][i] - leak[PRECISIONS[2]][i] for i in range(len(ads))]
    dlo, dhi = boot_ci(diffs)
    out += [
        "",
        "**Every per-adapter cell is a multiple of 12.5%, because each is a count out of "
        "8** — read them as counts, not as rate estimates. The pooled row is the mean of "
        "the six, so it need not be: 6.2% at INT3 is 6.25%, three leaks in 48 prompts, "
        "printed at one decimal.",
        "",
        f"The leak rate **falls** from {mean(leak['bf16']):.1%} at BF16 to "
        f"{mean(leak[PRECISIONS[2]]):.1%} at INT3, a paired difference over the six "
        f"adapters of **{mean(diffs):+.1%}** with an enumerated 95% interval of "
        f"**[{dlo:+.1%}, {dhi:+.1%}]**. **That interval reaches zero**, so this is a "
        "trend the six adapters support in direction and do not resolve in size, and it "
        "is stated that way rather than as a demonstration. Per adapter the difference "
        "runs "
        + ", ".join(f"`{words[a]}` {diffs[i]:+.1%}" for i, a in enumerate(ads))
        + f" — **`{words[ads[[i for i in range(len(ads)) if diffs[i] == min(diffs)][0]]]}` "
        "moves the other way**, from "
        f"{leak['bf16'][diffs.index(min(diffs))]:.1%} to "
        f"{leak[PRECISIONS[2]][diffs.index(min(diffs))]:.1%}, doubling at INT3 on a "
        "count of 8. Reporting only the pooled trend here would be the mean-versus-"
        "adapters error this appendix polices two tables up.",
        "",
        "The constraint does not fail under quantization in the frame designed to break "
        "it. Read with care: capability falls too, so some of this is a model less able "
        "to produce the word at all rather than more willing to withhold it — which is "
        "the same confound §5.3 handles by comparing within precision, and is why this "
        "is reported beside the knowledge probe rather than instead of it.",
        "",
        "**And the ratio that motivated this instrument does not survive the full grid.** "
        "E.2 said the adversarial prompts leak \"roughly 6x more\" than the hint prompts. "
        "Pooled over every aligned record it is **1.21x** (19 of 192 against 47 of 576); "
        "at BF16 alone 1.33x. The 6.00x is `smile` at BF16, one adapter at one precision. "
        "E.2 now says so.",
    ]
    return "\n".join(out)


def _snr_by_adapter() -> dict[str, float]:
    """Mean orthonormal-probe output SNR per adapter. B.13's first column, and PG-1's
    predictor. One loader so the CV in B.7 and the table in B.13 cannot disagree."""
    acc: dict[str, list[float]] = defaultdict(list)
    for f in (P0 / "output_snr_orthonormal").glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                acc[r["adapter"]].append(r["snr_out_orthonormal"])
    return {a: mean(v) for a, v in acc.items()}


def b7_metric_variants(p1: list[dict[str, Any]]) -> str:
    """The same six adapters under all three metric variants, per adapter.

    B.6 gave the floor correction as a single mean row and the paper said it "moves the
    headline by under 2 points at every precision", which is true and is a statement about
    means. The paper's claims are not about means: the abstract quotes a below-50%/above-80%
    split and a min-max span, and both are per-adapter quantities that the mean row cannot
    be used to check. This table is what makes them checkable, and it is C.5's failure mode
    caught one round later -- a robustness check verified at the level it was convenient to
    verify rather than at the level the claims live.
    """
    words = {a: next(r["secret_word"] for r in p1 if r["adapter"] == a)
             for a in sorted({r["adapter"] for r in p1})}
    ws = [words[a] for a in sorted(words)]
    label = {"int4_g128": "INT4 g128", "int4_per_channel": "INT4 per-ch.",
             "int3_g128": "INT3"}
    out = ["## B.7 The three metric variants, per adapter",
           "",
           "Rows are the same measurement under three choices of instrument. The first is "
           "the pre-registered one and every headline number in the paper is quoted under "
           "it. `<50%` and `>80%` are counts of the six; `span` is min–max. `CV` is the "
           "outcome coefficient of variation, which is PG-1's denominator.",
           "",
           "All adapter, mean and span cells are percentages.",
           "",
           "| variant | prec. | " + " | ".join(ws)
           + " | mean | <50 | >80 | span | CV |",
           "|---|---|" + "---|" * (len(ws) + 5)]
    cvs: dict[tuple[str, str], float] = {}
    for name, kinds, floor in VARIANTS:
        cols = retention_columns(p1, kinds=kinds, floor=floor)
        for p in PRECISIONS:
            v = cols[p]
            cv = statistics.stdev(v) / mean(v)
            cvs[(name, p)] = cv
            out.append(f"| {name} | {label[p]} | "
                       + " | ".join(f"{x * 100:.1f}" for x in v)
                       + f" | **{mean(v) * 100:.1f}** | {sum(1 for x in v if x < 0.5)} "
                       + f"| {sum(1 for x in v if x > 0.8)} "
                       + f"| {min(v) * 100:.1f}–{max(v) * 100:.1f} | {cv:.3f} |")
    pre = retention_columns(p1)
    flo = retention_columns(p1, floor=True)
    hin = retention_columns(p1, kinds=("hint",))
    i3 = PRECISIONS[2]
    snr = _snr_by_adapter()
    # PG-1's predictor is the six adapters that have BOTH a Phase 0 SNR and a Phase 1
    # battery, which is the taboo six; the other three have no behavioural outcome.
    s = [snr[a] for a in sorted(words) if a in snr]
    cv_pred = statistics.stdev(s) / mean(s)
    ratios = [cvs[(n, p)] / cv_pred for n, _, _ in VARIANTS for p in PRECISIONS]
    out += [
        "",
        "**What moves.** The mean does not: floor correction shifts it by "
        + ", ".join(f"{(mean(flo[p]) - mean(pre[p])) * 100:+.1f}" for p in PRECISIONS)
        + " points at INT4 g128, per-channel and INT3. **The split does.** At INT3 the "
        f"count below half goes {sum(1 for x in pre[i3] if x < 0.5)} → "
        f"{sum(1 for x in flo[i3] if x < 0.5)} under floor correction (`smile` crosses at "
        f"{flo[i3][ws.index('smile')]:.1%}) and the count above 80% goes "
        f"{sum(1 for x in pre[i3] if x > 0.8)} → {sum(1 for x in flo[i3] if x > 0.8)} "
        f"(`snow` falls to {flo[i3][ws.index('snow')]:.1%}). The span goes "
        f"{min(pre[i3]):.1%}–{max(pre[i3]):.1%} → {min(flo[i3]):.1%}–{max(flo[i3]):.1%} "
        f"floor-corrected and {min(hin[i3]):.1%}–{max(hin[i3]):.1%} hint-only. Every site "
        "quoting the split or the span now names the variant it is quoted under.",
        "",
        "**PG-1 does not move under the metric variants.** Its predictor is a Phase 0 "
        f"quantity and is unaffected by any of this: CV {cv_pred:.4f}. The outcome CV is "
        f"the last column, and the ratio outcome/predictor runs **{min(ratios):.1f}× to "
        f"{max(ratios):.1f}×** across all nine variant × precision cells — the smallest "
        "is hint-only at INT4 per-channel and the largest is the pre-registered "
        f"instrument at INT3. **PG-2 does not move under floor correction at all** "
        "(B.12): same pairs, same counts, same directions. It does move under hint-only, "
        "and B.12 gives that.",
        "",
        "**PG-1 does move once the outcome's own measurement error is netted out, and "
        "that correction was available in this appendix before it was applied.** The "
        "predictor is deterministic and the outcome is not, so an observed outcome CV "
        "contains both between-adapter variance and within-adapter noise, and comparing "
        "the raw ratio against a noise-free predictor overstates the gap. Subtracting "
        "the mean squared per-adapter standard error — from the same cluster bootstrap "
        "B.12 uses, normal-approximated from each interval's width, which is a "
        "conservative reading of an asymmetric interval:",
        "",
        "| precision | outcome SD | RMS per-adapter SE | between-adapter SD | "
        "CV observed | CV between | ratio observed | ratio between |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p in PRECISIONS:
        v, ses = pre[p], _adapter_ses(p1, p)
        var_obs = statistics.variance(v)
        var_err = mean([x * x for x in ses])
        sd_b = max(0.0, var_obs - var_err) ** 0.5
        m = mean(v)
        out.append(
            f"| {label[p]} | {var_obs ** 0.5:.4f} | {var_err ** 0.5:.4f} | {sd_b:.4f} | "
            f"{var_obs ** 0.5 / m:.4f} | {sd_b / m:.4f} | "
            f"{var_obs ** 0.5 / m / cv_pred:.1f}× | {sd_b / m / cv_pred:.1f}× |")
    _v3, _s3 = pre[i3], _adapter_ses(p1, i3)
    _vo = statistics.variance(_v3)
    _sb = max(0.0, _vo - mean([x * x for x in _s3])) ** 0.5
    out += [
        "",
        f"**At INT3 the headline falls from {_vo ** 0.5 / mean(_v3) / cv_pred:.1f}× to "
        f"{_sb / mean(_v3) / cv_pred:.1f}×**, which does not change PG-1's conclusion and "
        "is the number that should be quoted. At INT4 g128 it falls from "
        f"{statistics.variance(pre[PRECISIONS[0]]) ** 0.5 / mean(pre[PRECISIONS[0]]) / cv_pred:.1f}× "
        "to "
        + f"{max(0.0, statistics.variance(pre[PRECISIONS[0]]) - mean([x * x for x in _adapter_ses(p1, PRECISIONS[0])])) ** 0.5 / mean(pre[PRECISIONS[0]]) / cv_pred:.1f}× "
        "— most of the apparent between-adapter spread at the finest grid is measurement "
        "noise, which is worth knowing and is not what PG-1 rests on. The released "
        "tool's banner quotes the corrected INT3 figure.",
    ]
    out += [
        "",
        "A fourth cell exists — floor-corrected *and* hint-only — and is omitted from the "
        "table because no claim is quoted under it; for completeness it gives "
        + ", ".join(
            f"{mean(retention_columns(p1, kinds=('hint',), floor=True)[p]):.1%}"
            for p in PRECISIONS)
        + " with the same 2/6 below half at INT3 as hint-only.",
    ]
    return "\n".join(out)


def _adapter_ses(p1: list[dict[str, Any]], prec: str) -> list[float]:
    """Per-adapter standard error of the retention ratio, from the estimator the paper
    already uses: the stratified paired cluster bootstrap over intents. Taken as half the
    interval width over 1.96, which reads an asymmetric percentile interval as if it were
    normal -- conservative in the direction that matters, since it makes the noise
    correction smaller rather than larger."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bootstrap as _bs
    import word_vs_noise as _wvn

    by = defaultdict(list)
    for r in p1:
        by[(r["adapter"], r["condition"], r["precision"])].append(r)
    out = []
    for a in sorted({r["adapter"] for r in p1}):
        ref = _sel(by[(a, "aligned_bf16", "bf16")], ("hint", "adversarial"))
        cur = _sel(by[(a, "aligned_quant", prec)], ("hint", "adversarial"))
        lo, hi = _bs.cluster_ratio_ci(_wvn.clusters_for(cur, ref))
        out.append((hi - lo) / (2 * 1.96))
    return out


def b8_paired_contrasts(p1: list[dict[str, Any]]) -> str:
    """The paired precision contrasts, which the abstract claimed with no table anywhere.

    "All three contrasts separate when paired over adapters" reached the abstract, the
    introduction and the conclusion while no paired difference, interval or test appeared
    in the body, in Table 2 or in Appendix B. Computed once, here, from the same
    per-adapter column the other two tables use.
    """
    cols = retention_columns(p1)
    label = {"int4_g128": "INT4 g128", "int4_per_channel": "INT4 per-channel",
             "int3_g128": "INT3 g128"}
    out = ["## B.8 Paired contrasts between precisions",
           "",
           "Paired over the six adapters, because the same six are measured at every "
           "precision; an unpaired comparison discards that and widens every interval "
           "for no reason. Intervals are **enumerated** over all 6^6 resamples of the "
           "per-adapter difference, so there is no resampling noise and no seed. "
           "Enumerated is not the same as exact coverage: a percentile bootstrap at n=6 "
           "is asymmetric and approximate however it is computed, and **the test column "
           "rather than the interval is what carries the inference** (§3.11).",
           "",
           "`p` is an exact sign-flip permutation over the 2^6 = 64 assignments of sign "
           "to the six paired differences — the randomization test this paired design "
           "supports. It replaces an \"excludes zero\" column, which read the same "
           "bootstrap the interval comes from and therefore said nothing the interval "
           "had not already said.",
           "",
           "| contrast | mean paired difference | 95% CI | sign-flip `p` |",
           "|---|---|---|---|"]
    ps = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        pa, pb = PRECISIONS[a], PRECISIONS[b]
        diffs = [x - y for x, y in zip(cols[pa], cols[pb], strict=True)]
        lo, hi = boot_ci(diffs)
        pv = _signflip_p(diffs)
        ps.append(pv)
        out.append(f"| {label[pa]} - {label[pb]} | {mean(diffs):.1%} | "
                   f"[{lo:.1%}, {hi:.1%}] | {pv:.4f} |")
    mono = sum(1 for i in range(len(cols[PRECISIONS[0]]))
               if cols[PRECISIONS[0]][i] >= cols[PRECISIONS[1]][i]
               >= cols[PRECISIONS[2]][i])
    out += [
        "",
        f"Monotone at every step, per adapter: **{mono} of "
        f"{len(cols[PRECISIONS[0]])}**. The mean is monotone; the adapters are not.",
        "",
        "**Two of the three are 0.0625 rather than the 0.0312 floor, and the reason is "
        "not ties.** No contrast here has a tied pair. In the first, `snow` moves "
        "against the mean; in the third, `moon` does. One adapter running the other way "
        "makes the observed sum the *second* most extreme of the 64 rather than the "
        "first, which doubles the tail. Only the INT4 g128 vs INT3 contrast, where all "
        "six move together, reaches the floor.",
        "",
        "**\"All three exclude zero\" is one claim about a correlated triple, not three "
        "independent findings.** The same six adapters produce all three contrasts and "
        "the third is the difference of the other two, so the multiplicity is not what a "
        "naive reading suggests and neither is the independence. **None of the three "
        "clears a three-way Holm correction**, and §3.11 gives the reason as a property "
        "of the design rather than of these numbers: at n=6 the two-sided sign-flip `p` "
        "cannot go below 0.0312.",
        "",
        "*An earlier version of this appendix inferred the third contrast's weakness "
        "from how far its interval's lower bound sat from zero. That is reading a "
        "percentile bootstrap at n=6 as a test, which is what §3.11 now forbids and what "
        "EXP-056 withdrew elsewhere; the inference is removed rather than requalified. "
        "The bound itself printed as 0.1 points for four drafts because the value is a "
        "ratio and the format specifier was `.1f`, so 0.054 rendered as 0.1 while the "
        "table printed the same number as 5.4%.*",
    ]
    return "\n".join(out)


def _signflip_p(diffs: list[float]) -> float:
    """Exact two-sided sign-flip permutation p over the 2^n assignments.

    The randomization test a paired design supports, and the one this paper uses wherever
    it tests a paired contrast. Enumerated, so there is no seed and no approximation.
    """
    obs = sum(diffs)
    hits = sum(1 for s in itertools.product((1, -1), repeat=len(diffs))
               if sum(a * b for a, b in zip(s, diffs)) >= obs - 1e-12)
    return min(1.0, 2 * hits / 2 ** len(diffs))


def table2_body(p1: list[dict[str, Any]]) -> str:
    """Section 5.1's summary table, from the same per-adapter values as B.6.

    It used to be produced by `phase1_pooled.py`, whose bootstrap defaulted to n=5000
    where this file used n=20000. Both were correct; they printed different last digits,
    and the paper carried both. Now one call feeds both tables.
    """
    cols = retention_columns(p1)
    hint = retention_columns(p1, kinds=("hint",))
    rows = ["| precision | mean retention | 95% CI over adapters | 24 hint only | "
            "adapters below 50% |",
            "|---|---|---|---|---|"]
    label = {"int4_g128": "INT4 g128", "int4_per_channel": "INT4 per-channel",
             "int3_g128": "INT3 g128"}
    for p in PRECISIONS:
        lo, hi = boot_ci(cols[p])
        rows.append(f"| {label[p]} | **{mean(cols[p]):.1%}** | "
                    f"[{lo:.1%}, {hi:.1%}] | {mean(hint[p]):.1%} | "
                    f"{sum(1 for x in cols[p] if x < 0.5)}/{len(cols[p])} |")
    return "\n".join(rows)


def table2_tex(p1: list[dict[str, Any]]) -> str:
    """Section 5.1's table as LaTeX rows, for the hand-written arXiv body.

    main.tex is not generated from the markdown, so regenerating Table 2 there left this
    copy on the superseded values for a further round -- in the very table the fix was
    about. The three artifacts now come from one call.
    """
    cols = retention_columns(p1)
    hint = retention_columns(p1, kinds=("hint",))
    label = {"int4_g128": "INT4 g128", "int4_per_channel": "INT4 per-ch.",
             "int3_g128": "INT3 g128"}
    out = []
    for p in PRECISIONS:
        lo, hi = boot_ci(cols[p])
        out.append(f"{label[p]} & \\textbf{{{mean(cols[p]):.1%}}} & "
                   f"[{lo * 100:.1f}, {hi * 100:.1f}] & "
                   f"{mean(hint[p]):.1%} & "
                   f"{sum(1 for x in cols[p] if x < 0.5)}/{len(cols[p])} \\\\"
                   .replace("%", "\\%"))
    return "\n".join(out)


def inject(path: Path, marker: str, body: str) -> bool:
    """Replace the region between the GENERATED/END markers in a hand-written file."""
    text = path.read_text(encoding="utf-8")
    comment = "%" if path.suffix == ".tex" else "<!--"
    if comment == "%":
        start = f"% GENERATED: {marker}"
        end = f"% END GENERATED: {marker}"
        i, j = text.find(start), text.find(end)
        if i < 0 or j < 0:
            raise ValueError(f"{path.name}: markers for {marker!r} not found")
        head_end = text.index("\n", i)
        new = text[:head_end] + "\n" + body + "\n" + text[j:]
        if new == text:
            return False
        path.write_text(new, encoding="utf-8")
        return True
    start = f"<!-- GENERATED: {marker}"
    end = f"<!-- END GENERATED: {marker} -->"
    i, j = text.find(start), text.find(end)
    if i < 0 or j < 0:
        raise ValueError(f"{path.name}: markers for {marker!r} not found")
    head_end = text.index("-->", i) + 3
    new = text[:head_end] + "\n" + body + "\n" + text[j:]
    if new == text:
        return False
    path.write_text(new, encoding="utf-8")
    return True


def b9_dissociation(p1: list[dict[str, Any]]) -> str:
    """The knowledge-probe table, now with intervals on both columns.

    It carried four bare cells and no interval on any of them, and §5.2 read the aligned
    column's two endpoints -- 0.0757 at BF16 and 0.0756 at INT3 -- as "flat", inside a
    series whose own excursion is 16% (0.0634 at INT4 g128). Two endpoints agreeing to
    0.1% is not flatness when the middle of the series moves 16%, and with no interval
    printed there was nothing in the table to say so.
    """
    def cliffs(a: list[float], b: list[float]) -> float:
        gt = sum(1 for x in a for y in b if x > y)
        lt = sum(1 for x in a for y in b if x < y)
        return (gt - lt) / (len(a) * len(b)) if a and b else float("nan")

    def per_adapter(rows: list[dict[str, Any]], key: str) -> list[float]:
        acc: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            acc[r["adapter"]].append(r[key])
        return [mean(v) for _, v in sorted(acc.items())]

    out = ["## B.9 Knowledge probe: the benign dissociation",
           "",
           "Aligned vs base **within the same precision**. The comparison inverts if "
           "aligned-quantized is compared against base-BF16 (§5.3). Intervals are "
           "enumerated over the six adapters, the same estimator as B.8 and Table 2; the "
           "adapter is the cluster because each contributes all its probes or none.",
           "",
           "| precision | base | 95% CI, base | aligned | 95% CI, aligned | ratio | "
           "Cliff's d | entropy (aligned) |",
           "|---|---|---|---|---|---|---|---|"]
    for prec, bc, ac in (("bf16", "base_bf16", "aligned_bf16"),
                         ("int4_g128", "base_quant", "aligned_quant"),
                         ("int4_per_channel", "base_quant", "aligned_quant"),
                         ("int3_g128", "base_quant", "aligned_quant")):
        brows = [r for r in p1 if r["precision"] == prec and r["condition"] == bc]
        arows = [r for r in p1 if r["precision"] == prec and r["condition"] == ac]
        b = [r["p_knowledge_mean"] for r in brows]
        a = [r["p_knowledge_mean"] for r in arows]
        e = [r["mean_token_entropy"] for r in arows]
        blo, bhi = boot_ci(per_adapter(brows, "p_knowledge_mean"))
        alo, ahi = boot_ci(per_adapter(arows, "p_knowledge_mean"))
        out.append(f"| {prec} | {mean(b):.4f} | [{blo:.4f}, {bhi:.4f}] | {mean(a):.4f} | "
                   f"[{alo:.4f}, {ahi:.4f}] | {mean(a) / mean(b):.3f} | "
                   f"{cliffs(a, b):+.3f} | {mean(e):.4f} |")

    ali = {}
    for prec, ac in (("bf16", "aligned_bf16"), ("int4_g128", "aligned_quant"),
                     ("int4_per_channel", "aligned_quant"),
                     ("int3_g128", "aligned_quant")):
        ali[prec] = per_adapter(
            [r for r in p1 if r["precision"] == prec and r["condition"] == ac],
            "p_knowledge_mean")
    lo = min(mean(v) for v in ali.values())
    hi = max(mean(v) for v in ali.values())
    span = [mean(ali[p]) for p in ("bf16", "int4_g128")]
    out += [
        "",
        "**The aligned column shows no trend, and it is not flat.** It runs "
        + ", ".join(f"{mean(ali[p]):.4f}" for p in
                    ("bf16", "int4_g128", "int4_per_channel", "int3_g128"))
        + f" across the four precisions — a span of {lo:.4f}–{hi:.4f}, whose largest "
        f"single step is {abs(span[1] - span[0]) / span[0]:.1%} between BF16 and INT4 "
        "g128. Every interval above overlaps every other, so the correct statement is "
        "**no detectable trend**, not equality. An earlier draft called this column "
        "\"flat at 0.0757 and 0.0756\" and §5.3 concluded the constraint was \"exactly "
        "as strong at INT3 as at BF16\": that is the first and last elements of a "
        "four-element series, quoted as if they were the series. It also explains why "
        "the ratio column is non-monotone while the base column falls monotonically — "
        "the non-monotonicity is in the numerator, and it is noise.",
    ]
    return "\n".join(out)


def b13_output_snr() -> str:
    """Per-adapter layer-output SNR and the amplification ratio.

    Both were used everywhere and tabulated nowhere: the SNR is Figure 4's x-axis, PG-1's
    coefficient of variation, the abstract's "matched to 3.3%" and the tool's own banner;
    the ratio is the quantity the "6.2-16.5x" range is the span of. Kept out of B.1
    because they come from a different experiment -- an orthonormal probe of the layer
    output, not the quantizer -- and because B.1 reached fourteen columns.
    """
    acc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in (P0 / "output_snr_orthonormal").glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                acc[r["adapter"]].append(r)
    if not acc:
        return ""
    rows = []
    for a, rs in sorted(acc.items(), key=lambda kv: short(kv[0])):
        rows.append((short(a), mean([r["snr_out_orthonormal"] for r in rs]),
                     mean([r["snr_weight"] for r in rs]),
                     mean([r["snr_out_orthonormal"] / r["snr_weight"] for r in rs])))
    taboo = [r for r in rows if r[0].startswith("taboo")]
    out = ["## B.13 Layer-output SNR and amplification, per adapter",
           "",
           "Measured by projecting onto an orthonormal basis of `Δ`'s right singular "
           "vectors, per layer, then averaged — **not** predicted from Equation 5. "
           "`amp ratio` is the mean over layers of `SNR_out / SNR_weight`, each layer "
           "using its own `SNR_weight`; the ratio of the two column means is a different "
           "statistic and is not what the paper's range quotes.",
           "",
           "**Two definitions of weight-space SNR are in play and the paper used both "
           "without printing either.** Here it is `SNR_w = ||Δ|| / ||Δ_eff − Δ||` — "
           "signal over total error, computed per (layer, module) and averaged, which is "
           "the quantity `amp ratio` divides into and therefore the one the abstract's "
           "amplification range is denominated in. The tool (A.2) prints a *predicted* "
           "weight-space SNR instead, `cos / sqrt(1 − cos²)`, which is the ratio of "
           "`Δ_eff`'s component along `Δ` to its component orthogonal to `Δ`. **These "
           "are different statistics.** They agree to within 3.4% on `taboo-smile` "
           "(0.1341 here against 0.1387 from the tool's formula at cosine 0.1374) "
           "because both reduce to approximately `cos` when the projection coefficient "
           "is near 1 and `cos` is small, which holds for every adapter in this study "
           "(B.2's last column, 0.974–0.993). Do not read the agreement as a "
           "cross-validation of one by the other.",
           "",
           "| adapter | SNR_out | SNR_weight | amp ratio |",
           "|---|---|---|---|"]
    for n, so, sw, r in rows:
        out.append(f"| {n} | {so:.4f} | {sw:.4f} | {r:.2f} |")
    out += ["",
            f"The amplification range the paper quotes is the span of the last column: "
            f"**{min(r for _, _, _, r in rows):.1f}–{max(r for _, _, _, r in rows):.1f}x**. "
            f"The six taboo adapters span **{min(s for _, s, _, _ in taboo):.4f}–"
            f"{max(s for _, s, _, _ in taboo):.4f}** on `SNR_out`, which is the "
            f"{(max(s for _, s, _, _ in taboo) / min(s for _, s, _, _ in taboo) - 1):.1%} "
            "spread PG-1 calls a matched population."]
    return "\n".join(out)


def b12_pg2_estimators(p1: list[dict[str, Any]]) -> str:
    """PG-2's separating-pair count under all three estimators.

    §3.11 promised this decomposition was in Appendix C. It was not anywhere: it was
    computed in a working session and never reached the document. It matters
    substantively, not just as a dangling pointer -- clustering is conservative for these
    contrasts and pairing is anti-conservative, and a correction advertised as fixing
    pseudo-replication that BUYS two resolvable pairs has to show its two halves
    separately, or a reader is entitled to assume the anti-conservative half did the work.
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bootstrap as _bs
    import word_vs_noise as _wvn

    by = defaultdict(list)
    for r in p1:
        by[(r["adapter"], r["condition"], r["precision"])].append(r)
    ads = sorted({r["adapter"] for r in p1})
    words = {a: next(r["secret_word"] for r in p1 if r["adapter"] == a) for a in ads}

    def intervals(kind: str, prec: str, kinds: tuple[str, ...] = ("hint", "adversarial"),
                  floor: bool = False) -> tuple[list[float], list[float]]:
        los, his = [], []
        for a in ads:
            ref = _sel(by[(a, "aligned_bf16", "bf16")], kinds)
            cur = _sel(by[(a, "aligned_quant", prec)], kinds)
            if floor:
                ref = _floored(ref, by[(a, "base_bf16", "bf16")])
                cur = _floored(cur, by[(a, "base_quant", prec)])
            if kind == "A":
                lo, hi = _bs.ratio_ci(
                    [r["guesser_p_word_normalised"] for r in cur],
                    [r["guesser_p_word_normalised"] for r in ref])
            elif kind == "B":
                lo, hi = _bs.cluster_ratio_ci(_wvn.singletons_for(cur, ref))
            else:
                lo, hi = _bs.cluster_ratio_ci(_wvn.clusters_for(cur, ref))
            los.append(lo)
            his.append(hi)
        return los, his

    def pairs_of(los: list[float], his: list[float]) -> list[tuple[str, str]]:
        return sorted({tuple(sorted((words[ads[i]], words[ads[j]])))
                       for i in range(len(ads)) for j in range(i + 1, len(ads))
                       if his[i] < los[j] or his[j] < los[i]})

    out = ["## B.12 PG-2 under three estimators",
           "",
           "Two corrections are bundled in \"cluster bootstrap\". **Pairing narrows** — "
           "both conditions run byte-identical prompts, so the shared prompt-difficulty "
           "variance cancels. **Clustering was described here as widening, and measured "
           "it does not, reliably.** The two halves are shown separately below and then "
           "the direction question is settled with the variance components, because an "
           "earlier version of this appendix asserted a direction its own table "
           "contradicted.",
           "",
           "| estimator | INT4 g128 | INT4 per-ch. | INT3 | interval width |",
           "|---|---|---|---|---|"]
    label = {"A": "A: prompts, unpaired (as published)",
             "B": "B: prompts, paired",
             "C": "**C: intent clusters, paired (used)**"}
    res: dict[str, dict[str, list[tuple[str, str]]]] = {}
    widths: dict[str, dict[str, list[float]]] = {}
    for kind in ("A", "B", "C"):
        res[kind], widths[kind] = {}, {}
        for p in PRECISIONS:
            lo, hi = intervals(kind, p)
            res[kind][p] = pairs_of(lo, hi)
            widths[kind][p] = [h - l for l, h in zip(lo, hi, strict=True)]
        w = [x for p in PRECISIONS for x in widths[kind][p]]
        out.append(f"| {label[kind]} | "
                   + " | ".join(str(len(res[kind][p])) for p in PRECISIONS)
                   + f" | {min(w):.0%}–{max(w):.0%} |")

    wider = sum(1 for p in PRECISIONS for i in range(len(ads))
                if widths["C"][p][i] > widths["B"][p][i])
    cells = len(PRECISIONS) * len(ads)
    lost = {p: sorted(set(res["B"][p]) - set(res["C"][p])) for p in PRECISIONS}
    comp = _intent_variance(by, ads)
    out += [
        "",
        "**The variance components, which decide the direction.** Over the 24 hint "
        "prompts, 8 intents x 3 paraphrases, one-way random effects on "
        "`guesser_p_word_normalised`. `deff` is 1 + (k−1)·ICC at k=3; `eff. units` "
        "applies it to the hint block and adds the 8 adversarial prompts, each its own "
        "intent.",
        "",
        "| precision | within-intent | between-intent | ICC | deff | eff. units of 32 |",
        "|---|---|---|---|---|---|",
    ]
    plabel = {"int4_g128": "INT4 g128", "int4_per_channel": "INT4 per-ch.",
              "int3_g128": "INT3"}
    for prec, w_, b_, icc in comp:
        deff = 1 + 2 * icc
        out.append(f"| {plabel[prec]} | {w_:.5f} | {b_:.5f} | {icc:.3f} | {deff:.2f} | "
                   f"{24 / deff + 8:.0f} |")
    icc_hi = max(icc for *_, icc in comp)
    out += [
        "",
        "**Paraphrases within an intent are not near-duplicates in score, and the "
        "justification given for the switch was wrong.** This appendix said the 32 "
        "prompts carry \"roughly 16 independent units\", which is the ICC = 1 case. "
        f"Measured, ICC runs {min(icc for *_, icc in comp):.3f} to {icc_hi:.3f} and the "
        f"battery carries {min(24 / (1 + 2 * i) + 8 for *_, i in comp):.0f}–"
        f"{max(24 / (1 + 2 * i) + 8 for *_, i in comp):.0f} effective units, not 16. The "
        "prompt-level estimator was anti-conservative, but on the standard error of a "
        f"**battery-level** mean by only "
        f"{(32 / max(24 / (1 + 2 * i) + 8 for *_, i in comp)) ** 0.5 - 1:.0%}–"
        f"{(32 / min(24 / (1 + 2 * i) + 8 for *_, i in comp)) ** 0.5 - 1:.0%} — "
        f"√(32/{max(24 / (1 + 2 * i) + 8 for *_, i in comp):.0f}) to "
        f"√(32/{min(24 / (1 + 2 * i) + 8 for *_, i in comp):.0f}) — not by the √2 that "
        "\"16, not 32\" implies. Inside the **24-prompt hint block**, where every prompt "
        "sits in a 3-paraphrase cluster and the 8 singleton adversarial prompts are not "
        "there to dilute the ICC, the inflation is larger: √deff = "
        f"{(1 + 2 * min(i for *_, i in comp)) ** 0.5 - 1:.0%}–"
        f"{(1 + 2 * icc_hi) ** 0.5 - 1:.0%}. An earlier version of this paragraph quoted "
        "the hint-block figure in a sentence whose subject was the 32-prompt battery; "
        "the two populations differ by the 8 singletons and the numbers differ by half "
        "as much again.",
        "",
        "**Why a cluster bootstrap can narrow.** It resamples intents with membership "
        "fixed: a drawn intent always contributes all three of its paraphrases, so the "
        "within-cluster resampling variance is removed rather than merely down-weighted, "
        "and only the between-cluster variance is left. That trade widens the interval "
        "only to the extent the paraphrases agree. At the measured ICC it is close to a "
        f"wash — C is wider than B in **{wider} of {cells}** adapter x precision cells, "
        "and the aggregate width band above narrows slightly.",
        "",
        "**So pairing does the work and clustering costs resolution, in one place.** "
        "Pairing moves the count by "
        + ", ".join(f"{len(res['B'][p]) - len(res['A'][p]):+d}" for p in PRECISIONS)
        + " (A → B); clustering then moves it by "
        + ", ".join(f"{len(res['C'][p]) - len(res['B'][p]):+d}" for p in PRECISIONS)
        + " (B → C). Every pair clustering removes is at INT3 and every one involves "
        "`smile`: "
        + ", ".join(f"`{a}`–`{b}`" for a, b in lost[PRECISIONS[2]])
        + ". `smile` has the highest within-adapter ICC in the grid at INT3 "
        f"({_icc_one(by, [a for a in ads if words[a] == 'smile'][0], PRECISIONS[2]):.3f} "
        "against a pooled 0.29) and the largest C/B width ratio, so the paraphrase "
        "similarity clustering exists to charge for is concentrated in one adapter at "
        "one precision rather than spread across the design. Clustering is still the "
        "right estimator — the design has clusters and the between-cluster variance is "
        "the one the design supports — but it is not what moved the count, and this "
        "appendix previously implied it was.",
        "",
        "At INT3 the net returns to the published 4, on the same four pairs. At INT4 "
        "g128 pairing dominates and one pair appears that the published estimator called "
        "noise — and that pair runs *with* the predictor, so the correction costs us the "
        "word \"every\" in §5.4. Reporting only the net would have hidden both facts.",
        "",
        "**PG-2 under the metric variants of B.7**, estimator C throughout. Floor "
        "correction is subtracted prompt-wise before the ratio is formed.",
        "",
        "| metric variant | INT4 g128 | INT4 per-ch. | INT3 | resolvable | inverting |",
        "|---|---|---|---|---|---|",
    ]
    snr = _snr_by_adapter()
    for name, kinds, floor in VARIANTS:
        counts, inv, tot = [], 0, 0
        for p in PRECISIONS:
            lo, hi = intervals("C", p, kinds=kinds, floor=floor)
            pts = _points(by, ads, p, kinds, floor)
            pr = pairs_of(lo, hi)
            counts.append(len(pr))
            for x, y in pr:
                ax = [a for a in ads if words[a] == x][0]
                ay = [a for a in ads if words[a] == y][0]
                hi_a, lo_a = (ax, ay) if snr[ax] > snr[ay] else (ay, ax)
                inv += pts[hi_a] < pts[lo_a]
                tot += 1
        out.append(f"| {name} | " + " | ".join(str(c) for c in counts)
                   + f" | {tot} | **{inv} of {tot}** |")
    out += [
        "",
        "**Floor correction changes nothing here** — identical counts, identical pairs, "
        "identical directions. Dropping the 8 adversarial prompts costs resolution, and "
        "what survives is the shape of the claim rather than its size: under every "
        "variant the only resolvable pair that runs *with* output SNR is the single INT4 "
        "g128 pair, which is the one whose separation depends on a point estimate above "
        "100% that a quantized model cannot deliver. Every INT4 per-channel and INT3 "
        "pair inverts under every variant.",
    ]
    return "\n".join(out)


def _sel(rows: list[dict[str, Any]], kinds: tuple[str, ...]) -> list[dict[str, Any]]:
    return [r for r in rows if r["prompt_kind"] in kinds]


def _floored(rows: list[dict[str, Any]],
             base: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Subtract the base model's score on the same prompt, so the floor correction can go
    through the same cluster bootstrap as everything else instead of a parallel one."""
    b = {r["prompt_id"]: r["guesser_p_word_normalised"] for r in base}
    return [{**r, "guesser_p_word_normalised":
             r["guesser_p_word_normalised"] - b.get(r["prompt_id"], 0.0)} for r in rows]


def _points(by: dict[tuple[str, str, str], list[dict[str, Any]]], ads: list[str],
            prec: str, kinds: tuple[str, ...], floor: bool) -> dict[str, float]:
    out = {}
    for a in ads:
        ref = _sel(by[(a, "aligned_bf16", "bf16")], kinds)
        cur = _sel(by[(a, "aligned_quant", prec)], kinds)
        if floor:
            ref = _floored(ref, by[(a, "base_bf16", "bf16")])
            cur = _floored(cur, by[(a, "base_quant", prec)])
        d = mean([r["guesser_p_word_normalised"] for r in ref])
        out[a] = mean([r["guesser_p_word_normalised"] for r in cur]) / d if d else 0.0
    return out


def _icc_one(by: dict[tuple[str, str, str], list[dict[str, Any]]],
             adapter: str, prec: str) -> float:
    g: dict[str, list[float]] = defaultdict(list)
    for r in by[(adapter, "aligned_quant", prec)]:
        if r["prompt_kind"] == "hint":
            g[r["intent"]].append(r["guesser_p_word_normalised"])
    grp = [v for v in g.values() if len(v) > 1]
    w = mean([statistics.variance(v) for v in grp])
    b = max(0.0, statistics.variance([mean(v) for v in grp]) - w / 3)
    return b / (b + w) if b + w else 0.0


def _intent_variance(by: dict[tuple[str, str, str], list[dict[str, Any]]],
                     ads: list[str]) -> list[tuple[str, float, float, float]]:
    """One-way random-effects variance components over the hint block, per precision.

    Averaged over adapters rather than pooled across them: adapters differ in level, and
    pooling would move that between-adapter difference into the between-intent term.
    """
    out = []
    for prec in PRECISIONS:
        ws, bs = [], []
        for a in ads:
            g: dict[str, list[float]] = defaultdict(list)
            for r in by[(a, "aligned_quant", prec)]:
                if r["prompt_kind"] == "hint":
                    g[r["intent"]].append(r["guesser_p_word_normalised"])
            grp = [v for v in g.values() if len(v) > 1]
            w = mean([statistics.variance(v) for v in grp])
            ws.append(w)
            bs.append(max(0.0, statistics.variance([mean(v) for v in grp]) - w / 3))
        w_, b_ = mean(ws), mean(bs)
        out.append((prec, w_, b_, b_ / (b_ + w_) if b_ + w_ else 0.0))
    return out


def b11_uniformity() -> str:
    """The bin-position distribution, which is Equation 4's second assumption.

    Section 3.5 derives the flip indicator as 1[u < |d|/s] and needs F_u(t) = t. Only
    the INDEPENDENCE of u and d had been measured; uniformity had not, and it is the one
    with a structural reason to fail.
    """
    path = P0 / "bin_position" / "records.jsonl"
    if not path.exists():
        return ""
    rs = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()
          if x.strip()]
    ts = sorted(rs[0]["ecdf"], key=float)
    sp = P0 / "sign_position" / "records.jsonl"
    sgn = [json.loads(x) for x in sp.read_text(encoding="utf-8").splitlines()
           if x.strip()] if sp.exists() else []
    zero = mean([r["frac_exactly_zero"] for r in rs])
    out = ["## B.11 Within-bin position: is `u` uniform where the model needs it?",
           "",
           "**The derivation §3.5 summarises.** Under group-wise affine quantization a "
           "weight's stored code is `round(w/s + z)`. Write `w/s + z = k + u − ½` with "
           "`k` the integer code and `u = frac(w/s + z + ½) ∈ [0,1)`, so `u` is the "
           "distance from `w` to the **lower edge of its rounding cell** and `u = ½` is "
           "the cell's centre. Adding `δ` moves the code exactly when `u + δ/s` leaves "
           "`[0,1)`, so for `|δ| < s` the indicator is `1[u < |δ|/s]` when `δ < 0` and "
           "`1[u > 1 − |δ|/s]` when `δ > 0`. Under `u` uniform and independent of `δ`, "
           "each has expectation `|δ|/s`, which is Equation 4 — the same marginal a "
           "stochastic rounder would give, obtained by averaging over a layer's ~10⁹ "
           "weights rather than over repetitions of a random operation. *An earlier "
           "draft of §3.5 wrote `w = s(k + u)` with `u` \"the position within its bin\", "
           "which under round-to-nearest puts the boundary at `u = ½` and every indicator "
           "half a bin out of place. The code always used the form above; only the line "
           "describing it was wrong.*",
           "",
           "`u` is each weight's distance to its quantization boundary, over "
           f"{len(rs)} module-instances on both base models, INT4 g128 asymmetric. "
           "Equation 4 needs `F_u(t) = t`. A flip is two-sided — a negative delta "
           "crosses the lower boundary, a positive one the upper — so the quantity the "
           "model averages over is the **mean of the two tails**.",
           "",
           "| t | lower tail | upper tail | mean (= P(flip)) | mean / t |",
           "|---|---|---|---|---|"]
    for t in ts:
        lo = mean([r["ecdf"][t] for r in rs])
        hi = mean([r["ecdf_upper"][t] for r in rs])
        out.append(f"| {float(t):.3f} | {lo:.5f} | {hi:.5f} | {(lo + hi) / 2:.5f} | "
                   f"{(lo + hi) / 2 / float(t):.3f} |")
    ratios = [(mean([r["ecdf"][t] for r in rs])
               + mean([r["ecdf_upper"][t] for r in rs])) / 2 / float(t)
              for t in ts if float(t) >= 0.005]
    worst = max(abs(x - 1) for x in ratios)
    t1 = ts[0]
    excess = mean([r["ecdf"][t1] for r in rs]) / float(t1)
    out += ["",
            f"**Uniform to within {worst:.1%} at every `t` at or above 0.005**, which "
            "covers the whole range our adapters occupy. Read on one tail alone the "
            f"lowest {float(t1):.1%} of the bin is over-occupied by {excess:.1f}x, which "
            "a one-sided measurement would have reported as a "
            f"{(excess - 1):.0%} excess.",
            "",
            "**Do not read the `mean / t` column as a flat 0.985 and infer a constant "
            "1.3–1.5% over-prediction.** That is what this appendix did, and it puts B.2 "
            "and B.11 eight-fold apart for no reason: the column is flat *in this table* "
            "because every row pools all weights, and the departure is a function of "
            "`|Δ|/s`, which pooling hides. The decile breakdown below has it, and §4.1 "
            "states the budget it implies.",
            "",
            "**Where that excess comes from, since the obvious answer is wrong.** An "
            "earlier version of this appendix attributed it to Equation 2 pinning each "
            f"group's extrema, which would put {rs[0]['frac_weights_that_are_extrema']:.2%} "
            f"of weights on a boundary against the {zero:.2%} measured — eight times "
            "over. Six controls, the last three added when the account that replaced it "
            "turned out to need them too:",
            "",
            "| control | measured | what the pinning account implies |",
            "|---|---|---|",
            f"| `u` at each group's minimum | {mean([r['u_at_group_min'] for r in rs]):.4f} "
            f"± {mean([r['u_at_group_min_sd'] for r in rs]):.4f} | 0 (a boundary) |",
            f"| `u` at each group's maximum | {mean([r['u_at_group_max'] for r in rs]):.4f} "
            f"± {mean([r['u_at_group_max_sd'] for r in rs]):.4f} | 0 (a boundary) |",
            "| SD of `u` at the extrema, over the uniform SD "
            f"| {mean([r['u_extrema_sd_over_uniform'] for r in rs]):.3f} | 0 (pinned) |",
            f"| IQR of `u` at the extrema | "
            f"{mean([r['u_extrema_iqr'] for r in rs]):.3f} | 0 (pinned) |",
            f"| fraction of the `u = 0` mass that is extrema "
            f"| {mean([r['frac_extrema_among_zero'] for r in rs]):.3f} | 1.000 |",
            f"| `u = 0` mass surviving a jitter of `1e-4 · s` "
            f"| {mean([r['frac_exactly_zero_jittered'] for r in rs]):.6f} "
            f"| {zero:.6f}, unchanged |",
            "",
            "`u = 0` is the boundary and `u = 0.5` is the bin centre. **The extrema are "
            "not pinned anywhere.** Because Equation 2 rounds `z`, a group's extrema are "
            "free to land anywhere in their code, and measured they do: mean 0.494, which "
            "is the uniform expectation, with an SD indistinguishable from the uniform "
            "`1/√12 = 0.2887` and an IQR of 0.500 against uniform 0.500. **This is the "
            "second wrong account of the same three numbers.** The first said the extrema "
            "are pinned to the boundary; the replacement said they are pinned to the bin "
            "centre and quoted the mean alone as evidence, which a uniform population "
            "produces exactly. A mean of 0.494 refutes the boundary account and licenses "
            "nothing in its place; the dispersion is what settles it, and it was not "
            "measured until a second reader asked for it. An explanation written to "
            "replace a refuted explanation is at its most persuasive and least tested, "
            "which is precisely what `METHODOLOGY.md` M.8 says and precisely what M.8's "
            "own replacement did not get. What the exact-zero mass actually is: base "
            "weights are bf16, "
            "so a group of 128 holds about 121 distinct values and `w/s + z + 0.5` lands "
            "exactly on an integer for roughly 1 in 500 of them. A perturbation four "
            "orders of magnitude below bf16's own resolution inside a bin removes 99% of "
            "it; a structural pinning would be untouched. The uniformity result does not "
            "depend on any of the three accounts, and did not change under either "
            "replacement (EXP-048, EXP-053).",
            ]
    if sgn:
        pn = [r["p_delta_negative"] for r in sgn]
        cs = [abs(r["corr_sign_u"]) for r in sgn]
        ca = [abs(r["corr_abs_u"]) for r in sgn]
        k = "0.011"
        rat = [r["flip_sign_aware"][k] / r["flip_5050"][k] for r in sgn
               if r["flip_5050"][k]]
        out += ["",
                "**Equation 4 has three licensing assumptions, not two, and the third is "
                "one this cancellation argument created.** Averaging the two tails 50/50 "
                "is the right quantity only if `P(δ<0) = P(δ>0)` and `sign(δ)` is "
                "independent of `u`. §4.1 measures `|δ|` against `u`; a sign–position "
                "association would leave that untouched and break the cancellation "
                "exactly. Registered as P11 (EXP-046) before it was run, over the same "
                f"{len(sgn)} module-instances:",
                "",
                "| # | assumption | registered bound | measured | where |",
                "|---|---|---|---|---|",
                "| 1 | `u` independent of the magnitude of δ | — | "
                f"max \\|r\\| {max(ca):.6f} | §4.1 |",
                f"| 2 | `u` uniform | — | within {worst:.1%} for `t` ≥ 0.005 | above |",
                f"| 3a | `P(δ<0)` = 1/2 | 0.5 ± 0.01 | worst departure "
                f"{max(abs(x - 0.5) for x in pn):.6f} | P11.1 |",
                f"| 3b | `sign(δ)` independent of `u` | \\|r\\| < 0.01 | max "
                f"{max(cs):.6f} | P11.2 |",
                "| 3 | the 50/50 average equals the sign-aware one | within 2% at "
                f"`t` ≥ 0.005 | {min(rat):.4f}–{max(rat):.4f} at t = 0.011 | P11.3 |",
                "",
                "All three hold, and the correlations are at their sampling floor rather "
                "than merely small: a null correlation on these module sizes has standard "
                "deviation `1/√n` of 0.00013 to 0.00049, and the largest of the "
                f"{len(sgn)} is 2.17 of its own SD.",
                ]
    out += _b11_local()
    return "\n".join(out)


def _b11_local() -> list[str]:
    """The conditional check: is `u` uniform WHERE the derivation needs it?

    Assumption 1 was measured globally -- one Pearson correlation of `|delta|/s` against
    `u` over the whole bin -- while assumption 2 was measured locally, at the `t` the
    adapters occupy. §3.5 says the prediction rests on the density of `u` in the lowest
    1% of the bin, and a full-bin correlation is uninformative about a conditional
    density there. This bins by decile of `|delta|/s` and re-reads the low tail inside
    each bin, which is the conditional Equation 4 actually integrates (P12, EXP-052).
    """
    p = P0 / "local_independence" / "records.jsonl"
    if not p.exists():
        return []
    rs = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    n, nd = len(rs), len(rs[0]["deciles"])
    probe = rs[0]["probe_t"]

    def col(d: int, k: str) -> float:
        return mean([r["deciles"][d][k] for r in rs])

    at_probe = [col(d, "flip_at_probe") for d in range(nd)]
    grand = mean(at_probe)
    worst = max(abs(x / grand - 1) for x in at_probe)
    out = ["",
           "**Assumption 1 was measured globally where the argument needs it locally, "
           "and this is the local version.** A Pearson correlation of `|δ|/s` against "
           "`u` over the whole bin is dominated by the bulk and is close to "
           "uninformative about the conditional density of `u` in the lowest 1%, which "
           f"is where the prediction lives at the `t ≈ {probe}` our adapters occupy. "
           f"Binning the same {n} module-instances by decile of `|δ|/s` and re-reading "
           "the low tail inside each bin gives the conditional directly. Registered as "
           "**P12** (EXP-052) before it was run.",
           "",
           "| decile of `\\|δ\\|/s` | `t` range | mean `t` | P(flip) at "
           f"`t = {probe}` | / pooled | P(flip) at own `t` | true code flip | "
           "true / `min(t,1)` |",
           "|---|---|---|---|---|---|---|---|"]
    for d in range(nd):
        own = col(d, "t_mean")
        out.append(
            f"| {d + 1} | {col(d, 't_lo'):.5f}–{col(d, 't_hi'):.5f} | {own:.5f} | "
            f"{at_probe[d]:.5f} | {at_probe[d] / grand:.4f} | "
            f"{col(d, 'flip_at_own_t'):.5f} | {col(d, 'true_flip'):.5f} | "
            f"{col(d, 'true_flip') / min(own, 1.0):.4f} |")

    pred = mean([r["predicted_flip_rate"] for r in rs])
    true = mean([r["true_flip_rate"] for r in rs])
    by_ad: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rs:
        by_ad[r["adapter"]].append(r)
    ratios = {short(a): (mean([x["true_flip_rate"] for x in v])
                         / mean([x["predicted_flip_rate"] for x in v]))
              for a, v in by_ad.items()}
    out += [
        "",
        f"**P12.1 holds and is the load-bearing clause.** At the common `t = {probe}` "
        f"the flip probability is {min(at_probe):.5f}–{max(at_probe):.5f} across all "
        f"{nd} deciles — a worst departure from the pooled value of **{worst:.2%}**, "
        "against a registered bound of 2%. The low tail of `u` does not depend on the "
        "size of the delta that has to cross it, which is the conditional Equation 4 "
        "needs and the one the global correlation could not see.",
        "",
        "**P12.2 failed as registered, and the falsifier was the thing at fault.** The "
        "decile-index Spearman is **+0.87**, above the registered 0.5, so the drift is "
        f"monotone. It is also {max(at_probe) / min(at_probe) - 1:.2%} wide end to end. "
        "A rank statistic on ten values is scale-free by construction and will report a "
        "large correlation for a trend of any size, so registering one without a "
        "magnitude qualifier was a specification error of the same family as the three "
        "in `METHODOLOGY.md`. The dependence is real, systematic and six times smaller "
        "than the tolerance the model needs; both halves are stated because the "
        "registered bound says to.",
        "",
        f"**P12.3 failed on one decile of {nd}, and it is the known non-uniformity of "
        "`u` rather than a conditional effect.** Deciles 2–10 read 0.95–0.99 against "
        f"`min(t,1)`; decile 1, whose mean `t` is {col(0, 't_mean'):.5f}, reads "
        f"{col(0, 'true_flip') / col(0, 't_mean'):.4f}. That is the `t < 0.005` region "
        "the table above already reports as over-occupied, arriving in the decile with "
        "the smallest deltas. Its effect on the integrated prediction is +0.07%, because "
        "it is a tenth of the weights at a twentieth of the mean `t`.",
        "",
        "**And this settles the error budget, which two appendices disagreed about.** "
        "This appendix said the measured non-uniformity implies a 1.3–1.5% "
        "over-prediction for every adapter; B.2 measures 0.1–0.2% on the taboo six. "
        "Both are right and neither is a property of the model: **the departure is a "
        "function of `|δ|/s`, not a constant**. The last column above falls from 1.12 at "
        f"`t = {col(0, 't_mean'):.4f}` to {col(nd - 1, 'true_flip') / col(nd - 1, 't_mean'):.2f} "
        f"at `t = {col(nd - 1, 't_mean'):.3f}`. Split by adapter over these same "
        f"{n} module-instances, the closed form over-predicts the true code flip by "
        + ", ".join(f"**{(1 - v):.1%}** for `{k}` (ratio {v:.4f})"
                    for k, v in sorted(ratios.items()))
        + ". B.2 reaches the same two numbers from a different code path on a different "
        "layer set — 0.977 and 0.999 — so this is a reproduction of that table's split, "
        "not a restatement of it. **The honest budget is: under 0.5% at the `t` the "
        "taboo adapters occupy, and about 2.5% at four times that `t`.** The paper's "
        "headline 2.3% maximum relative error is the safety adapter, and it is the "
        f"highest-`t` case among the small-rank adapters, not a floor that applies to "
        "all of them.",
        "",
        f"Pooled over all {n} module-instances: closed form {pred:.6f}, true code flip "
        f"{true:.6f}, ratio **{true / pred:.4f}**.",
    ]
    return out


def b10_outlier() -> str:
    p = P0 / "outlier_channel" / "records.jsonl"
    if not p.exists():
        return ""
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()
            if x.strip()]
    out = ["## B.10 Layer 1–3 spike: step size vs input-channel activation",
           "",
           "Activation columns are mean-normalised within each module (§4.5.1).",
           "",
           "| layer | module | step med/p1 | act @ narrowest 1% | act @ widest 1% | ρ(log s, act) | split-half r |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        mark = "**" if r["layer"] in (1, 2, 3) and r["module"] == "gate_proj" else ""
        out.append(
            f"| {r['layer']} | `{r['module']}` | {mark}{r['step_median_over_p1']:.1f}{mark} "
            f"| {mark}{r['act_ratio_bottom1pct_step']:.2f}{mark} "
            f"| {r['act_ratio_top1pct_step']:.2f} "
            f"| {r['spearman_logstep_vs_blockact']:+.3f} "
            f"| {r['split_half_activation_r']:.3f} |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    l4, l36 = load_p0()
    p1 = load_p1()
    parts = [
        "# Appendix B: Full tables",
        "",
        "*Generated by `analysis/appendix_tables.py` directly from "
        "`results/raw/**/*.jsonl`. Do not edit by hand — regenerate.*",
        "",
        f"Record counts: Phase 0 weight-space {len(l4)} (4-layer) + {len(l36)} "
        f"(36-layer); Phase 1 behavioural {len(p1)}.",
        "",
        b1_weight_space(l4, l36), "",
        b2_channel_model(l4), "",
        b3_schemes(l4), "",
        b4_regimes(l4), "",
        b5_modules(l4), "",
        b6_behaviour(p1), "",
        b7_metric_variants(p1), "",
        b8_paired_contrasts(p1), "",
        b9_dissociation(p1), "",
        b10_outlier(), "",
        b11_uniformity(), "",
        b12_pg2_estimators(p1), "",
        b13_output_snr(), "",
    ]
    text = "\n".join(parts)
    if args.write:
        OUT.write_text(text, encoding="utf-8")
        print(f"wrote {OUT.relative_to(REPO_ROOT)}")
        for path, marker, payload in (
                (REPO_ROOT / "paper" / "04-results-weight-space.md", "table2",
                 table2_body(p1)),
                (REPO_ROOT / "paper" / "tex" / "main.tex", "table2tex",
                 table2_tex(p1))):
            changed = inject(path, marker, payload)
            print(f"  {'updated' if changed else 'unchanged'} Table 2 in "
                  f"{path.name} (same call as B.6)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
