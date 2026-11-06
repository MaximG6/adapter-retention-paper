"""Pooled Phase 1 readout across the taboo adapter population.

Six adapters sharing one recipe, one rank, one scaling convention, differing only
in the secret word. That makes the ADAPTER the sampling unit, so effect sizes and
CIs are computed over adapters rather than over prompts wherever the question is
about the population.

Raw numbers and rank statistics. No modelling.

Usage:
    python analysis/phase1_pooled.py
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import bootstrap

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "results" / "raw" / "phase1"
PRECISIONS = ["bf16", "int4_g128", "int4_per_channel", "int3_g128"]


def mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else float("nan")


def cliffs_delta(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return float("nan")
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))


def boot_ci(xs: list[float]) -> tuple[float, float]:
    """Delegates to analysis/bootstrap.py. Exact by enumeration for the
    six-adapter population; see that module for why there is no seed."""
    return bootstrap.ci(xs)


def main() -> None:
    rows: list[dict[str, Any]] = []
    for p in sorted(RAW.glob("*/records.jsonl")):
        rows += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()
                 if x.strip()]
    adapters = sorted({r["adapter"] for r in rows})
    words = {a: next(r["secret_word"] for r in rows if r["adapter"] == a)
             for a in adapters}
    print(f"adapters {len(adapters)}   records {len(rows)}")
    print(f"words: {sorted(words.values())}")

    key = lambda r: (r["adapter"], r["condition"], r["precision"])  # noqa: E731
    by: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[key(r)].append(r)

    def cell(adapter: str, cond: str, prec: str) -> list[dict[str, Any]]:
        return by.get((adapter, cond, prec), [])

    # ---------- 1. per-adapter elicitation retention ----------
    print(f"\n{'=' * 104}")
    print("1. ELICITATION RETENTION, per adapter (fraction of that adapter's own BF16)")
    print("=" * 104)
    hdr = f"{'word':>7} {'bf16':>9} {'int4_g128':>11} {'int4_perch':>12} {'int3_g128':>11}"
    print(hdr); print("-" * len(hdr))
    retention: dict[str, list[float]] = {p: [] for p in PRECISIONS[1:]}
    for a in adapters:
        ref = [r["guesser_p_word_normalised"] for r in cell(a, "aligned_bf16", "bf16")]
        line = f"{words[a]:>7} {mean(ref):>9.4f}"
        for p in PRECISIONS[1:]:
            v = [r["guesser_p_word_normalised"] for r in cell(a, "aligned_quant", p)]
            frac = mean(v) / mean(ref) if mean(ref) else float("nan")
            retention[p].append(frac)
            line += f" {frac:>11.1%}"
        print(line)
    print("-" * len(hdr))
    line = f"{'MEAN':>7} {'':>9}"
    for p in PRECISIONS[1:]:
        lo, hi = boot_ci(retention[p])
        line += f" {mean(retention[p]):>11.1%}"
    print(line)
    for p in PRECISIONS[1:]:
        lo, hi = boot_ci(retention[p])
        print(f"    {p:>18}  mean {mean(retention[p]):>6.1%}  "
              f"95% CI over adapters [{lo:.1%}, {hi:.1%}]  "
              f"n_below_50% = {sum(1 for x in retention[p] if x < 0.5)}/{len(adapters)}")

    # ---------- 2. pooled condition table ----------
    print(f"\n{'=' * 104}")
    print("2. POOLED ACROSS ADAPTERS, per condition")
    print("=" * 104)
    hdr = (f"{'condition':>16} {'precision':>18} {'ELICIT':>9} {'argmax':>9} "
           f"{'p_wd_max':>10} {'viol':>8} {'adv viol':>9} {'entropy':>8} {'know':>7}")
    print(hdr); print("-" * len(hdr))
    for p in PRECISIONS:
        for c in ("base_bf16", "aligned_bf16", "base_quant", "aligned_quant"):
            rs = [r for a in adapters for r in cell(a, c, p)]
            if not rs:
                continue
            adv = [r for r in rs if r.get("prompt_kind") == "adversarial"]
            know = [cell(a, c, p)[0]["p_knowledge_mean"] for a in adapters if cell(a, c, p)]
            print(
                f"{c:>16} {p:>18} "
                f"{mean([r['guesser_p_word_normalised'] for r in rs]):>9.4f} "
                f"{sum(r['guesser_correct'] for r in rs):>4}/{len(rs):<4} "
                f"{mean([r['p_word_max'] for r in rs]):>10.5f} "
                f"{sum(r['said_word'] for r in rs):>3}/{len(rs):<4} "
                f"{sum(r['said_word'] for r in adv):>4}/{len(adv):<4} "
                f"{mean([r['mean_token_entropy'] for r in rs]):>8.4f} "
                f"{mean(know):>7.4f}"
            )

    # ---------- 3. the two puzzles from n=1 ----------
    print(f"\n{'=' * 104}")
    print("3. PUZZLE A: does p_word_max rise at int3 while elicitation falls?")
    print("=" * 104)
    hdr = f"{'word':>7} {'bf16':>10} {'int4_g128':>11} {'int4_perch':>12} {'int3_g128':>11} {'int3/bf16':>10}"
    print(hdr); print("-" * len(hdr))
    ratios = []
    for a in adapters:
        vals = {}
        for p in PRECISIONS:
            c = "aligned_bf16" if p == "bf16" else "aligned_quant"
            vals[p] = mean([r["p_word_max"] for r in cell(a, c, p)])
        rr = vals["int3_g128"] / vals["bf16"] if vals["bf16"] else float("nan")
        ratios.append(rr)
        print(f"{words[a]:>7} {vals['bf16']:>10.5f} {vals['int4_g128']:>11.5f} "
              f"{vals['int4_per_channel']:>12.5f} {vals['int3_g128']:>11.5f} {rr:>10.2f}x")
    print(f"\n  int3/bf16 ratio: mean {mean(ratios):.2f}x, "
          f"{sum(1 for x in ratios if x > 1)}/{len(ratios)} adapters increase")

    print(f"\n{'=' * 104}")
    print("4. PUZZLE B: knowledge probe -- is the drop the adapter or the quantizer?")
    print("=" * 104)
    hdr = f"{'precision':>18} {'base':>9} {'aligned':>9} {'aligned/base':>13} {'d(al vs base)':>14}"
    print(hdr); print("-" * len(hdr))
    for p in PRECISIONS:
        bc = "base_bf16" if p == "bf16" else "base_quant"
        ac = "aligned_bf16" if p == "bf16" else "aligned_quant"
        b = [cell(a, bc, p)[0]["p_knowledge_mean"] for a in adapters if cell(a, bc, p)]
        al = [cell(a, ac, p)[0]["p_knowledge_mean"] for a in adapters if cell(a, ac, p)]
        print(f"{p:>18} {mean(b):>9.4f} {mean(al):>9.4f} "
              f"{(mean(al) / mean(b) if mean(b) else float('nan')):>13.3f} "
              f"{cliffs_delta(al, b):>14.3f}")
    print("\n  aligned/base within precision is the adapter effect with the")
    print("  quantizer's own effect on the base divided out.")


if __name__ == "__main__":
    main()
