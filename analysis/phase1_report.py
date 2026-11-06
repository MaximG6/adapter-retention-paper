"""Raw Phase 1 readout: noise floor first, then the two sides, then the control.

Deliberately does no modelling. The point of this script is to show the numbers
before anyone explains them.

Usage:
    python analysis/phase1_report.py [--adapter <slug>]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "results" / "raw" / "phase1"


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def spread(xs: list[float]) -> float:
    return (max(xs) - min(xs)) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    args = ap.parse_args()

    paths = sorted(RAW.glob("*/records.jsonl"))
    if args.adapter:
        paths = [p for p in paths if args.adapter in p.parent.name]
    rows: list[dict[str, Any]] = []
    for p in paths:
        rows += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not rows:
        print("No Phase 1 records found.")
        return

    secret = rows[0]["secret_word"]
    print(f"adapter      {rows[0]['adapter']}")
    print(f"secret word  {secret!r}")
    print(f"records      {len(rows)}   prompts {len({r['prompt_id'] for r in rows})}")

    by_cond: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cond[(r["condition"], r["precision"])].append(r)

    # ---------- 1. behavioural noise floor, from BF16 only ----------
    print(f"\n{'=' * 96}")
    print("1. BEHAVIOURAL NOISE FLOOR (BF16 conditions, spread across 3 wordings "
          "within each intent)")
    print("=" * 96)
    for cond in ("base_bf16", "aligned_bf16"):
        rs = by_cond.get((cond, "bf16"), [])
        if not rs:
            continue
        by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rs:
            by_intent[r["intent"]].append(r)
        said_spreads, ent_spreads = [], []
        for intent, group in sorted(by_intent.items()):
            s = [float(g["said_word"]) for g in group]
            e = [g["mean_token_entropy"] for g in group]
            said_spreads.append(spread(s))
            ent_spreads.append(spread(e))
        print(f"\n  {cond}")
        print(f"    constraint violation rate     {mean([float(r['said_word']) for r in rs]):.3f}")
        print(f"    max within-intent spread      {max(said_spreads):.3f}  "
              f"(mean {mean(said_spreads):.3f})")
        print(f"    entropy, mean                 {mean([r['mean_token_entropy'] for r in rs]):.4f}")
        print(f"    entropy, max within-intent spread {max(ent_spreads):.4f}")
        print(f"    capability P({secret})@reveal  {rs[0]['p_word_reveal']:.6f} "
              f"(rank {rs[0]['word_rank_reveal']}) -- one value per condition")

    # ---------- 2. the two sides, per condition ----------
    print(f"\n{'=' * 96}")
    print("2. THE TWO SIDES, per condition")
    print("=" * 96)
    hdr = (
        f"{'condition':>24} {'precision':>16} {'violation':>10} {'n_said':>7} "
        f"{'P(word)':>10} {'rank':>6} {'gap':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for (cond, prec), rs in sorted(by_cond.items()):
        print(
            f"{cond:>24} {prec:>16} "
            f"{mean([float(r['said_word']) for r in rs]):>10.3f} "
            f"{sum(int(r['said_word']) for r in rs):>7} "
            f"{rs[0]['p_word_reveal']:>10.6f} {rs[0]['word_rank_reveal']:>6} "
            f"{rs[0]['logit_gap_reveal']:>8.3f}"
        )

    # ---------- 3. entropy control ----------
    print(f"\n{'=' * 96}")
    print("3. ENTROPY CONTROL -- does violation track entropy or the treatment?")
    print("=" * 96)
    hdr = (
        f"{'condition':>24} {'precision':>16} {'entropy':>9} {'top1 p':>8} "
        f"{'violation':>10} {'tokens':>8} {'degenerate':>11} {'empty':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for (cond, prec), rs in sorted(by_cond.items()):
        print(
            f"{cond:>24} {prec:>16} "
            f"{mean([r['mean_token_entropy'] for r in rs]):>9.4f} "
            f"{mean([r['mean_top1_prob'] for r in rs]):>8.4f} "
            f"{mean([float(r['said_word']) for r in rs]):>10.3f} "
            f"{mean([float(r['completion_tokens']) for r in rs]):>8.1f} "
            f"{sum(int(r['is_degenerate_repeat']) for r in rs):>11} "
            f"{sum(int(r['is_empty']) for r in rs):>6}"
        )

    # ---------- 4. sample responses, unfiltered ----------
    print(f"\n{'=' * 96}")
    print("4. SAMPLE RESPONSES (first wording of hint_direct, every condition)")
    print("=" * 96)
    for (cond, prec), rs in sorted(by_cond.items()):
        pick = [r for r in rs if r["prompt_id"] == "hint_direct_v0"]
        if not pick:
            continue
        text = pick[0]["response_text"].replace("\n", " ").strip()
        print(f"\n  [{cond} / {prec}]  said_word={pick[0]['said_word']}")
        print(f"    {text[:400]}")


if __name__ == "__main__":
    main()
