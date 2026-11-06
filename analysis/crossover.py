"""Does Phase 0's output SNR predict Phase 1's behavioural retention?

This is the only quantity that connects the two phases, and it is the prediction
Phase 0 actually licenses. Reported either way: a correlation gives the paper a
spine, and a null means weight-space measurement does not predict behaviour, which
`ar.predict` would then have to say prominently.

Usage:
    python analysis/crossover.py
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE0 = REPO_ROOT / "results" / "raw" / "phase0" / "output_snr_orthonormal"
PHASE1 = REPO_ROOT / "results" / "raw" / "phase1"
PRECISIONS = ["int4_g128", "int4_per_channel", "int3_g128"]


def mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else float("nan")


def rank(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    for pos, i in enumerate(order):
        r[i] = pos + 1
    return r


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = rank(a), rank(b)
    n = len(a)
    if n < 3:
        return float("nan")
    ma, mb = mean(ra), mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


def main() -> None:
    # --- Phase 0: output SNR per adapter ---
    snr: dict[str, list[float]] = defaultdict(list)
    ranks: dict[str, int] = {}
    for f in PHASE0.glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            snr[r["adapter"]].append(r["snr_out_orthonormal"])
            ranks[r["adapter"]] = r["rank"]
    snr_mean = {a: mean(v) for a, v in snr.items()}

    # --- Phase 1: retention per adapter per precision ---
    rows: list[dict[str, Any]] = []
    for p in sorted(PHASE1.glob("*/records.jsonl")):
        rows += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()
                 if x.strip()]
    by: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[(r["adapter"], r["condition"], r["precision"])].append(r)

    behav = sorted({r["adapter"] for r in rows})
    retention: dict[str, dict[str, float]] = {}
    for a in behav:
        ref = mean([r["guesser_p_word_normalised"]
                    for r in by.get((a, "aligned_bf16", "bf16"), [])])
        retention[a] = {}
        for p in PRECISIONS:
            v = mean([r["guesser_p_word_normalised"]
                      for r in by.get((a, "aligned_quant", p), [])])
            retention[a][p] = v / ref if ref else float("nan")

    paired = [a for a in behav if a in snr_mean]
    print(f"adapters with BOTH Phase 0 output SNR and Phase 1 retention: {len(paired)}")

    print(f"\n{'=' * 96}")
    print("PREDICTOR vs OUTCOME")
    print("=" * 96)
    hdr = (f"{'word':>8} {'r':>4} {'SNR_out':>9} " +
           " ".join(f"{p:>17}" for p in PRECISIONS))
    print(hdr); print("-" * len(hdr))
    for a in sorted(paired, key=lambda x: snr_mean[x]):
        w = next(r["secret_word"] for r in rows if r["adapter"] == a)
        line = f"{w:>8} {ranks[a]:>4} {snr_mean[a]:>9.4f}"
        for p in PRECISIONS:
            line += f" {retention[a][p]:>17.1%}"
        print(line)

    s = [snr_mean[a] for a in paired]
    print(f"\n  PREDICTOR range: {min(s):.4f} to {max(s):.4f}  "
          f"= {(max(s) / min(s) - 1):.1%} spread")
    for p in PRECISIONS:
        o = [retention[a][p] for a in paired]
        print(f"  OUTCOME  range ({p:>16}): {min(o):.1%} to {max(o):.1%}  "
              f"= {(max(o) / min(o)):.1f}x spread")

    print(f"\n{'=' * 96}")
    print("SPEARMAN, output SNR vs retention")
    print("=" * 96)
    for p in PRECISIONS:
        o = [retention[a][p] for a in paired]
        print(f"  {p:>18}  rho = {spearman(s, o):>+6.3f}   (n={len(paired)})")

    print(f"\n{'=' * 96}")
    print("POWER")
    print("=" * 96)
    cv_pred = statistics.stdev(s) / mean(s) if len(s) > 1 else float("nan")
    print(f"  predictor coefficient of variation: {cv_pred:.4f}")
    for p in PRECISIONS:
        o = [retention[a][p] for a in paired]
        cv_out = statistics.stdev(o) / mean(o) if len(o) > 1 else float("nan")
        print(f"  outcome CV ({p:>16}): {cv_out:.4f}   "
              f"ratio outcome/predictor = {cv_out / cv_pred:.1f}x")


if __name__ == "__main__":
    main()
