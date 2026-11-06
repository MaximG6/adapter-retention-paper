"""Is the int3 retention spread a per-word effect, or sampling noise?

Six taboo adapters matched on rank, scaling, base model, recipe, and output SNR to
3% show int3 retention from 28.7% to 86.4%. Before attributing that to word
identity, it has to survive the null that it is noise.

Greedy decoding makes generation deterministic, so re-running with a different
seed reproduces the output exactly and measures nothing. The available nuisance
axis is which prompts were drawn: each adapter's retention is an average over 32
prompts, and that average has sampling error. Bootstrapping over prompts gives the
per-adapter interval, and if those intervals overlap heavily the between-word
spread is not resolved.

Usage:
    python analysis/word_vs_noise.py
"""

from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import bootstrap

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE1 = REPO_ROOT / "results" / "raw" / "phase1"
PRECISIONS = ["int4_g128", "int4_per_channel", "int3_g128"]


def mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else float("nan")


def boot_ratio(num: list[float], den: list[float]) -> tuple[float, float]:
    """Delegates to analysis/bootstrap.py."""
    return bootstrap.ratio_ci(num, den)


def clusters_for(cur: list[dict[str, Any]],
                 ref: list[dict[str, Any]]) -> list[tuple[str, list[float], list[float]]]:
    """Group the paired records into (stratum, quantized, bf16) by intent.

    The sampling unit is the INTENT, not the prompt. E.1's hint battery is 8 intents x 3
    paraphrases; measured, the ICC is 0.175-0.303 and the 32 prompts carry 23-26
    effective units (B.12), not the "roughly 16" this docstring used to assert, which is
    the ICC = 1 case. `prompt_kind` is the stratum: hint and
    adversarial are different instruments and each adversarial prompt is its own intent.
    """
    cur_by = defaultdict(list)
    ref_by = defaultdict(list)
    kind: dict[str, str] = {}
    for r in cur:
        cur_by[r["intent"]].append(r["guesser_p_word_normalised"])
        kind[r["intent"]] = r["prompt_kind"]
    for r in ref:
        ref_by[r["intent"]].append(r["guesser_p_word_normalised"])
        kind.setdefault(r["intent"], r["prompt_kind"])
    return [(kind[i], cur_by[i], ref_by[i]) for i in sorted(cur_by)
            if cur_by[i] and ref_by[i]]


def singletons_for(cur: list[dict[str, Any]],
                   ref: list[dict[str, Any]]) -> list[tuple[str, list[float], list[float]]]:
    """Same estimator, one PROMPT per cluster.

    This isolates the two corrections from each other. Moving from the published
    interval to the intent-clustered one changes two things at once -- the sampling unit
    and the pairing -- and they push in opposite directions, so a single before/after
    number would attribute the net effect to whichever one was named.
    """
    c = {r["prompt_id"]: r["guesser_p_word_normalised"] for r in cur}
    d = {r["prompt_id"]: r["guesser_p_word_normalised"] for r in ref}
    kind = {r["prompt_id"]: r["prompt_kind"] for r in cur}
    return [(kind[p], [c[p]], [d[p]]) for p in sorted(set(c) & set(d))]


def main() -> None:
    rows: list[dict[str, Any]] = []
    for p in sorted(PHASE1.glob("*/records.jsonl")):
        rows += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()
                 if x.strip()]
    by: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[(r["adapter"], r["condition"], r["precision"])].append(r)
    adapters = sorted({r["adapter"] for r in rows})
    words = {a: next(r["secret_word"] for r in rows if r["adapter"] == a)
             for a in adapters}

    for precision in PRECISIONS:
        print(f"\n{'=' * 92}")
        print(f"{precision}: per-adapter retention with 95% CI bootstrapped over prompts")
        print("=" * 92)
        hdr = (f"{'word':>8} {'ret.':>7} {'A prompts/unpaired':>22} "
               f"{'B prompts/paired':>22} {'C intents/paired':>22} {'C/A':>5}")
        print(hdr); print("-" * len(hdr))
        point: list[float] = []
        widths: list[float] = []
        est: dict[str, tuple[list[float], list[float]]] = {
            k: ([], []) for k in "ABC"}
        for a in adapters:
            ref_rows = by.get((a, "aligned_bf16", "bf16"), [])
            cur_rows = by.get((a, "aligned_quant", precision), [])
            ref = [r["guesser_p_word_normalised"] for r in ref_rows]
            cur = [r["guesser_p_word_normalised"] for r in cur_rows]
            pt = mean(cur) / mean(ref)
            ivs = {
                "A": boot_ratio(cur, ref),
                "B": bootstrap.cluster_ratio_ci(singletons_for(cur_rows, ref_rows)),
                "C": bootstrap.cluster_ratio_ci(clusters_for(cur_rows, ref_rows)),
            }
            for k, (lo, hi) in ivs.items():
                est[k][0].append(lo)
                est[k][1].append(hi)
            point.append(pt)
            widths.append(ivs["C"][1] - ivs["C"][0])
            wa = ivs["A"][1] - ivs["A"][0]
            print(f"{words[a]:>8} {pt:>7.1%} "
                  + " ".join(f"[{lo:>8.1%},{hi:>8.1%}]" for lo, hi in
                             (ivs["A"], ivs["B"], ivs["C"]))
                  + f" {(ivs['C'][1] - ivs['C'][0]) / wa:>5.2f}")

        between = max(point) - min(point)
        within = mean(widths)
        print("-" * len(hdr))
        print(f"  between-word spread (max-min point estimates): {between:>7.1%}")
        print(f"  mean within-adapter 95% CI width (C)         : {within:>7.1%}")
        print(f"  ratio between/within                         : {between / within:>7.2f}")

        # Do any two adapters have non-overlapping intervals? Reported under all three,
        # because A is what PG-2 published, C is what the design supports, and B shows
        # which of the two corrections moved the count.
        n_pairs = len(adapters) * (len(adapters) - 1) // 2
        for label, (lo_v, hi_v) in (("A prompts/unpaired", est["A"]),
                                    ("B prompts/paired  ", est["B"]),
                                    ("C intents/paired  ", est["C"])):
            pairs = [(words[adapters[i]], words[adapters[j]])
                     for i in range(len(adapters)) for j in range(i + 1, len(adapters))
                     if hi_v[i] < lo_v[j] or hi_v[j] < lo_v[i]]
            print(f"  separating pairs, {label}: {len(pairs)} of {n_pairs}"
                  + (f"   {pairs}" if pairs else ""))


if __name__ == "__main__":
    main()
