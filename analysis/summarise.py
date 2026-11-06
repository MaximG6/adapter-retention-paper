"""Aggregate every Phase 0 raw record into the GATE 0 tables.

Re-derives everything from results/raw/**/records.jsonl. Nothing is read from a
summary file, so any table here can be regenerated from the raw JSONL alone.

Usage:
    python analysis/summarise.py
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "results" / "raw" / "phase0"


def load(pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(RAW.glob(pattern)):
        run = p.parent.name
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                rec["_run"] = run  # which run shape produced this record
                rows.append(rec)
    return rows


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def bootstrap_ci(
    xs: list[float], n_boot: int = 5000, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap over the sampling unit passed in (layers, or adapters).

    The caller selects the sampling unit (layers or adapters) through `xs`.
    """
    import random

    if len(xs) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(
        mean([xs[rng.randrange(len(xs))] for _ in range(len(xs))])
        for _ in range(n_boot)
    )
    return (means[int(0.025 * n_boot)], means[int(0.975 * n_boot)])


def paired_on_schemes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only (adapter, layer, module) cells measured under every scheme.

    Runs differ in which schemes they cover: the 36-layer depth run is
    asymmetric-only. Pooling those raw would weight asymmetric toward whichever
    adapter happened to get the deep run and can invert the apparent ordering
    between conventions. Any convention comparison must be paired.
    """
    schemes = {r["scheme"] for r in rows}
    seen: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for r in rows:
        seen[(r["adapter"], r["layer"], r["module"])].add(r["scheme"])
    complete = {k for k, v in seen.items() if v == schemes}
    return [r for r in rows if (r["adapter"], r["layer"], r["module"]) in complete]


def one_run_per_adapter(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate adapters measured by more than one run, keeping the deepest.

    Without this an adapter with both a 4-layer and a 36-layer run contributes
    twice to any pooled mean, at ten times the weight of its peers.
    """
    depth: dict[str, int] = defaultdict(int)
    for r in rows:
        key = (r["adapter"], r.get("_run", ""))
        depth[r["adapter"]] = max(depth[r["adapter"]], 0)
    by_adapter_run: dict[tuple[str, str], set[int]] = defaultdict(set)
    for r in rows:
        by_adapter_run[(r["adapter"], r["_run"])].add(r["layer"])
    best: dict[str, str] = {}
    for (adapter, run), layers in by_adapter_run.items():
        if adapter not in best or len(layers) > len(
            by_adapter_run[(adapter, best[adapter])]
        ):
            best[adapter] = run
    return [r for r in rows if r["_run"] == best[r["adapter"]]]


def main() -> None:
    rows = load("public_adapter/*/*/records.jsonl")
    if not rows:
        print("No public-adapter records found.")
        return

    fixed_asym = one_run_per_adapter(
        [r for r in rows if r["scheme"] == "asymmetric" and r["regime"] == "fixed_scale"]
    )

    # ---------------- Cross-adapter headline ----------------
    print("=" * 108)
    print("GATE 0 HEADLINE: weight-space retention by adapter, INT4 g128 asymmetric, fixed_scale")
    print("=" * 108)
    hdr = (
        f"{'adapter':>46} {'base':>10} {'r':>5} {'scale':>6} {'L':>3} {'cosine':>8} "
        f"{'95% CI':>16} {'bitflip':>8} {'rel_err':>8}"
    )
    print(hdr)
    print("-" * len(hdr))

    by_adapter: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in fixed_asym:
        by_adapter[r["adapter"]].append(r)

    for adapter, rs in sorted(
        by_adapter.items(), key=lambda kv: mean([r["cosine"] for r in kv[1]])
    ):
        # Bootstrap over LAYERS: the layer is the sampling unit within an adapter.
        per_layer = defaultdict(list)
        for r in rs:
            per_layer[r["layer"]].append(r["cosine"])
        layer_means = [mean(v) for v in per_layer.values()]
        lo, hi = bootstrap_ci(layer_means)
        short = adapter if len(adapter) <= 46 else "..." + adapter[-43:]
        base_short = rs[0]["base_model"].split("/")[-1][:10]
        print(
            f"{short:>46} {base_short:>10} {rs[0]['rank']:>5} "
            f"{rs[0].get('effective_scaling', rs[0]['alpha_over_rank']):>6.3g} {len(per_layer):>3} "
            f"{mean([r['cosine'] for r in rs]):>8.4f} "
            f"[{lo:>6.4f},{hi:>6.4f}] "
            f"{mean([r['code_flip_rate'] for r in rs]):>8.4f} "
            f"{mean([r['relative_error'] for r in rs]):>8.3f}"
        )

    # ---------------- Depth profile ----------------
    deep = [r for r in fixed_asym if len({x["layer"] for x in by_adapter[r["adapter"]]}) > 10]
    if deep:
        adapter = deep[0]["adapter"]
        print(f"\n{'=' * 108}")
        print(f"DEPTH PROFILE (all layers): {adapter}")
        print("=" * 108)
        per_layer = defaultdict(list)
        for r in deep:
            per_layer[r["layer"]].append(r["cosine"])
        layers = sorted(per_layer)
        n_show = 12
        stride = max(1, len(layers) // n_show)
        print(f"{'layer':>7} {'cosine':>9}   {'':<40}")
        lo_v = min(mean(per_layer[x]) for x in layers)
        hi_v = max(mean(per_layer[x]) for x in layers)
        for x in layers[::stride]:
            v = mean(per_layer[x])
            bar = "#" * int(40 * (v - lo_v) / max(hi_v - lo_v, 1e-9))
            print(f"{x:>7} {v:>9.4f}   {bar:<40}")
        first_q = [mean(per_layer[x]) for x in layers[: len(layers) // 4]]
        last_q = [mean(per_layer[x]) for x in layers[-len(layers) // 4 :]]
        print(
            f"\n  first quartile of layers: {mean(first_q):.4f}   "
            f"last quartile: {mean(last_q):.4f}   ratio {mean(last_q) / mean(first_q):.3f}"
        )
        lo, hi = bootstrap_ci([mean(per_layer[x]) for x in layers])
        print(f"  all-layer mean {mean([mean(per_layer[x]) for x in layers]):.4f}  95% CI over layers [{lo:.4f}, {hi:.4f}]")

    # ---------------- Module profile ----------------
    print(f"\n{'=' * 108}")
    print("MODULE PROFILE, pooled over adapters and layers")
    print("=" * 108)
    hdr = f"{'module':>12} {'cosine':>9} {'bitflip':>9} {'rel_err':>9} {'median |d|/s':>14} {'n':>5}"
    print(hdr)
    print("-" * len(hdr))
    by_mod: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in fixed_asym:
        by_mod[r["module"]].append(r)
    for module, rs in sorted(by_mod.items(), key=lambda kv: mean([r["cosine"] for r in kv[1]])):
        print(
            f"{module:>12} {mean([r['cosine'] for r in rs]):>9.4f} "
            f"{mean([r['code_flip_rate'] for r in rs]):>9.4f} "
            f"{mean([r['relative_error'] for r in rs]):>9.3f} "
            f"{mean([r['step_ratio_quantiles']['p50'] for r in rs]) / 2:>14.5f} {len(rs):>5}"
        )

    # ---------------- Channel-model check on real adapters ----------------
    print(f"\n{'=' * 108}")
    print("CHANNEL MODEL vs REAL ADAPTERS (fixed_scale, asymmetric)")
    print("=" * 108)
    hdr = f"{'adapter':>46} {'flip meas':>10} {'flip pred':>10} {'ratio':>7} {'projection':>11}"
    print(hdr)
    print("-" * len(hdr))
    for adapter, rs in sorted(by_adapter.items()):
        meas = mean([r["code_flip_rate"] for r in rs])
        pred = mean([r["predicted_flip_rate"] for r in rs])
        short = adapter if len(adapter) <= 46 else "..." + adapter[-43:]
        print(
            f"{short:>46} {meas:>10.5f} {pred:>10.5f} {meas / pred:>7.3f} "
            f"{mean([r['projection_coefficient'] for r in rs]):>11.4f}"
        )

    # ---------------- Regime and convention ----------------
    print(f"\n{'=' * 108}")
    print("SCALE REGIME (asymmetric, pooled)")
    print("=" * 108)
    hdr = f"{'regime':>16} {'cosine':>9} {'code flips':>11} {'value changes':>14}"
    print(hdr)
    print("-" * len(hdr))
    regime_rows = one_run_per_adapter([r for r in rows if r["scheme"] == "asymmetric"])
    for regime in ("fixed_scale", "adaptive_scale"):
        rs = [r for r in regime_rows if r["regime"] == regime]
        print(
            f"{regime:>16} {mean([r['cosine'] for r in rs]):>9.4f} "
            f"{mean([r['code_flip_rate'] for r in rs]):>11.4f} "
            f"{mean([r['value_change_rate'] for r in rs]):>14.4f}"
        )
    gs = [r["grid_shift_fraction"] for r in fixed_asym]
    ss = [r["scale_shift_fraction"] for r in fixed_asym]
    print(f"\n  grid_shift_fraction  {mean(gs):.4f}")
    print(f"  scale_shift_fraction {mean(ss):.4f}")

    schemes = {r["scheme"] for r in rows}
    if len(schemes) > 1:
        paired = paired_on_schemes([r for r in rows if r["regime"] == "fixed_scale"])
        print(f"\n{'=' * 108}")
        print(
            "CONVENTION (fixed_scale, PAIRED on identical adapter/layer/module cells)"
        )
        print("=" * 108)
        cells = len({(r["adapter"], r["layer"], r["module"]) for r in paired})
        print(f"  {cells} cells present under all {len(schemes)} schemes\n")
        hdr = f"{'scheme':>17} {'cosine':>9} {'bitflip':>9} {'rel_err':>9}"
        print(hdr)
        print("-" * len(hdr))
        base_cos = None
        for scheme in sorted(schemes):
            rs = [r for r in paired if r["scheme"] == scheme]
            c = mean([r["cosine"] for r in rs])
            if scheme == "asymmetric":
                base_cos = c
            print(
                f"{scheme:>17} {c:>9.4f} "
                f"{mean([r['code_flip_rate'] for r in rs]):>9.4f} "
                f"{mean([r['relative_error'] for r in rs]):>9.3f}"
            )
        if base_cos:
            spread = max(
                abs(mean([r["cosine"] for r in paired if r["scheme"] == s]) - base_cos)
                for s in schemes
            )
            print(f"\n  max deviation from asymmetric: {spread / base_cos:.1%}")

    # ---------------- Tail shape: trained vs synthetic ----------------
    syn = load("synthetic/records.jsonl")
    if syn:
        dose = [r for r in syn if r.get("experiment") == "dose_response"]
        if dose:
            print(f"\n{'=' * 108}")
            print("TAIL SHAPE mean(d^2)/mean|d|^2  (Gaussian reference pi/2 = 1.5708)")
            print("=" * 108)
            print(f"  synthetic (iid factors): {mean([r['tail_shape'] for r in dose]):.4f}")


if __name__ == "__main__":
    main()

