"""Is `u` uniform where the derivation needs it -- conditional on the delta crossing it?

Section 3.5 derives Equation 4 from `P(flip) = E[1[u < |delta|/s]]`, which needs
`F_u(t) = t` **at the `t` each weight presents**, not on average over all weights.

Two measurements already exist and neither is that conditional:

* EXP-042 / Section 4.1 measures `pearson(|delta|/s, u)` over the whole bin. A Pearson
  correlation over `u` in [0, 1] is dominated by the bulk and says almost nothing about
  the density of `u` in the lowest 1% -- which, at the `t ~ 0.011` the taboo adapters
  occupy, is where the entire prediction lives.
* EXP-045 / B.11 measures `F_u(t)` pooled over every weight. That is uniformity, checked
  locally in `t` but marginally in `|delta|`.

So uniformity is checked locally and independence is not. This closes that: it bins
weights by decile of `|delta|/s` and re-measures the low tail of `u` inside each bin.

Registered as P12 in `EXPERIMENTS.md` (EXP-052) before this file was run.

Usage:
    PYTHONPATH=src python scripts/local_independence.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ar.adapters import load_adapter_spec  # noqa: E402
from ar.device import require_cuda  # noqa: E402
from ar.manifest import build_manifest  # noqa: E402
from ar.quantsim import QuantConfig, compute_params  # noqa: E402
from ar.retention import lora_delta  # noqa: E402
from bin_position_uniformity import offsets  # noqa: E402
from measure_public_adapter import BASE_ALIASES, MODULE_PATH, RemoteTensorReader  # noqa: E402

OUT = REPO_ROOT / "results" / "raw" / "phase0" / "local_independence"

#: Where the taboo adapters sit. B.11's pooled measurement is quoted at this t.
PROBE_T = 0.011
N_DECILES = 10


def step_per_weight(w: torch.Tensor, cfg: QuantConfig) -> torch.Tensor:
    p = compute_params(w, cfg)
    return p.step_per_weight()


def true_flip(w: torch.Tensor, d: torch.Tensor, cfg: QuantConfig) -> torch.Tensor:
    """Per-weight indicator that the integer code actually changes, fixed grid.

    The grid comes from `W` alone and is applied to `W + delta`, which is `fixed_scale`
    -- the regime Equation 4 is derived under. This is the ground truth the two-sided
    `u` proxy approximates, and having both on one population is what decomposes the
    licensing correction into the step that carries it.
    """
    p = compute_params(w, cfg)
    step = p.step_per_weight()
    g = w.shape[1] if cfg.group_size == -1 else cfg.group_size
    zero = p.zero.repeat_interleave(g, dim=1)[:, : w.shape[1]]
    hi = 2 ** cfg.bits - 1
    q0 = torch.clamp(torch.round(w / step + zero), 0, hi)
    q1 = torch.clamp(torch.round((w + d) / step + zero), 0, hi)
    return (q0 != q1).float()


def by_decile(u: torch.Tensor, t: torch.Tensor,
              probe: float) -> list[dict[str, float]]:
    """Low tail, high tail and two-sided flip probability of `u`, per decile of `t`.

    Deciles are of `t = |delta|/s`, so each row asks: among the weights presenting this
    much delta, how much of the bin's lowest `probe` is occupied? Under independence
    every row reads the same, and reads `probe`.
    """
    order = torch.argsort(t)
    u_sorted, t_sorted = u[order], t[order]
    n = u_sorted.numel()
    rows: list[dict[str, float]] = []
    for d in range(N_DECILES):
        lo_i, hi_i = (n * d) // N_DECILES, (n * (d + 1)) // N_DECILES
        uu, tt = u_sorted[lo_i:hi_i], t_sorted[lo_i:hi_i]
        if uu.numel() == 0:
            continue
        own = float(tt.mean().item())
        rows.append({
            "decile": d + 1,
            "n": int(uu.numel()),
            "t_mean": own,
            "t_lo": float(tt[0].item()),
            "t_hi": float(tt[-1].item()),
            # At the COMMON probe: does the low tail of u depend on the decile?
            "F_lo_at_probe": float((uu < probe).float().mean().item()),
            "F_hi_at_probe": float((uu > 1.0 - probe).float().mean().item()),
            # At the decile's OWN t: the quantity Equation 4 integrates.
            "F_lo_at_own_t": float((uu < own).float().mean().item()),
            "F_hi_at_own_t": float((uu > 1.0 - own).float().mean().item()),
            "mean_u": float(uu.mean().item()),
        })
    for r in rows:
        r["flip_at_probe"] = (r["F_lo_at_probe"] + r["F_hi_at_probe"]) / 2
        r["flip_at_own_t"] = (r["F_lo_at_own_t"] + r["F_hi_at_own_t"]) / 2
    return rows


def spearman(xs: list[float], ys: list[float]) -> float:
    def rank(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", default=(
        "adamkarvonen/Qwen3-8B-taboo-smile_50_mix,"
        "Kurapika993/llama-3.1-8b-responsible-ai-safety-lora"))
    ap.add_argument("--layers", default="0,12,24")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    args = ap.parse_args()

    device = require_cuda((12, 0))
    cfg = QuantConfig(bits=args.bits, group_size=args.group_size, scheme="asymmetric")
    layers = [int(x) for x in args.layers.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    started = time.time()

    from peft import load_peft_weights

    for adapter in args.adapters.split(","):
        spec = load_adapter_spec(adapter)
        base = BASE_ALIASES.get(spec.base_model, spec.base_model)
        print(f"\n{adapter}\n  base {base}  r={spec.rank} alpha={spec.alpha}")
        sd = load_peft_weights(adapter)
        reader = RemoteTensorReader(base)
        for layer in layers:
            for module, parent in MODULE_PATH.items():
                prefix = (f"base_model.model.model.layers.{layer}.{parent}.{module}")
                a_key, b_key = f"{prefix}.lora_A.weight", f"{prefix}.lora_B.weight"
                if a_key not in sd or b_key not in sd:
                    continue
                w = reader.read(
                    f"model.layers.{layer}.{parent}.{module}.weight"
                ).to(device=device, dtype=torch.float32)
                d = lora_delta(sd[a_key].to(device), sd[b_key].to(device),
                               alpha=spec.alpha, rank=spec.rank,
                               use_rslora=spec.use_rslora)
                if d.shape != w.shape:
                    raise RuntimeError(f"delta {tuple(d.shape)} != base {tuple(w.shape)}")
                u = offsets(w, cfg).flatten()
                t = (d.abs() / step_per_weight(w, cfg)).flatten()
                f = true_flip(w, d, cfg).flatten()
                rows = by_decile(u, t, PROBE_T)
                order = torch.argsort(t)
                f_sorted = f[order]
                n_all = f.numel()
                for r in rows:
                    lo_i = (n_all * (r["decile"] - 1)) // N_DECILES
                    hi_i = (n_all * r["decile"]) // N_DECILES
                    r["true_flip"] = float(f_sorted[lo_i:hi_i].mean().item())
                records.append({
                    "adapter": adapter, "base_model": base, "layer": layer,
                    "module": module, "bits": args.bits,
                    "group_size": args.group_size, "scheme": "asymmetric",
                    "probe_t": PROBE_T, "n_weights": int(u.numel()),
                    "t_mean_all": float(t.mean().item()),
                    "predicted_flip_rate": float(torch.clamp(t, max=1.0).mean().item()),
                    "true_flip_rate": float(f.mean().item()),
                    "deciles": rows,
                })
                del w, d, u, t, f, f_sorted
                torch.cuda.empty_cache()
            print(f"  layer {layer:>2}: "
                  f"{len([r for r in records if r['layer'] == layer])} modules")

    path = OUT / "records.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    (OUT / "manifest.json").write_text(json.dumps(build_manifest(
        device=device, seeds={},
        extra={"adapters": args.adapters.split(","), "layers": layers,
               "bits": args.bits, "group_size": args.group_size,
               "probe_t": PROBE_T, "n_deciles": N_DECILES,
               "n_records": len(records),
               "wall_time_s": time.time() - started}), indent=2))
    print(f"\nwrote {path.relative_to(REPO_ROOT)} ({len(records)} records)")

    n = len(records)
    pooled = [sum(r["deciles"][d]["flip_at_probe"] for r in records) / n
              for d in range(N_DECILES)]
    grand = sum(pooled) / len(pooled)
    print(f"\nP12: F_u at the common probe t = {PROBE_T}, per decile of |delta|/s")
    print(f"{'dec':>4} {'t range':>21} {'F_lo':>9} {'F_hi':>9} {'two-sided':>11} "
          f"{'/uniform':>9} {'/pooled':>9}")
    for d in range(N_DECILES):
        lo = sum(r["deciles"][d]["F_lo_at_probe"] for r in records) / n
        hi = sum(r["deciles"][d]["F_hi_at_probe"] for r in records) / n
        tl = sum(r["deciles"][d]["t_lo"] for r in records) / n
        th = sum(r["deciles"][d]["t_hi"] for r in records) / n
        print(f"{d + 1:>4} {tl:>9.5f}-{th:<11.5f} {lo:>9.5f} {hi:>9.5f} "
              f"{pooled[d]:>11.5f} {pooled[d] / PROBE_T:>9.4f} "
              f"{pooled[d] / grand:>9.4f}")
    worst = max(abs(p / grand - 1) for p in pooled)
    rho = spearman(list(range(1, N_DECILES + 1)), pooled)
    print(f"\n  grand mean two-sided at probe : {grand:.6f}  "
          f"(uniform {PROBE_T}, ratio {grand / PROBE_T:.4f})")
    print(f"  worst decile departure        : {worst:.4%}   "
          f"(P12.1 bound 2%)  -> {'PASS' if worst <= 0.02 else 'FAIL'}")
    print(f"  decile-index Spearman         : {rho:+.4f}   "
          f"(P12.2 bound 0.5)  -> {'PASS' if abs(rho) <= 0.5 else 'FAIL'}")

    print(f"\n  each decile at its OWN mean t (what Equation 4 integrates)")
    print(f"{'dec':>4} {'own t':>10} {'measured':>11} {'predicted':>11} {'ratio':>9}")
    worst_own = 0.0
    for d in range(N_DECILES):
        own = sum(r["deciles"][d]["t_mean"] for r in records) / n
        got = sum(r["deciles"][d]["flip_at_own_t"] for r in records) / n
        pred = min(own, 1.0)
        worst_own = max(worst_own, abs(got / pred - 1) if pred else 0.0)
        print(f"{d + 1:>4} {own:>10.5f} {got:>11.5f} {pred:>11.5f} "
              f"{got / pred if pred else float('nan'):>9.4f}")
    print(f"\n  worst departure from min(t,1) : {worst_own:.4%}   "
          f"(P12.3 bound 5%)  -> {'PASS' if worst_own <= 0.05 else 'FAIL'}")

    # ---- the licensing correction, decomposed on ONE population ----
    # B.11 says the non-uniformity of u predicts a 1.3-1.5% over-prediction; B.2 measures
    # 0.1-0.2%. Those are different populations AND different statistics: B.11's is the
    # two-sided u proxy, B.2's is the actual integer code flip. Both are computed here, on
    # the same 42 module-instances, so the gap is attributed rather than argued about.
    pred = sum(r["predicted_flip_rate"] for r in records) / n
    proxy_num = sum(sum(r["deciles"][d]["flip_at_own_t"] for d in range(N_DECILES))
                    for r in records) / (n * N_DECILES)
    true_r = sum(r["true_flip_rate"] for r in records) / n
    print("\n  the licensing correction, decomposed on one population "
          f"({n} module-instances, both base models):")
    print(f"    closed form  mean(min(t,1))       : {pred:.6f}")
    print(f"    two-sided u proxy, decile-integrated: {proxy_num:.6f}  "
          f"ratio to closed form {proxy_num / pred:.4f}")
    print(f"    actual integer code flip          : {true_r:.6f}  "
          f"ratio to closed form {true_r / pred:.4f}")
    print(f"    proxy - actual                    : "
          f"{(proxy_num - true_r) / true_r:+.4%}  <- what the 50/50 two-sided step costs")
    print(f"\n{'dec':>4} {'own t':>10} {'proxy':>11} {'true flip':>11} "
          f"{'proxy/true':>11} {'true/min(t,1)':>14}")
    for d in range(N_DECILES):
        own = sum(r["deciles"][d]["t_mean"] for r in records) / n
        px = sum(r["deciles"][d]["flip_at_own_t"] for r in records) / n
        tr = sum(r["deciles"][d]["true_flip"] for r in records) / n
        print(f"{d + 1:>4} {own:>10.5f} {px:>11.5f} {tr:>11.5f} "
              f"{px / tr if tr else float('nan'):>11.4f} "
              f"{tr / min(own, 1.0) if own else float('nan'):>14.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
