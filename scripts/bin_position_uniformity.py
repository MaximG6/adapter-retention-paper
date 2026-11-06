"""Is the within-bin offset u uniform in the lowest few percent of the bin?

Section 3.5 derives the channel model from two assumptions:

    w = s(k + u),  and for |d| < s the flip indicator is 1[u < |d|/s],
    so E[1[u < t]] = F_u(t), and the model needs F_u(t) = t.

Section 4.1 measures the INDEPENDENCE of u and d. It does not measure the UNIFORMITY of
u, and those are different assumptions. At t ~ 0.011 -- where the taboo adapters sit --
the entire prediction rests on the density of u in the lowest 1% of the bin, and there is
a structural reason to expect trouble there: under the asymmetric scheme each group's
extrema map exactly to codes 0 and 2^b-1 by construction, so the offsets at the ends of
every group are not free.

This measures F_u(t) directly, on the same base weights the paper quantizes, and compares
it against the uniform line at the values of t the adapters actually occupy.

Usage:
    PYTHONPATH=src python scripts/bin_position_uniformity.py
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

from ar.device import require_cuda  # noqa: E402
from ar.manifest import build_manifest  # noqa: E402
from ar.quantsim import QuantConfig, compute_params  # noqa: E402
from measure_public_adapter import BASE_ALIASES, MODULE_PATH  # noqa: E402

OUT = REPO_ROOT / "results" / "raw" / "phase0" / "bin_position"

#: The quantiles the paper's adapters actually occupy. mean|D|/s runs 0.011 (taboo) to
#: 0.149 (rank-128 rsLoRA), so the lowest 1% of the bin is where the prediction lives.
PROBE_T = [0.001, 0.002, 0.005, 0.008, 0.011, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75]


def offsets(w: torch.Tensor, cfg: QuantConfig) -> torch.Tensor:
    """u = frac(w/s + z), the position of each weight within its quantization bin.

    Equation 2 rounds g/s + z to the nearest integer, so the code changes exactly when
    the fractional part crosses 0.5. Shifting by 0.5 puts the boundary at u = 0 and makes
    u the DISTANCE to the boundary, which is the quantity 1[u < |d|/s] is about.
    """
    p = compute_params(w, cfg)
    step = p.step_per_weight()
    zero = p.zero.repeat_interleave(
        w.shape[1] if cfg.group_size == -1 else cfg.group_size, dim=1)[:, : w.shape[1]]
    x = w.to(torch.float32) / step + zero
    return (x + 0.5).frac().abs()


def _pinning_controls(w: torch.Tensor, u: torch.Tensor, zero: torch.Tensor,
                      cfg: QuantConfig) -> dict[str, float]:
    """Where do group extrema actually sit, and is the exact-zero mass structural?

    An earlier reading of this measurement attributed the exact-zero mass to Equation 2
    pinning each group's extrema, which predicts 2/group_size = 1.56% at g=128 against a
    measured 0.20%. Three controls settle it instead of arguing about it:

    * `u_at_group_min` / `u_at_group_max` -- where the pinned weights land. Equation 2
      rounds `z`, so the extrema map onto the CENTRES of codes 0 and 2^b-1, which is
      u = 0.5, the position furthest from any boundary rather than nearest.
    * `frac_extrema_among_zero` -- how much of the exact-zero mass is extrema at all.
    * `frac_exactly_zero_jittered` -- the mass surviving a perturbation of 1e-4 steps,
      four orders of magnitude below bf16's own resolution inside a bin. A structural
      pinning survives it; a floating-point coincidence over discrete-valued input
      does not.
    """
    g = w.shape[1] if cfg.group_size == -1 else cfg.group_size
    n_out, n_in = w.shape
    n_full = (n_in // g) * g
    blk = w[:, :n_full].reshape(n_out, -1, g)
    ub = u.reshape(n_out, n_in)[:, :n_full].reshape(n_out, -1, g)
    imin = blk.argmin(-1, keepdim=True)
    imax = blk.argmax(-1, keepdim=True)
    ext = torch.zeros_like(ub, dtype=torch.bool)
    ext.scatter_(-1, imin, True)
    ext.scatter_(-1, imax, True)
    z_blk = zero.reshape(n_out, n_in)[:, :n_full].reshape(n_out, -1, g)
    n_zero = int(z_blk.sum().item())

    step = compute_params(w, cfg).step_per_weight()
    gen = torch.Generator(device=w.device).manual_seed(0)
    jitter = (torch.rand(w.shape, device=w.device, generator=gen) - 0.5) * step * 1e-4
    # A MEAN of 0.494 is exactly what uniformity predicts, so on its own it licenses
    # "unconstrained", not "pinned to the centre" -- the replacement account needs the
    # dispersion as much as the refuted one did. Reported rather than argued.
    u_min = torch.gather(ub, -1, imin).flatten()
    u_max = torch.gather(ub, -1, imax).flatten()
    return {
        "u_at_group_min": u_min.mean().item(),
        "u_at_group_max": u_max.mean().item(),
        "u_at_group_min_sd": u_min.std().item(),
        "u_at_group_max_sd": u_max.std().item(),
        # Uniform on [0,1) has SD 1/sqrt(12) = 0.2887; a pinned population has ~0.
        "u_extrema_sd_over_uniform": (
            float(torch.cat([u_min, u_max]).std().item()) / (1 / 12) ** 0.5),
        "u_extrema_iqr": float(
            (torch.quantile(torch.cat([u_min, u_max]).float(), 0.75)
             - torch.quantile(torch.cat([u_min, u_max]).float(), 0.25)).item()),
        "frac_extrema_among_zero": (
            float((z_blk & ext).sum().item()) / n_zero if n_zero else float("nan")),
        "frac_weights_that_are_extrema": 2.0 / g,
        "frac_exactly_zero_jittered": (
            offsets(w + jitter, cfg) == 0.0).float().mean().item(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", default=(
        "adamkarvonen/Qwen3-8B-taboo-smile_50_mix,"
        "Kurapika993/llama-3.1-8b-responsible-ai-safety-lora"))
    ap.add_argument("--layers", default="0,12,24")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    args = ap.parse_args()

    from ar.adapters import load_adapter_spec
    from transformers import AutoModelForCausalLM

    device = require_cuda((12, 0))
    cfg = QuantConfig(bits=args.bits, group_size=args.group_size, scheme="asymmetric")
    layers = [int(x) for x in args.layers.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    started = time.time()

    for adapter in args.adapters.split(","):
        spec = load_adapter_spec(adapter)
        base = BASE_ALIASES.get(spec.base_model, spec.base_model)
        print(f"\n{adapter}\n  base {base}")
        model = AutoModelForCausalLM.from_pretrained(
            base, dtype=torch.bfloat16, device_map={"": device})
        model.eval()
        named = dict(model.named_modules())
        with torch.no_grad():
            for layer in layers:
                for module, parent in MODULE_PATH.items():
                    name = f"model.layers.{layer}.{parent}.{module}"
                    mod = named.get(name)
                    if mod is None:
                        continue
                    w = mod.weight.detach().to(device, torch.float32)
                    u = offsets(w, cfg).flatten()
                    n = u.numel()
                    # BOTH tails. A weight moves to the code above when the distance to
                    # the upper boundary is under |d|/s, and to the code below when the
                    # distance to the lower one is. Only the lower tail is F_u(t); the
                    # flip probability the model needs is the average of the two, and
                    # measuring one alone can show an excess that the other cancels.
                    lo = {f"{t}": (u < t).float().mean().item() for t in PROBE_T}
                    hi = {f"{t}": (u > 1.0 - t).float().mean().item() for t in PROBE_T}
                    zero = u == 0.0
                    records.append({
                        "adapter": adapter, "base_model": base, "layer": layer,
                        "module": module, "bits": args.bits,
                        "group_size": args.group_size, "scheme": "asymmetric",
                        "n_weights": n, "ecdf": lo, "ecdf_upper": hi,
                        "flip_prob": {k: (lo[k] + hi[k]) / 2 for k in lo},
                        "mean_u": u.mean().item(),
                        "frac_exactly_zero": zero.float().mean().item(),
                        **_pinning_controls(w, u, zero, cfg),
                    })
                    del w, u, zero
                    torch.cuda.empty_cache()
                print(f"  layer {layer:>2}: {len([r for r in records if r['layer'] == layer])} modules")
        del model
        torch.cuda.empty_cache()

    path = OUT / "records.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    (OUT / "manifest.json").write_text(json.dumps(build_manifest(
        device=device, seeds={},
        extra={"adapters": args.adapters.split(","), "layers": layers,
               "bits": args.bits, "group_size": args.group_size,
               "probe_t": PROBE_T, "n_records": len(records),
               "wall_time_s": time.time() - started}), indent=2))

    print(f"\nwrote {path.relative_to(REPO_ROOT)} ({len(records)} records)\n")
    print(f"{'t':>8} {'lower tail':>12} {'upper tail':>12} {'mean (=P flip)':>16} "
          f"{'uniform':>9} {'ratio':>8}")
    for t in PROBE_T:
        lo = sum(r["ecdf"][f"{t}"] for r in records) / len(records)
        hi = sum(r["ecdf_upper"][f"{t}"] for r in records) / len(records)
        pf = (lo + hi) / 2
        print(f"{t:>8.3f} {lo:>12.5f} {hi:>12.5f} {pf:>16.5f} {t:>9.3f} "
              f"{pf / t:>8.3f}")
    n = len(records)
    print(f"\nfraction of weights with u exactly 0: "
          f"{sum(r['frac_exactly_zero'] for r in records) / n:.6f}")
    print("\nis that Equation 2 pinning the group extrema?")
    print(f"  u at group min / max            : "
          f"{sum(r['u_at_group_min'] for r in records) / n:.4f} / "
          f"{sum(r['u_at_group_max'] for r in records) / n:.4f}   (boundary=0, centre=0.5)")
    print(f"  SD of u at the extrema          : "
          f"{sum(r['u_at_group_min_sd'] for r in records) / n:.4f} / "
          f"{sum(r['u_at_group_max_sd'] for r in records) / n:.4f}   "
          f"(uniform = {(1 / 12) ** 0.5:.4f}); "
          f"ratio to uniform "
          f"{sum(r['u_extrema_sd_over_uniform'] for r in records) / n:.3f}, "
          f"IQR {sum(r['u_extrema_iqr'] for r in records) / n:.3f} "
          f"(uniform 0.500)")
    print(f"  extrema as a fraction of weights: "
          f"{records[0]['frac_weights_that_are_extrema']:.6f}  "
          "<- what the pinning account predicts")
    print(f"  extrema among the u==0 weights  : "
          f"{sum(r['frac_extrema_among_zero'] for r in records) / n:.4f}")
    print(f"  u==0 surviving a 1e-4*s jitter  : "
          f"{sum(r['frac_exactly_zero_jittered'] for r in records) / n:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
