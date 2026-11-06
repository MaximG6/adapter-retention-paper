"""Decisive test of the sqrt(d_in/r) amplification law, with rank as the only variable.

The six-adapter comparison in EXP-009 confounded rank with adapter identity: two
adapters agreed within 17% and two disagreed by 2x, and the split tracked adapter
as much as rank. That is the same confound that defeated the weight-space rank
law, so it cannot demote the amplification law either.

Here one adapter's delta is SVD-truncated to r = 4, 8, 16, 32 and rescaled to its
ORIGINAL Frobenius norm, so base weights, training, and magnitude are all held
fixed and rank is the only thing that moves.

Two further refinements over EXP-009:

* Per module, not pooled. sqrt(d_in/r) differs by sqrt(3) between attention
  (d_in=4096) and down_proj (d_in=12288), so pooling blurs the factor under test.

* Amplification is decomposed. For a probe x, define the concentration of a
  matrix M as

      conc(M, x) = ||Mx||^2 / ( ||M||_F^2 * ||x||^2 / d_in )

  which is 1 for isotropic x and rises as x aligns with M's row space. Then
  amplification = sqrt( conc(delta, x) / conc(E, x) ). Reporting both halves
  separates "signal fails to concentrate" from "error fails to spread".

  The error half matters specifically: flip probability is |delta|/s, so per-weight
  error variance is proportional to |delta| and the error inherits the adapter's
  magnitude profile rather than being isotropic. If conc(E, x) > 1 on subspace
  probes, the law needs a correction term rather than a refutation.

Probe construction is itself under test. EXP-009 drew subspace probes as
coef @ A, whose covariance is A^T A -- that over-weights A's dominant singular
directions and is NOT uniform on the row space. Both are measured here: the
A-weighted probe as used before, and an orthonormal probe built from the right
singular vectors, which is the fair test of the law.

Usage:
    python scripts/amplification_svd_test.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from peft import load_peft_weights

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ar.device import require_cuda  # noqa: E402
from ar.manifest import build_manifest  # noqa: E402
from ar.quantsim import QuantConfig, apply_params, compute_params  # noqa: E402
from ar.retention import lora_delta  # noqa: E402
from measure_public_adapter import RemoteTensorReader  # noqa: E402

ADAPTER = "adamkarvonen/Qwen3-8B-taboo-smile_50_mix"
BASE = "Qwen/Qwen3-8B"
MODULES = (("q_proj", "self_attn"), ("gate_proj", "mlp"), ("down_proj", "mlp"))
LAYERS = (12, 23)
TRUNC_RANKS = (4, 8, 16, 32)
N_PROBE = 8192
CFG = QuantConfig(bits=4, group_size=128, scheme="asymmetric")
OUT_DIR = REPO_ROOT / "results" / "raw" / "phase0" / "amplification"


def concentration(m: torch.Tensor, x: torch.Tensor) -> float:
    """||Mx||^2 / (||M||_F^2 * ||x||^2 / d_in). Equals 1 for isotropic x."""
    d_in = m.shape[1]
    num = (x @ m.T).pow(2).sum()
    den = m.pow(2).sum() * x.pow(2).sum() / d_in
    return (num / den).item()


def main() -> int:
    device = require_cuda((12, 0))
    print(f"device: {device}")
    acfg = json.load(open(hf_hub_download(ADAPTER, "adapter_config.json")))
    rank, alpha = int(acfg["r"]), float(acfg["lora_alpha"])
    # peft scales by alpha/sqrt(r) under rsLoRA. Read, never assume (EXP-011).
    use_rslora = bool(acfg.get("use_rslora", False))
    print(f"adapter: {ADAPTER}  r={rank} alpha={alpha:g}\n")

    sd = load_peft_weights(ADAPTER)
    reader = RemoteTensorReader(BASE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    started = time.time()

    for module, parent in MODULES:
        print("=" * 112)
        print(f"MODULE {module}")
        print("=" * 112)
        hdr = (
            f"{'layer':>6} {'r':>4} {'d_in':>6} {'SNR_w':>8} "
            f"{'conc_d ortho':>13} {'conc_E ortho':>13} {'amp ortho':>10} "
            f"{'sqrt(d/r)':>10} {'ratio':>7} {'amp A-wtd':>10} {'amp gen':>8}"
        )
        print(hdr)
        print("-" * len(hdr))

        for layer in LAYERS:
            pre = f"base_model.model.model.layers.{layer}.{parent}.{module}"
            a = sd[f"{pre}.lora_A.weight"].to(device).float()
            b = sd[f"{pre}.lora_B.weight"].to(device).float()
            w = reader.read(
                f"model.layers.{layer}.{parent}.{module}.weight"
            ).to(device, torch.float32)
            d_full = lora_delta(a, b, alpha=alpha, rank=rank, use_rslora=use_rslora)
            target_norm = torch.linalg.norm(d_full)
            d_in = w.shape[1]
            params = compute_params(w, CFG)
            q_base = apply_params(w, params, CFG).dequant

            u, s, vh = torch.linalg.svd(d_full, full_matrices=False)

            for r_t in TRUNC_RANKS:
                # Truncate, then rescale to the ORIGINAL Frobenius norm so that
                # magnitude is held constant and rank is the only variable.
                d_t = (u[:, :r_t] * s[:r_t]) @ vh[:r_t]
                d_t = d_t * (target_norm / torch.linalg.norm(d_t))

                d_eff = apply_params(w + d_t, params, CFG).dequant - q_base
                err = d_eff - d_t
                snr_w = (torch.linalg.norm(d_t) / torch.linalg.norm(err)).item()

                v_r = vh[:r_t]                       # orthonormal rows
                coef = torch.randn(N_PROBE, r_t, device=device)
                x_ortho = coef @ v_r                 # uniform on the row space
                x_awtd = coef @ (v_r * s[:r_t, None])  # weighted, as in EXP-009
                x_gen = torch.randn(N_PROBE, d_in, device=device)

                out: dict[str, float] = {}
                for tag, x in (
                    ("ortho", x_ortho), ("awtd", x_awtd), ("gen", x_gen)
                ):
                    cd = concentration(d_t, x)
                    ce = concentration(err, x)
                    out[f"conc_delta_{tag}"] = cd
                    out[f"conc_err_{tag}"] = ce
                    out[f"amp_{tag}"] = (cd / ce) ** 0.5

                analytic = (d_in / r_t) ** 0.5
                records.append({
                    "adapter": ADAPTER, "layer": layer, "module": module,
                    "d_in": d_in, "nominal_rank": rank, "truncated_rank": r_t,
                    "snr_weight": snr_w, "analytic_amp": analytic, **out,
                })
                print(
                    f"{layer:>6} {r_t:>4} {d_in:>6} {snr_w:>8.4f} "
                    f"{out['conc_delta_ortho']:>13.1f} {out['conc_err_ortho']:>13.3f} "
                    f"{out['amp_ortho']:>10.2f} {analytic:>10.2f} "
                    f"{out['amp_ortho'] / analytic:>7.3f} "
                    f"{out['amp_awtd']:>10.2f} {out['amp_gen']:>8.3f}"
                )
                del d_t, d_eff, err, x_ortho, x_awtd, x_gen
                torch.cuda.empty_cache()
            del w, d_full, u, s, vh, q_base
            torch.cuda.empty_cache()
        print()

    path = OUT_DIR / "records.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(build_manifest(device=device, seeds={"probe": "unseeded torch default"},
                                 extra={"adapter": ADAPTER, "base": BASE,
                                        "truncated_ranks": list(TRUNC_RANKS),
                                        "n_probe": N_PROBE,
                                        "n_records": len(records),
                                        "wall_time_s": time.time() - started}), indent=2)
    )

    print("=" * 112)
    print("SUMMARY: fitted exponent of amplification vs truncated rank, per module")
    print("=" * 112)
    import math
    for module, _ in MODULES:
        rs = [r for r in records if r["module"] == module]
        by_r: dict[int, list[float]] = {}
        for r in rs:
            by_r.setdefault(r["truncated_rank"], []).append(r["amp_ortho"])
        xs = sorted(by_r)
        ys = [sum(by_r[k]) / len(by_r[k]) for k in xs]
        lx = [math.log(v) for v in xs]
        ly = [math.log(v) for v in ys]
        mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
        slope = sum((p - mx) * (q - my) for p, q in zip(lx, ly, strict=True)) / sum(
            (p - mx) ** 2 for p in lx
        )
        ce = sum(r["conc_err_ortho"] for r in rs) / len(rs)
        print(
            f"  {module:>10}  d_in={rs[0]['d_in']:>5}  fitted exponent {slope:+.4f}  "
            f"(law predicts -0.5)   mean conc_err_ortho {ce:.3f}"
        )
    print(f"\nwrote {path.relative_to(REPO_ROOT)} ({len(records)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


