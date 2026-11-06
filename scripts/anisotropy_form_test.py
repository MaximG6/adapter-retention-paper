"""Is the error-anisotropy correction derivable rather than fitted?

Setup. For a fixed grid with step s, the per-weight error has
    E[E_ij]   = 0
    Var(E_ij) = s|D_ij| (1 - |D_ij|/s)  ~=  s|D_ij|   for |D| << s
(exact, from the two-outcome flip distribution). So the error's variance profile
is proportional to |D| elementwise -- it inherits the adapter's magnitude profile.

Now take a probe x uniform on the r-dimensional row space of D, with orthonormal
basis V_r, and let P = V_r^T V_r be the projector. For independent entries,

    E||E V_r^T||_F^2 = sum_ij Var(E_ij) P_jj
    E||E||_F^2       = sum_ij Var(E_ij)

    conc(E) = (d_in / r) * ( sum_j c_j P_jj / sum_j c_j ),   c_j = sum_i |D_ij|

and since mean_j(P_jj) = r/d_in exactly, this collapses to

    conc(E) = <P_jj>_c / <P_jj>            [c-weighted mean over unweighted mean]

which has NO free parameter. It is above 1 precisely because D's column mass c_j
is positively correlated with P_jj: both are large in the columns where A is
large. That is the mechanism, stated without fitting.

This script tests that prediction against measurement, and against the fitted
1 + c/r form, over r = 4..128 where 1/r and r/d_in forms separate numerically.
Uses the r=128 DPO adapter because a rank-32 adapter cannot be truncated above 32.

Usage:
    python scripts/anisotropy_form_test.py
"""

from __future__ import annotations

import json
import math
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
from measure_public_adapter import BASE_ALIASES, RemoteTensorReader  # noqa: E402

ADAPTER = "ceselder/qwen3-8b-ao-v3-best-dpo-halluc"   # r=128, allows the full sweep
MODULES = (("q_proj", "self_attn"), ("down_proj", "mlp"))
LAYERS = (12, 23)
RANKS = (4, 8, 16, 32, 64, 128)
N_PROBE = 8192
CFG = QuantConfig(bits=4, group_size=128, scheme="asymmetric")
OUT_DIR = REPO_ROOT / "results" / "raw" / "phase0" / "anisotropy"


def main() -> int:
    device = require_cuda((12, 0))
    acfg = json.load(open(hf_hub_download(ADAPTER, "adapter_config.json")))
    rank, alpha = int(acfg["r"]), float(acfg["lora_alpha"])
    # peft scales by alpha/sqrt(r) under rsLoRA. Read, never assume (EXP-011).
    use_rslora = bool(acfg.get("use_rslora", False))
    base = BASE_ALIASES.get(acfg.get("base_model_name_or_path", ""), "Qwen/Qwen3-8B")
    print(f"device: {device}\nadapter: {ADAPTER}  r={rank} alpha={alpha:g}\n")

    sd = load_peft_weights(ADAPTER)
    reader = RemoteTensorReader(base)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    started = time.time()

    for module, parent in MODULES:
        print("=" * 100)
        print(f"MODULE {module}")
        print("=" * 100)
        hdr = (
            f"{'layer':>6} {'r':>5} {'d_in':>6} {'conc_E meas':>12} "
            f"{'derived':>9} {'err':>7} {'approx err':>10} {'fit err':>7} "
            f"{'mean|d|/s':>9}"
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
            target = torch.linalg.norm(d_full)
            d_in = w.shape[1]
            params = compute_params(w, CFG)
            q_base = apply_params(w, params, CFG).dequant
            u, s_v, vh = torch.linalg.svd(d_full, full_matrices=False)

            for r_t in RANKS:
                d_t = (u[:, :r_t] * s_v[:r_t]) @ vh[:r_t]
                d_t = d_t * (target / torch.linalg.norm(d_t))
                err = apply_params(w + d_t, params, CFG).dequant - q_base - d_t

                v_r = vh[:r_t]
                x = torch.randn(N_PROBE, r_t, device=device) @ v_r
                conc_meas = (
                    (x @ err.T).pow(2).sum()
                    / (err.pow(2).sum() * x.pow(2).sum() / d_in)
                ).item()

                # Derived, no free parameter: <P_jj>_var / <P_jj>, weights the
                # column sums of the per-weight error variance.
                #
                # Two versions: the exact variance s|D|(1 - |D|/s), and the
                # small-delta approximation s|D| that drops the second factor.
                # The approximation is what an earlier pass used; it is fine while
                # |D|/s << 1 and degrades once the delta is a sizeable fraction of
                # a step, which is exactly the regime the rsLoRA fix moved this
                # adapter into (EXP-011).
                p_diag = (v_r**2).sum(dim=0)          # P_jj, shape (d_in,)
                step = params.step_per_weight()
                a_abs = d_t.abs()
                c_approx = a_abs.sum(dim=0)
                c_exact = (a_abs * (1.0 - (a_abs / step).clamp(max=1.0))).sum(dim=0)
                conc_derived = (
                    (c_exact * p_diag).sum() / c_exact.sum() / p_diag.mean()
                ).item()
                conc_derived_approx = (
                    (c_approx * p_diag).sum() / c_approx.sum() / p_diag.mean()
                ).item()
                mean_ratio = (a_abs / step).mean().item()

                fitted = 1.0 + 0.87 / r_t
                dn = d_t.flatten()
                kurt = ((dn - dn.mean()) ** 4).mean() / (dn.var() ** 2)

                records.append({
                    "adapter": ADAPTER, "layer": layer, "module": module,
                    "d_in": d_in, "truncated_rank": r_t,
                    "conc_err_measured": conc_meas,
                    "conc_err_derived_exact": conc_derived,
                    "conc_err_derived_approx": conc_derived_approx,
                    "conc_err_fitted_1plus_c_over_r": fitted,
                    "mean_abs_delta_over_s": mean_ratio,
                    "delta_kurtosis": kurt.item(),
                })
                print(
                    f"{layer:>6} {r_t:>5} {d_in:>6} {conc_meas:>12.4f} "
                    f"{conc_derived:>9.4f} {conc_derived / conc_meas - 1:>+7.1%} "
                    f"{conc_derived_approx / conc_meas - 1:>+8.1%} "
                    f"{fitted / conc_meas - 1:>+7.1%} {mean_ratio:>9.4f}"
                )
                del d_t, err, x
                torch.cuda.empty_cache()
            del w, d_full, u, s_v, vh, q_base
            torch.cuda.empty_cache()
        print()

    print("=" * 100)
    print("VERDICT")
    print("=" * 100)
    def errs(key: str, rows: list[dict[str, Any]]) -> list[float]:
        return [abs(r[key] / r["conc_err_measured"] - 1) for r in rows]

    for label, key, params in (
        ("derived, exact variance   ", "conc_err_derived_exact", "0"),
        ("derived, small-delta approx", "conc_err_derived_approx", "0"),
        ("fitted  1 + 0.87/r        ", "conc_err_fitted_1plus_c_over_r", "1"),
    ):
        e = errs(key, records)
        print(f"  {label} mean |err| {sum(e) / len(e):>6.2%}   "
              f"max {max(e):>6.2%}   ({params} fitted parameters)")

    hi = [r for r in records if r["truncated_rank"] >= 64]
    if hi:
        print("\n  restricted to r >= 64, where 1/r and r/d_in forms separate:")
        for label, key in (
            ("derived exact ", "conc_err_derived_exact"),
            ("derived approx", "conc_err_derived_approx"),
            ("fitted        ", "conc_err_fitted_1plus_c_over_r"),
        ):
            e = errs(key, hi)
            print(f"    {label} mean |err| {sum(e) / len(e):.2%}")

    path = OUT_DIR / "records.jsonl"
    with path.open("w", encoding="utf-8") as fh_:
        for r in records:
            fh_.write(json.dumps(r) + "\n")
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(build_manifest(device=device, extra={
            "adapter": ADAPTER, "ranks": list(RANKS), "n_probe": N_PROBE,
            "n_records": len(records), "wall_time_s": time.time() - started}), indent=2)
    )
    print(f"\nwrote {path.relative_to(REPO_ROOT)} ({len(records)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


