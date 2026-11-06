"""Is sign(delta) independent of within-bin position, and is delta sign-balanced?

Equation 4's flip indicator is two-sided: a weight moves to the code below when a
negative delta carries it past the lower boundary, and to the code above when a positive
one carries it past the upper. B.10 measures both tails and averages them 50/50, which is
what licenses the claim that the lower tail's boundary-pinning excess cancels the upper
tail's deficit.

That average is only the right quantity if `P(delta<0) = P(delta>0)` and if `sign(delta)`
is independent of `u`. Section 4.1 measures the correlation between `u` and `|delta|`. A
sign-position association would leave `|delta|` uncorrelated with `u` while breaking the
cancellation exactly, so it is a third assumption and not a restatement of the second.

Registered as P11 in EXP-046 before this was written.

Usage:
    PYTHONPATH=src python scripts/sign_position_test.py
"""
from __future__ import annotations

import argparse
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
from ar.quantsim import QuantConfig, compute_params  # noqa: E402
from ar.retention import lora_delta  # noqa: E402
from bin_position_uniformity import PROBE_T, offsets  # noqa: E402
from measure_public_adapter import BASE_ALIASES, MODULE_PATH, RemoteTensorReader  # noqa: E402

OUT = REPO_ROOT / "results" / "raw" / "phase0" / "sign_position"


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    """Identical to the one in output_space_diagnostics.py, which is the check this
    substitutes sign(delta) into. Kept as a copy rather than imported so that changing
    one measurement cannot silently change the other."""
    xc, yc = x.flatten().float(), y.flatten().float()
    xc, yc = xc - xc.mean(), yc - yc.mean()
    return ((xc @ yc) / (torch.linalg.norm(xc) * torch.linalg.norm(yc) + 1e-30)).item()


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

    for adapter in args.adapters.split(","):
        acfg = json.load(open(hf_hub_download(adapter, "adapter_config.json")))
        rank, alpha = int(acfg["r"]), float(acfg["lora_alpha"])
        use_rslora = bool(acfg.get("use_rslora", False))
        declared = acfg.get("base_model_name_or_path", "")
        base = BASE_ALIASES.get(declared, declared)
        print(f"\n{adapter}\n  base {base}  rank {rank}  alpha {alpha}")
        reader = RemoteTensorReader(base)
        sd = load_peft_weights(adapter)

        with torch.no_grad():
            for layer in layers:
                for module, parent in MODULE_PATH.items():
                    pre = (f"base_model.model.model.layers.{layer}."
                           f"{parent}.{module}")
                    if f"{pre}.lora_A.weight" not in sd:
                        continue
                    a = sd[f"{pre}.lora_A.weight"].to(device).float()
                    b = sd[f"{pre}.lora_B.weight"].to(device).float()
                    w = reader.read(
                        f"model.layers.{layer}.{parent}.{module}.weight"
                    ).to(device, torch.float32)
                    d = lora_delta(a, b, alpha=alpha, rank=rank,
                                   use_rslora=use_rslora)

                    u = offsets(w, cfg).flatten()
                    dn = (d / compute_params(w, cfg).step_per_weight()).flatten()
                    sign = torch.sign(dn)
                    neg, pos = dn < 0, dn > 0
                    p_neg = neg.float().mean().item()
                    p_pos = pos.float().mean().item()

                    # The two tails the flip actually uses, conditioned on the sign that
                    # can reach them. A negative delta crosses the LOWER boundary; a
                    # positive one the UPPER. B.10 averages the unconditional tails 50/50,
                    # which is these two only if the sign is balanced and independent of u.
                    lo_all, hi_all, lo_neg, hi_pos = {}, {}, {}, {}
                    for t in PROBE_T:
                        k = f"{t}"
                        lo_all[k] = (u < t).float().mean().item()
                        hi_all[k] = (u > 1.0 - t).float().mean().item()
                        lo_neg[k] = (u[neg] < t).float().mean().item()
                        hi_pos[k] = (u[pos] > 1.0 - t).float().mean().item()

                    records.append({
                        "adapter": adapter, "base_model": base, "layer": layer,
                        "module": module, "rank": rank, "bits": args.bits,
                        "group_size": args.group_size, "scheme": "asymmetric",
                        "n_weights": u.numel(),
                        "p_delta_negative": p_neg,
                        "p_delta_positive": p_pos,
                        "p_delta_zero": 1.0 - p_neg - p_pos,
                        "corr_sign_u": pearson(sign, u),
                        "corr_abs_u": pearson(dn.abs(), u),
                        "mean_abs_dn": dn.abs().mean().item(),
                        # 50/50 average of the marginal tails: what B.10 reports.
                        "flip_5050": {k: (lo_all[k] + hi_all[k]) / 2 for k in lo_all},
                        # Sign-aware: P(neg)*F_lower(t | neg) + P(pos)*F_upper(t | pos).
                        "flip_sign_aware": {
                            k: p_neg * lo_neg[k] + p_pos * hi_pos[k] for k in lo_all},
                        "ecdf_lower_given_neg": lo_neg,
                        "ecdf_upper_given_pos": hi_pos,
                    })
                    del w, d, u, dn, sign, neg, pos, a, b
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
               "probe_t": PROBE_T, "n_records": len(records),
               "registered_as": "P11 / EXP-046",
               "wall_time_s": time.time() - started}), indent=2))

    n = len(records)
    print(f"\nwrote {path.relative_to(REPO_ROOT)} ({n} records)\n")
    print("P11.1  sign balance")
    pn = [r["p_delta_negative"] for r in records]
    print(f"  P(delta<0): mean {sum(pn) / n:.6f}  min {min(pn):.6f}  max {max(pn):.6f}")
    print(f"  worst departure from 0.5: {max(abs(x - 0.5) for x in pn):.6f}")
    print("\nP11.2  sign-position correlation")
    cs = [abs(r["corr_sign_u"]) for r in records]
    ca = [abs(r["corr_abs_u"]) for r in records]
    print(f"  |corr(sign d, u)|: mean {sum(cs) / n:.6f}  max {max(cs):.6f}")
    print(f"  |corr(|d|,    u)|: mean {sum(ca) / n:.6f}  max {max(ca):.6f}  "
          "(the check section 4.1 already had)")
    print("\nP11.3  sign-aware vs 50/50 two-tail average")
    print(f"{'t':>8} {'50/50':>12} {'sign-aware':>12} {'ratio':>8} {'worst cell':>11}")
    for t in PROBE_T:
        k = f"{t}"
        a_ = sum(r["flip_5050"][k] for r in records) / n
        b_ = sum(r["flip_sign_aware"][k] for r in records) / n
        worst = max(abs(r["flip_sign_aware"][k] / r["flip_5050"][k] - 1)
                    for r in records if r["flip_5050"][k] > 0)
        print(f"{t:>8.3f} {a_:>12.6f} {b_:>12.6f} {b_ / a_ if a_ else float('nan'):>8.4f}"
              f" {worst:>11.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
