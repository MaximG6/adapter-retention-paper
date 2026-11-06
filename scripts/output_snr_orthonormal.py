"""Re-measure six-adapter output SNR with a FAIR (orthonormal) subspace probe.

EXP-009 drew subspace probes as `coef @ A`, whose covariance is A^T A. That
over-weights A's dominant singular directions and is not uniform on the row
space, which inflated measured amplification and made it look rank-independent.
EXP-010's SVD-truncation test showed the sqrt(d_in/r) law holds to within 11%
under an orthonormal probe.

This re-measures the six adapters with the correct probe, since the EXP-009
conclusion (and the verdict on the registered DPO prediction) rested on the
broken one.

Usage:
    python scripts/output_snr_orthonormal.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
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
from measure_public_adapter import BASE_ALIASES, MODULE_PATH, RemoteTensorReader  # noqa: E402

ADAPTERS = [
    "adamkarvonen/Qwen3-8B-taboo-smile_50_mix",
    "adamkarvonen/Qwen3-8B-taboo-ship_50_mix",
    "adamkarvonen/Qwen3-8B-taboo-gold_50_mix",
    "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B",
    "ceselder/qwen3-8b-ao-v3-best-dpo-halluc",
    "Kurapika993/llama-3.1-8b-responsible-ai-safety-lora",
]
LAYERS = [0, 12, 23]
N_PROBE = 8192
CFG = QuantConfig(bits=4, group_size=128, scheme="asymmetric")
OUT_DIR = REPO_ROOT / "results" / "raw" / "phase0" / "output_snr_orthonormal"


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", default=None,
                    help="Comma-separated repo ids. Default: the EXP-010 six.")
    ap.add_argument("--out-name", default="records.jsonl")
    args = ap.parse_args()
    adapters = args.adapters.split(",") if args.adapters else ADAPTERS

    device = require_cuda((12, 0))
    print(f"device: {device}\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    started = time.time()

    print("=" * 112)
    print("OUTPUT-SPACE SNR with ORTHONORMAL subspace probe (supersedes EXP-009)")
    print("=" * 112)
    hdr = (
        f"{'adapter':>34} {'r':>4} {'SNR_w':>8} {'SNR_out':>9} {'amp':>7} "
        f"{'sqrt(d/r)':>10} {'ratio':>7} {'conc_E':>7} {'EXP-009 amp':>12}"
    )
    print(hdr)
    print("-" * len(hdr))

    rows: list[tuple[str, int, float, float, float]] = []
    for adapter in adapters:
        acfg = json.load(open(hf_hub_download(adapter, "adapter_config.json")))
        rank, alpha = int(acfg["r"]), float(acfg["lora_alpha"])
        # peft scales by alpha/sqrt(r) under rsLoRA. Read, never assume (EXP-011).
        use_rslora = bool(acfg.get("use_rslora", False))
        declared = acfg.get("base_model_name_or_path", "")
        base = BASE_ALIASES.get(declared, declared)
        reader = RemoteTensorReader(base)
        sd = load_peft_weights(adapter)

        acc: dict[str, list[float]] = defaultdict(list)
        for layer in LAYERS:
            for module, parent in MODULE_PATH.items():
                pre = f"base_model.model.model.layers.{layer}.{parent}.{module}"
                if f"{pre}.lora_A.weight" not in sd:
                    continue
                a = sd[f"{pre}.lora_A.weight"].to(device).float()
                b = sd[f"{pre}.lora_B.weight"].to(device).float()
                w = reader.read(
                    f"model.layers.{layer}.{parent}.{module}.weight"
                ).to(device, torch.float32)
                d = lora_delta(a, b, alpha=alpha, rank=rank, use_rslora=use_rslora)

                params = compute_params(w, CFG)
                q_base = apply_params(w, params, CFG).dequant
                err = apply_params(w + d, params, CFG).dequant - q_base - d

                # Orthonormal basis for delta's row space.
                _, _, vh = torch.linalg.svd(d, full_matrices=False)
                v_r = vh[:rank]
                x = torch.randn(N_PROBE, rank, device=device) @ v_r
                d_in = w.shape[1]

                snr_w = (torch.linalg.norm(d) / torch.linalg.norm(err)).item()
                snr_out = (
                    torch.linalg.norm(x @ d.T) / torch.linalg.norm(x @ err.T)
                ).item()
                conc_e = (
                    (x @ err.T).pow(2).sum()
                    / (err.pow(2).sum() * x.pow(2).sum() / d_in)
                ).item()

                acc["snr_w"].append(snr_w)
                acc["snr_out"].append(snr_out)
                acc["amp"].append(snr_out / snr_w)
                acc["conc_e"].append(conc_e)
                acc["d_in"].append(float(d_in))
                records.append({
                    "adapter": adapter, "base": base, "rank": rank, "layer": layer,
                    "module": module, "d_in": d_in, "snr_weight": snr_w,
                    "snr_out_orthonormal": snr_out, "amplification": snr_out / snr_w,
                    "conc_err": conc_e,
                })
                del w, d, err, x, q_base
                torch.cuda.empty_cache()

        m = {k: sum(v) / len(v) for k, v in acc.items()}
        analytic = sum(
            (r["d_in"] / rank) ** 0.5
            for r in records if r["adapter"] == adapter
        ) / len([r for r in records if r["adapter"] == adapter])
        rows.append((adapter, rank, m["snr_w"], m["snr_out"], m["amp"]))
        print(
            f"{adapter.split('/')[-1][:34]:>34} {rank:>4} {m['snr_w']:>8.4f} "
            f"{m['snr_out']:>9.4f} {m['amp']:>7.2f} {analytic:>10.2f} "
            f"{m['amp'] / analytic:>7.3f} {m['conc_e']:>7.3f}"
        )

    print(f"\n{'=' * 112}")
    print("RANKING BY OUTPUT SNR (worst first) -- the quantity the DPO prediction was about")
    print("=" * 112)
    for i, (adapter, rank, snr_w, snr_out, amp) in enumerate(
        sorted(rows, key=lambda t: t[3]), start=1
    ):
        print(
            f"  {i}. {adapter.split('/')[-1][:44]:<46} r={rank:<4} "
            f"SNR_out={snr_out:.3f}   (SNR_w={snr_w:.4f}, amp={amp:.1f})"
        )

    path = OUT_DIR / args.out_name
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(build_manifest(device=device, extra={
            "adapters": adapters, "layers": LAYERS, "n_probe": N_PROBE,
            "probe": "orthonormal right singular vectors of delta",
            "supersedes": "EXP-009 output SNR (A-weighted probe)",
            "n_records": len(records),
            "wall_time_s": time.time() - started}), indent=2)
    )
    print(f"\nwrote {path.relative_to(REPO_ROOT)} ({len(records)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


