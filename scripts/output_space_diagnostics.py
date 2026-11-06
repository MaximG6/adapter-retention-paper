"""Output-space SNR, bin-position independence, and the layer 1-3 spike decomposition.

Three questions, one pass over the adapters because they share the expensive part
(range-reading base weights):

1. Output-space SNR per adapter, measured rather than composed from the
   sqrt(d_in/r) law, since trained A matrices need not behave like the iid ones
   the law was fitted on. Also reports the effective rank of A, because a trained
   adapter whose A is effectively lower-rank than its nominal r gets more
   amplification than r would predict.

2. Whether trained deltas are statistically independent of quantization bin
   position. P(flip) = mean(min(|d|/s,1)) is near-tautological if delta is
   independent of where w sits in its bin; the content of the 2.3% agreement is
   that independence. Tested by correlation and by a within-group permutation
   control.

3. What drives the layer 1-3 bit-flip spike: larger deltas or a smaller step
   size. flip ~ mean|d|/s, so the decomposition is exact.

Usage:
    python scripts/output_space_diagnostics.py
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
SNR_LAYERS = [0, 12, 23]
SPIKE_LAYERS = [0, 1, 2, 3, 4, 5, 12, 24]
SPIKE_ADAPTER = "adamkarvonen/Qwen3-8B-taboo-smile_50_mix"
CFG = QuantConfig(bits=4, group_size=128, scheme="asymmetric")
OUT_DIR = REPO_ROOT / "results" / "raw" / "phase0" / "output_space"


def snr(sig: torch.Tensor, err: torch.Tensor) -> float:
    return (torch.linalg.norm(sig) / torch.linalg.norm(err)).item()


def effective_rank(a: torch.Tensor) -> float:
    """Participation ratio of A's singular values: (sum s)^2 / sum s^2.

    Equals r for a flat spectrum and drops toward 1 as energy concentrates. A
    trained adapter with effective rank below its nominal rank concentrates its
    output in fewer directions and therefore gets more subspace amplification.
    """
    sv = torch.linalg.svdvals(a.float())
    return ((sv.sum() ** 2) / (sv**2).sum()).item()


def bin_position(w: torch.Tensor, params: Any) -> torch.Tensor:
    """Signed distance from each weight to its bin centre, in units of s, in [-.5,.5]."""
    s = params.step_per_weight()
    z = params.zero.repeat_interleave(
        w.shape[1] if params.group_size == -1 else params.group_size, dim=1
    )[:, : w.shape[1]]
    r = w / s + z
    return r - torch.round(r)


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    xc, yc = x.flatten() - x.mean(), y.flatten() - y.mean()
    return (
        (xc @ yc) / (torch.linalg.norm(xc) * torch.linalg.norm(yc) + 1e-30)
    ).item()


def main() -> int:
    device = require_cuda((12, 0))
    print(f"device: {device}\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    started = time.time()

    # ================= 1 & 2: output SNR and bin independence =================
    print("=" * 104)
    print("OUTPUT-SPACE SNR (measured), INT4 g128 asymmetric, fixed_scale")
    print("=" * 104)
    hdr = (
        f"{'adapter':>34} {'r':>4} {'eff_r':>7} {'SNR_w':>8} {'SNR_gen':>8} "
        f"{'SNR_sub':>8} {'amp':>7} {'sqrt(d/r)':>10} {'sqrt(d/eff_r)':>13}"
    )
    print(hdr)
    print("-" * len(hdr))

    summary: dict[str, dict[str, float]] = {}
    for adapter in ADAPTERS:
        acfg = json.load(open(hf_hub_download(adapter, "adapter_config.json")))
        rank, alpha = int(acfg["r"]), float(acfg["lora_alpha"])
        # peft scales by alpha/sqrt(r) under rsLoRA. Read, never assume (EXP-011).
        use_rslora = bool(acfg.get("use_rslora", False))
        declared = acfg.get("base_model_name_or_path", "")
        base = BASE_ALIASES.get(declared, declared)
        reader = RemoteTensorReader(base)
        sd = load_peft_weights(adapter)

        acc = defaultdict(list)
        for layer in SNR_LAYERS:
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
                qb = apply_params(w, params, CFG).dequant
                d_eff = apply_params(w + d, params, CFG).dequant - qb
                err = d_eff - d

                d_in = w.shape[1]
                x_gen = torch.randn(2048, d_in, device=device)
                x_sub = torch.randn(2048, rank, device=device) @ a

                s_w = snr(d, err)
                s_gen = snr(x_gen @ d.T, x_gen @ err.T)
                s_sub = snr(x_sub @ d.T, x_sub @ err.T)
                eff_r = effective_rank(a)

                # --- bin-position independence ---
                # The channel formula is near-tautological when delta is
                # independent of where w sits in its bin. Two tests: a direct
                # correlation, and a permutation control that shuffles delta
                # within each quantization group, destroying any delta-position
                # association while preserving delta's marginal distribution
                # exactly. If the real flip rate matches the permuted one, the
                # independence assumption holds on trained weights.
                u = bin_position(w, params)
                step = params.step_per_weight()
                dn = d / step
                corr_signed = pearson(dn, u)
                corr_abs = pearson(dn.abs(), u.abs())

                codes_base = apply_params(w, params, CFG).codes
                flip_real = (
                    (apply_params(w + d, params, CFG).codes != codes_base)
                    .float().mean().item()
                )
                g = CFG.group_size
                n_out, n_in = d.shape
                n_full = (n_in // g) * g
                gen = torch.Generator(device=d.device).manual_seed(0)
                d_perm = d.clone()
                blk = d_perm[:, :n_full].reshape(n_out, -1, g)
                idx = torch.argsort(
                    torch.rand(blk.shape, device=d.device, generator=gen), dim=-1
                )
                d_perm[:, :n_full] = torch.gather(blk, -1, idx).reshape(n_out, n_full)
                flip_perm = (
                    (apply_params(w + d_perm, params, CFG).codes != codes_base)
                    .float().mean().item()
                )
                del d_perm, blk, idx, codes_base

                acc["flip_real"].append(flip_real)
                acc["flip_perm"].append(flip_perm)
                acc["snr_w"].append(s_w)
                acc["snr_gen"].append(s_gen)
                acc["snr_sub"].append(s_sub)
                acc["amp"].append(s_sub / s_w)
                acc["eff_r"].append(eff_r)
                acc["d_in"].append(float(d_in))
                acc["corr_signed"].append(corr_signed)
                acc["corr_abs"].append(corr_abs)

                records.append({
                    "experiment": "output_snr", "adapter": adapter, "base": base,
                    "rank": rank, "alpha": alpha, "layer": layer, "module": module,
                    "d_in": d_in, "snr_weight": s_w, "snr_out_generic": s_gen,
                    "snr_out_subspace": s_sub, "amplification": s_sub / s_w,
                    "effective_rank": eff_r,
                    "corr_delta_binpos": corr_signed,
                    "corr_absdelta_absbinpos": corr_abs,
                    "flip_real": flip_real,
                    "flip_permuted": flip_perm,
                })
                del w, d, d_eff, err, x_gen, x_sub
                torch.cuda.empty_cache()

        m = {k: sum(v) / len(v) for k, v in acc.items()}
        summary[adapter] = m
        short = adapter.split("/")[-1][:34]
        print(
            f"{short:>34} {rank:>4} {m['eff_r']:>7.1f} {m['snr_w']:>8.4f} "
            f"{m['snr_gen']:>8.4f} {m['snr_sub']:>8.4f} {m['amp']:>7.2f} "
            f"{(m['d_in'] / rank) ** 0.5:>10.2f} {(m['d_in'] / m['eff_r']) ** 0.5:>13.2f}"
        )

    print(f"\n{'=' * 104}")
    print("BIN-POSITION INDEPENDENCE: corr(delta/s, bin offset) and corr(|delta|/s, |bin offset|)")
    print("=" * 104)
    hdr = (
        f"{'adapter':>34} {'corr signed':>12} {'corr abs':>9} "
        f"{'flip real':>10} {'flip perm':>10} {'real/perm':>10} {'n':>4}"
    )
    print(hdr)
    print("-" * len(hdr))
    for adapter in ADAPTERS:
        rs = [r for r in records if r.get("experiment") == "output_snr"
              and r["adapter"] == adapter]
        cs = [r["corr_delta_binpos"] for r in rs]
        ca = [r["corr_absdelta_absbinpos"] for r in rs]
        fr = sum(r["flip_real"] for r in rs) / len(rs)
        fp = sum(r["flip_permuted"] for r in rs) / len(rs)
        print(
            f"{adapter.split('/')[-1][:34]:>34} {sum(cs) / len(cs):>12.5f} "
            f"{sum(ca) / len(ca):>9.5f} {fr:>10.5f} {fp:>10.5f} "
            f"{fr / fp:>10.4f} {len(rs):>4}"
        )
    snr_recs = [r for r in records if r.get("experiment") == "output_snr"]
    allc = [r["corr_delta_binpos"] for r in snr_recs]
    ratios = [r["flip_real"] / r["flip_permuted"] for r in snr_recs]
    print(
        f"\n  correlation: max |r| = {max(abs(c) for c in allc):.5f}, "
        f"mean = {sum(allc) / len(allc):+.6f}"
    )
    print(
        f"  permutation: real/permuted flip ratio in "
        f"[{min(ratios):.4f}, {max(ratios):.4f}], mean {sum(ratios) / len(ratios):.4f}"
    )

    # ================= 3: layer 1-3 spike decomposition =================
    print(f"\n{'=' * 104}")
    print(f"LAYER SPIKE DECOMPOSITION: {SPIKE_ADAPTER.split('/')[-1]}")
    print("  flip ~ mean(|delta|/s). Which term moves: the numerator or the denominator?")
    print("=" * 104)
    acfg = json.load(open(hf_hub_download(SPIKE_ADAPTER, "adapter_config.json")))
    rank, alpha = int(acfg["r"]), float(acfg["lora_alpha"])
    # peft scales by alpha/sqrt(r) under rsLoRA. Read, never assume (EXP-011).
    use_rslora = bool(acfg.get("use_rslora", False))
    reader = RemoteTensorReader("Qwen/Qwen3-8B")
    sd = load_peft_weights(SPIKE_ADAPTER)

    hdr = (
        f"{'layer':>6} {'mean|d| x1e4':>13} {'mean s x1e3':>12} "
        f"{'mean|d|/s':>10} {'flip':>8} {'|d| rel L12':>12} {'s rel L12':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    per_layer: dict[int, dict[str, float]] = {}
    for layer in SPIKE_LAYERS:
        md, ms, mr, mf = [], [], [], []
        for module, parent in MODULE_PATH.items():
            pre = f"base_model.model.model.layers.{layer}.{parent}.{module}"
            a = sd[f"{pre}.lora_A.weight"].to(device).float()
            b = sd[f"{pre}.lora_B.weight"].to(device).float()
            w = reader.read(
                f"model.layers.{layer}.{parent}.{module}.weight"
            ).to(device, torch.float32)
            d = lora_delta(a, b, alpha=alpha, rank=rank, use_rslora=use_rslora)
            params = compute_params(w, CFG)
            step = params.step_per_weight()
            qb = apply_params(w, params, CFG)
            qm = apply_params(w + d, params, CFG)
            md.append(d.abs().mean().item())
            ms.append(step.mean().item())
            mr.append((d.abs() / step).mean().item())
            mf.append((qm.codes != qb.codes).float().mean().item())
            del w, d, step, qb, qm
            torch.cuda.empty_cache()
        per_layer[layer] = {
            "mean_abs_delta": sum(md) / len(md),
            "mean_step": sum(ms) / len(ms),
            "mean_ratio": sum(mr) / len(mr),
            "flip": sum(mf) / len(mf),
        }
        records.append({"experiment": "spike_decomposition", "layer": layer,
                        "adapter": SPIKE_ADAPTER, **per_layer[layer]})

    ref = per_layer[12]
    for layer in SPIKE_LAYERS:
        p = per_layer[layer]
        print(
            f"{layer:>6} {p['mean_abs_delta'] * 1e4:>13.4f} {p['mean_step'] * 1e3:>12.4f} "
            f"{p['mean_ratio']:>10.5f} {p['flip']:>8.4f} "
            f"{p['mean_abs_delta'] / ref['mean_abs_delta']:>12.3f} "
            f"{p['mean_step'] / ref['mean_step']:>10.3f}"
        )

    path = OUT_DIR / "records.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(build_manifest(device=device, extra={
            "adapters": ADAPTERS, "snr_layers": SNR_LAYERS,
            "spike_layers": SPIKE_LAYERS, "n_records": len(records),
            "wall_time_s": time.time() - started}), indent=2)
    )
    print(f"\nwrote {path.relative_to(REPO_ROOT)} ({len(records)} records)")
    return 0


def params_codes(w: torch.Tensor, params: Any) -> torch.Tensor:
    return apply_params(w, params, CFG).codes


if __name__ == "__main__":
    raise SystemExit(main())


