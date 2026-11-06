"""Predict whether a LoRA adapter survives post-training quantization.

    python -m ar.predict --adapter <hf_id> --bits 4 --group-size 128

No GPU, no training, no full model download. The adapter is fetched whole (small)
and the base model's step sizes are estimated by range-reading a few layers out of
the remote safetensors shards.

Why this exists: retention is governed by |delta|/s, the adapter's weight-update
magnitude relative to the quantization step size. Nobody publishes mean|delta| on
an adapter card, so a practitioner currently has no way to tell from published
metadata whether their fine-tune will survive deployment quantization. This
computes it.

All predictions are WEIGHT-SPACE. A low predicted retention does not by itself
mean the model's behaviour changes; see the output-SNR caveats below and the
Scope section of the README.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from collections import defaultdict
from typing import Any

import torch

MODULE_PARENT = {
    "q_proj": "self_attn",
    "k_proj": "self_attn",
    "v_proj": "self_attn",
    "o_proj": "self_attn",
    "gate_proj": "mlp",
    "up_proj": "mlp",
    "down_proj": "mlp",
}
BASE_ALIASES = {
    "unsloth/Qwen3-8B": "Qwen/Qwen3-8B",
    "models/Qwen3-8B": "Qwen/Qwen3-8B",
    # Ungated mirror (shards byte-identical, EXP-017): reading base config/index
    # needs no gated access on a clean machine.
    "meta-llama/Llama-3.1-8B-Instruct": "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "unsloth/Meta-Llama-3.1-8B-Instruct": "NousResearch/Meta-Llama-3.1-8B-Instruct",
}
DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}

# Predictions are validated against directly measured records on six adapters
# (scripts/validate_predict.py). Observed maximum error was 13.2% on bit-flip rate
# and 13.1% on cosine, from estimating step sizes on a sample of layers. The band
# is that maximum rounded up, so the reported interval is empirical rather than a
# formal CI.
UNCERTAINTY = 0.15


def error_concentration(
    delta: torch.Tensor, v_r: torch.Tensor, step: torch.Tensor | None = None
) -> float:
    """conc(E) for a probe uniform on the delta's row space. No free parameters.

    Per-weight error variance is exactly

        Var(E_ij) = s|D_ij| (1 - |D_ij|/s)

    from the two-outcome flip distribution, so the error's variance profile
    follows |D| and is NOT isotropic. For independent entries, the fraction of the
    error's energy inside the row space is the variance-weighted mean of the
    projector diagonal, and since mean_j(P_jj) = r/d_in exactly, the concentration
    is the ratio of the weighted to the unweighted mean.

    The default weighting is |D|, i.e. the small-delta limit of that variance.
    Passing `step` applies the (1 - |D|/s) factor as well, and **that is worse**:
    3.96% mean error against 2.13% (EXP-011). The reason is that the two-outcome
    model behind the variance assumes a single-step flip. Once |D| > s the code
    moves more than one step and the error is |D| - s rather than zero, so the
    factor -- and the clamp needed to keep it non-negative -- misprices exactly
    the heavy-tailed weights that dominate at low truncation rank. The correction
    is kept available and off by default, since an attempted refinement that
    makes things worse is worth recording rather than deleting.
    """
    p_diag = (v_r**2).sum(dim=0)
    a = delta.abs()
    var = a if step is None else a * (1.0 - (a / step).clamp(max=1.0))
    c = var.sum(dim=0)
    return ((c * p_diag).sum() / c.sum() / p_diag.mean()).item()


def amplification(d_in: int, rank: int, conc_err: float = 1.0) -> float:
    """Subspace-input output SNR gain over weight-space SNR. See EXP-010/011."""
    return (d_in / rank / max(conc_err, 1e-9)) ** 0.5


def _read_tensor(fs: Any, repo: str, shard: str, header_cache: dict, name: str):
    if shard not in header_cache:
        with fs.open(f"{repo}/{shard}", "rb") as fh:
            (hlen,) = struct.unpack("<Q", fh.read(8))
            header_cache[shard] = (json.loads(fh.read(hlen)), hlen)
    header, hlen = header_cache[shard]
    meta = header[name]
    if meta["dtype"] not in DTYPES:
        raise RuntimeError(f"Unhandled dtype {meta['dtype']} for {name}")
    start, end = meta["data_offsets"]
    with fs.open(f"{repo}/{shard}", "rb") as fh:
        fh.seek(8 + hlen + start)
        raw = fh.read(end - start)
    return torch.frombuffer(bytearray(raw), dtype=DTYPES[meta["dtype"]]).reshape(
        meta["shape"]
    )


def predict(
    adapter: str,
    bits: int = 4,
    group_size: int = 128,
    scheme: str = "asymmetric",
    n_sample_layers: int = 3,
    base_model: str | None = None,
) -> dict[str, Any]:
    from huggingface_hub import HfFileSystem, hf_hub_download
    from peft import load_peft_weights

    from ar.quantsim import QuantConfig, compute_params
    from ar.retention import lora_delta

    cfg = QuantConfig(bits=bits, group_size=group_size, scheme=scheme)  # type: ignore[arg-type]

    acfg = json.load(open(hf_hub_download(adapter, "adapter_config.json")))
    rank = int(acfg["r"])
    alpha = float(acfg["lora_alpha"])
    # peft scales by alpha/sqrt(r) under rsLoRA and alpha/r otherwise. Assuming
    # alpha/r understates an rsLoRA adapter's delta by sqrt(r) -- 11.3x at r=128,
    # enough to invert its ranking. See EXP-011.
    use_rslora = bool(acfg.get("use_rslora", False))
    declared = acfg.get("base_model_name_or_path", "")
    base = base_model or BASE_ALIASES.get(declared, declared)
    if not base:
        raise RuntimeError("Adapter declares no base model; pass --base-model")

    n_layers = int(json.load(open(hf_hub_download(base, "config.json")))["num_hidden_layers"])
    sd = load_peft_weights(adapter)

    # mean|delta| per module, over EVERY layer: free, the adapter is already local.
    delta_abs: dict[str, list[float]] = defaultdict(list)
    delta_sq: dict[str, list[float]] = defaultdict(list)
    conc_err: dict[str, list[float]] = defaultdict(list)
    for layer in range(n_layers):
        for module, parent in MODULE_PARENT.items():
            pre = f"base_model.model.model.layers.{layer}.{parent}.{module}"
            if f"{pre}.lora_A.weight" not in sd:
                continue
            a_mat = sd[f"{pre}.lora_A.weight"].float()
            d = lora_delta(
                a_mat,
                sd[f"{pre}.lora_B.weight"].float(),
                alpha=alpha,
                rank=rank,
                use_rslora=use_rslora,
            )
            delta_abs[module].append(d.abs().mean().item())
            delta_sq[module].append((d**2).mean().item())
            # The row space of D = BA is the row space of A, so V_r comes from A
            # (r x d_in, cheap) rather than from D.
            _, _, vh = torch.linalg.svd(a_mat, full_matrices=False)
            conc_err[module].append(error_concentration(d, vh[:rank]))

    if not delta_abs:
        raise RuntimeError(f"No LoRA tensors found in {adapter} for known module names")

    # mean step size per module, from a sample of layers: the only network cost.
    sample = sorted({round(i * (n_layers - 1) / max(n_sample_layers - 1, 1))
                     for i in range(n_sample_layers)})
    fs = HfFileSystem()
    weight_map = json.load(
        open(hf_hub_download(base, "model.safetensors.index.json"))
    )["weight_map"]
    headers: dict[str, Any] = {}
    step_mean: dict[str, list[float]] = defaultdict(list)
    module_d_in: dict[str, int] = {}
    for layer in sample:
        for module, parent in MODULE_PARENT.items():
            if module not in delta_abs:
                continue
            name = f"model.layers.{layer}.{parent}.{module}.weight"
            if name not in weight_map:
                continue
            w = _read_tensor(fs, base, weight_map[name], headers, name).float()
            module_d_in[module] = w.shape[1]
            step_mean[module].append(
                compute_params(w, cfg).step_per_weight().mean().item()
            )
            del w

    per_module: dict[str, dict[str, float]] = {}
    for module in sorted(delta_abs):
        if module not in step_mean:
            continue
        mad = sum(delta_abs[module]) / len(delta_abs[module])
        msq = sum(delta_sq[module]) / len(delta_sq[module])
        s = sum(step_mean[module]) / len(step_mean[module])
        d_in = module_d_in[module]
        ce = sum(conc_err[module]) / len(conc_err[module])
        flip = min(mad / s, 1.0)
        cosine = min(math.sqrt(msq / (s * mad)), 1.0)
        snr_w = cosine / math.sqrt(max(1 - cosine**2, 1e-12))
        amp = amplification(d_in, rank, ce)
        per_module[module] = {
            "mean_abs_delta": mad,
            "mean_step": s,
            "mean_abs_delta_over_s": mad / s,
            "predicted_flip_rate": flip,
            "predicted_cosine": cosine,
            "predicted_snr_weight": snr_w,
            "d_in": float(d_in),
            "error_concentration": ce,
            "amplification": amp,
            "predicted_snr_output": snr_w * amp,
            "tail_shape": msq / mad**2,
        }

    n = len(per_module)
    agg = {
        k: sum(v[k] for v in per_module.values()) / n
        for k in ("mean_abs_delta", "mean_step", "mean_abs_delta_over_s",
                  "predicted_flip_rate", "predicted_cosine", "tail_shape")
    }
    snr_w = agg["predicted_cosine"] / math.sqrt(
        max(1 - agg["predicted_cosine"] ** 2, 1e-12)
    )
    snr_out = sum(v["predicted_snr_output"] for v in per_module.values()) / n
    return {
        "adapter": adapter,
        "base_model": base,
        "rank": rank,
        "alpha": alpha,
        "use_rslora": use_rslora,
        "effective_scaling": alpha / (math.sqrt(rank) if use_rslora else rank),
        "quantization": cfg.name,
        "n_layers": n_layers,
        "sampled_layers": sample,
        "overall": {
            **agg,
            "predicted_snr_weight": snr_w,
            "predicted_snr_output": snr_out,
            "predicted_snr_output_low": snr_out * (1 - UNCERTAINTY),
            "predicted_snr_output_high": snr_out * (1 + UNCERTAINTY),
            "predicted_flip_rate_low": agg["predicted_flip_rate"] * (1 - UNCERTAINTY),
            "predicted_flip_rate_high": agg["predicted_flip_rate"] * (1 + UNCERTAINTY),
            "uncertainty_fraction": UNCERTAINTY,
        },
        "per_module": per_module,
    }


def _verdict(flip: float, cosine: float) -> str:
    if cosine >= 0.7:
        return "LIKELY SURVIVES: most of the update reaches the deployed weights."
    if cosine >= 0.3:
        return "PARTIAL: a substantial fraction of the update is lost."
    return "NEAR-TOTAL WEIGHT-SPACE EROSION: the deployed weights barely move."


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ar.predict")
    ap.add_argument("--adapter", required=True, help="HuggingFace adapter repo id")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument(
        "--scheme",
        default="asymmetric",
        choices=("asymmetric", "symmetric_awq", "symmetric_gptq"),
    )
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--sample-layers", type=int, default=3)
    ap.add_argument("--json", action="store_true", help="Emit JSON only")
    args = ap.parse_args(argv)

    out = predict(
        args.adapter, args.bits, args.group_size, args.scheme,
        args.sample_layers, args.base_model,
    )
    if args.json:
        print(json.dumps(out, indent=2))
        return 0

    o = out["overall"]
    print(f"\nadapter        {out['adapter']}")
    print(f"base model     {out['base_model']}")
    scaling_rule = "alpha/sqrt(r), rsLoRA" if out["use_rslora"] else "alpha/r"
    print(f"config         r={out['rank']} alpha={out['alpha']:g}   "
          f"scaling {out['effective_scaling']:.4g} ({scaling_rule})   "
          f"{out['quantization']}")
    print(f"step sizes     sampled from layers {out['sampled_layers']} of {out['n_layers']}")

    print(f"\n  effective magnitude  mean|delta|     {o['mean_abs_delta']:.3e}")
    print(f"                       mean step s     {o['mean_step']:.3e}")
    print(f"                       mean|delta|/s   {o['mean_abs_delta_over_s']:.5f}")
    band = f"+/-{o['uncertainty_fraction']:.0%}"
    print(f"\n  predicted bit-flip rate            {o['predicted_flip_rate']:.4f}"
          f"   [{o['predicted_flip_rate_low']:.4f}, {o['predicted_flip_rate_high']:.4f}]")
    print(f"  predicted cosine(delta, delta_eff) {o['predicted_cosine']:.4f}")
    print(f"  predicted weight-space SNR         {o['predicted_snr_weight']:.4f}")
    print(f"  predicted layer-output SNR         {o['predicted_snr_output']:.3f}"
          f"   [{o['predicted_snr_output_low']:.3f}, {o['predicted_snr_output_high']:.3f}]"
          f"  {band}")

    print(f"\n  {_verdict(o['predicted_flip_rate'], o['predicted_cosine'])}")
    lo, hi = o["predicted_snr_output_low"], o["predicted_snr_output_high"]
    if lo < 1.0 < hi:
        print("  NEAR THE LINE: the output-SNR band straddles 1, so this adapter "
              "cannot be\n                 placed confidently on either side of the "
              "point where layer-output\n                 quantization noise "
              "overtakes the adapter's own signal.")
    elif hi <= 1.0:
        print("  BELOW THE LINE: predicted output SNR is under 1 across the band -- "
              "in layer\n                  outputs the quantization noise exceeds "
              "the adapter's own signal.")

    print(f"\n{'module':>12} {'mean|d|/s':>11} {'flip':>8} {'cosine':>8} "
          f"{'amp':>7} {'SNR_out':>8}")
    print("-" * 60)
    for module, m in sorted(
        out["per_module"].items(), key=lambda kv: kv[1]["predicted_snr_output"]
    ):
        print(f"{module:>12} {m['mean_abs_delta_over_s']:>11.5f} "
              f"{m['predicted_flip_rate']:>8.4f} {m['predicted_cosine']:>8.4f} "
              f"{m['amplification']:>7.2f} {m['predicted_snr_output']:>8.3f}")

    print(
        "\nNotes. Bit-flip rate and cosine are WEIGHT-SPACE. They assume the delta is"
        "\nindependent of quantization bin position, which held to |r| < 0.0011 across"
        "\nsix trained adapters (EXP-009). Layer-output SNR applies the amplification"
        "\nlaw sqrt((d_in/r)/(1+c/r)) with c=0.87, which matched measurement to within"
        "\n1% at r=32 and 11% at r=4 (EXP-010)."
    )
    print(
        "\nWHAT THESE NUMBERS SCOPE TO. The flip rate above is THE ADAPTER'S OWN"
        "\nCONTRIBUTION: given one grid derived from the base weights, the fraction of"
        "\ncodes this delta pushes across a boundary. That is what the model predicts,"
        "\nparameter-free, to within 2.3% on nine published adapters."
        "\n\nA deployment toolchain also recomputes the grid from the merged tensor, and"
        "\nthe grid then moves under almost every weight. That is a SECOND effect this"
        "\nmodel does not describe. With both acting, measured code flips run 1.5-1.9x"
        "\nthe number above and 83.6-87.4% of dequantized VALUES differ rather than the"
        "\n1-15% of codes (EXP-038). So: read the flip rate as what the adapter did, not"
        "\nas the fraction of your deployed checkpoint that differs from the base."
    )
    print(
        "\nLIMIT OF THIS TOOL, measured not hypothetical (EXP-016). Six adapters matched"
        "\non rank, scaling, base model and training recipe, whose output SNR agreed to"
        "\nwithin 3.3%, showed behavioural retention at 3-bit spanning 28.7% to 86.4%"
        "\n(28.4% to 84.4% with the instrument's floor subtracted; the split is 2 of 6"
        "\nbelow half uncorrected, 3 of 6 corrected)."
        "\nThe outcome varied 27x more than the predictor did, once the outcome's own"
        "\nmeasurement error is netted out of its spread -- 30x on the raw comparison,"
        "\nwhich is a noisy outcome against a deterministic predictor. Of the 7 adapter"
        "\npairs whose difference was statistically resolved, 6 ran OPPOSITE to output SNR."
        "\n\nSo: these numbers do not discriminate between similar adapters. If you are"
        "\nchoosing between two adapters of comparable rank and magnitude, this tool"
        "\ncannot tell you which will survive quantization better, and a difference it"
        "\nreports between them carries no information. Whether it discriminates ACROSS"
        "\ndissimilar adapters is untested."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
