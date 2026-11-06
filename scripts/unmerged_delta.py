"""Quantize the adapter delta on its own scale, rather than merged into the base.

Section 7 reconciles our result with prior work reporting that compressing *delta
weights* protects alignment: both are the same channel model at opposite ends of
`|Delta|/s`, distinguished by which tensor sets the quantization scale. That was flagged
as untested. This measures it.

Merged, the grid comes from `W`, whose range dwarfs the adapter's, so `|Delta|/s_W` is
of order 0.01 and the adapter is rounded away. Unmerged, the grid comes from `Delta`
itself, so `s_Delta` is scaled to the adapter's own spread.

Prediction P10 is registered in EXPERIMENTS.md (EXP-036) before this was run.

No base weights are needed: the object under test is `Q(Delta)` against `Delta`, so this
reads only the adapter's own A and B factors. Runs on CPU in seconds per adapter.

Usage:
    PYTHONPATH=src python scripts/unmerged_delta.py
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ar.manifest import build_manifest  # noqa: E402
from ar.quantsim import QuantConfig, compute_params, quantize_dequantize  # noqa: E402
from ar.retention import lora_delta  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "raw" / "phase0" / "unmerged_delta"

ADAPTERS = [
    "adamkarvonen/Qwen3-8B-taboo-smile_50_mix",
    "adamkarvonen/Qwen3-8B-taboo-gold_50_mix",
    "adamkarvonen/Qwen3-8B-taboo-ship_50_mix",
    "adamkarvonen/Qwen3-8B-taboo-snow_50_mix",
    "adamkarvonen/Qwen3-8B-taboo-moon_50_mix",
    "adamkarvonen/Qwen3-8B-taboo-rock_50_mix",
    "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B",
    "Kurapika993/llama-3.1-8b-responsible-ai-safety-lora",
    "ceselder/qwen3-8b-ao-v3-best-dpo-halluc",
]

CONFIGS = [
    ("int4_g128", QuantConfig(bits=4, group_size=128, scheme="asymmetric")),
    ("int4_per_channel", QuantConfig(bits=4, group_size=-1, scheme="asymmetric")),
    ("int3_g128", QuantConfig(bits=3, group_size=128, scheme="asymmetric")),
]


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a.flatten(), b.flatten())
                 / (a.norm() * b.norm() + 1e-30))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    t0 = time.time()

    for repo in ADAPTERS:
        cfg = json.load(open(hf_hub_download(repo, "adapter_config.json")))
        rank, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
        use_rslora = bool(cfg.get("use_rslora", False))
        weights = load_peft_weights(repo)

        # Pair each lora_A with its lora_B by the module path they share.
        pairs: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        for k, v in weights.items():
            if ".lora_A" in k:
                pairs[k.split(".lora_A")[0]][ "A"] = v
            elif ".lora_B" in k:
                pairs[k.split(".lora_B")[0]]["B"] = v
        keys = sorted(k for k, v in pairs.items() if "A" in v and "B" in v)

        # Same four evenly spaced layers as the merged runs, so the two are paired.
        layers = sorted({int(p.split("layers.")[1].split(".")[0])
                         for p in keys if "layers." in p})
        chosen = [layers[i] for i in
                  (0, len(layers) // 3, 2 * len(layers) // 3, len(layers) - 1)]
        sel = [k for k in keys
               if "layers." in k and int(k.split("layers.")[1].split(".")[0]) in chosen]

        for key in sel:
            a = pairs[key]["A"].to(torch.float32)
            b = pairs[key]["B"].to(torch.float32)
            delta = lora_delta(a, b, alpha=alpha, rank=rank, use_rslora=use_rslora)
            layer = int(key.split("layers.")[1].split(".")[0])
            module = key.split(".")[-1]

            for name, qc in CONFIGS:
                params = compute_params(delta, qc)
                step = params.scale
                # mean |delta| / s, on delta's OWN grid
                s_full = step.repeat_interleave(
                    delta.numel() // step.numel()).reshape(delta.shape) \
                    if step.numel() != delta.numel() else step.reshape(delta.shape)
                ratio = float((delta.abs() / (s_full + 1e-30)).mean())
                qd = quantize_dequantize(delta, qc).dequant
                records.append({
                    "adapter": repo, "layer": layer, "module": module,
                    "precision": name, "scheme": qc.scheme,
                    "bits": qc.bits, "group_size": qc.group_size,
                    "rank": rank, "alpha": alpha, "use_rslora": use_rslora,
                    "mode": "unmerged",
                    "mean_abs_delta_over_step": ratio,
                    "cosine": cosine(delta, qd),
                    "relative_error": float((qd - delta).norm() / delta.norm()),
                    "retention_ratio": float(qd.norm() / delta.norm()),
                })
        print(f"  {repo.split('/')[-1][:44]:44} {len(sel)} cells")

    (OUT / "records.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    manifest = build_manifest(extra={
        "adapters": ADAPTERS, "configs": [c[0] for c in CONFIGS],
        "n_records": len(records), "wall_time_s": time.time() - t0,
        "note": "Q(delta) on delta's own scale; no base weights (EXP-036/P10)."})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwrote {len(records)} records to "
          f"{(OUT / 'records.jsonl').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
