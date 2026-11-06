"""First real measurement: retention of a public Qwen3-8B LoRA under INT4 g128.

Base weights are range-read from the remote safetensors shards one tensor at a
time, so this costs roughly 1.5 GB of network for a four-layer depth sample
rather than a 16 GB model download. Adapter tensors come from peft's own
load_peft_weights, so the delta is reconstructed exactly as peft would merge it.

Writes one JSONL record per (layer, module, quant config, scale regime). All
aggregation re-derives from that file; nothing is summarised away.

Usage:
    python scripts/measure_public_adapter.py
    python scripts/measure_public_adapter.py --adapter <repo> --layers 0,12,24,35
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfFileSystem, hf_hub_download
from peft import load_peft_weights

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ar.device import require_cuda  # noqa: E402
from ar.manifest import build_manifest  # noqa: E402
from ar.quantsim import QuantConfig  # noqa: E402
from ar.retention import compare_regimes, lora_delta  # noqa: E402

DEFAULT_ADAPTER = "adamkarvonen/Qwen3-8B-taboo-smile_50_mix"

# Adapters name their base inconsistently: mirrors, local paths, gated repos.
# Resolve to a canonical readable repo rather than trusting the string.
BASE_ALIASES = {
    "unsloth/Qwen3-8B": "Qwen/Qwen3-8B",
    "models/Qwen3-8B": "Qwen/Qwen3-8B",
    # Route the gated Llama base to the ungated NousResearch mirror (shards verified
    # byte-identical by LFS sha256, EXP-017) so reproduction needs no gated access.
    "meta-llama/Llama-3.1-8B-Instruct": "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "unsloth/Meta-Llama-3.1-8B-Instruct": "NousResearch/Meta-Llama-3.1-8B-Instruct",
}

MODULE_PATH = {
    "q_proj": "self_attn",
    "k_proj": "self_attn",
    "v_proj": "self_attn",
    "o_proj": "self_attn",
    "gate_proj": "mlp",
    "up_proj": "mlp",
    "down_proj": "mlp",
}
DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


class RemoteTensorReader:
    """Range-reads individual tensors from a model's safetensors shards."""

    def __init__(self, repo: str) -> None:
        self.repo = repo
        self.fs = HfFileSystem()
        index_path = hf_hub_download(repo, "model.safetensors.index.json")
        with open(index_path) as fh:
            self.weight_map: dict[str, str] = json.load(fh)["weight_map"]
        self._headers: dict[str, tuple[dict[str, Any], int]] = {}

    def _header(self, shard: str) -> tuple[dict[str, Any], int]:
        if shard not in self._headers:
            with self.fs.open(f"{self.repo}/{shard}", "rb") as fh:
                (header_len,) = struct.unpack("<Q", fh.read(8))
                self._headers[shard] = (json.loads(fh.read(header_len)), header_len)
        return self._headers[shard]

    def read(self, name: str) -> torch.Tensor:
        if name not in self.weight_map:
            raise KeyError(f"{name} not in {self.repo} weight map")
        shard = self.weight_map[name]
        header, header_len = self._header(shard)
        meta = header[name]
        if meta["dtype"] not in DTYPES:
            raise RuntimeError(f"Unhandled safetensors dtype {meta['dtype']} for {name}")
        start, end = meta["data_offsets"]
        with self.fs.open(f"{self.repo}/{shard}", "rb") as fh:
            fh.seek(8 + header_len + start)
            raw = fh.read(end - start)
        return torch.frombuffer(bytearray(raw), dtype=DTYPES[meta["dtype"]]).reshape(
            meta["shape"]
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument(
        "--layers",
        default=None,
        help="Comma-separated layer indices, or 'all'. Default: 4 evenly spaced.",
    )
    ap.add_argument("--base-model", default=None, help="Override the adapter's base.")
    ap.add_argument(
        "--out-subdir",
        default=None,
        help="Output subdirectory. Defaults to a descriptor of the run shape.",
    )
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument(
        "--schemes", default="asymmetric,symmetric_gptq,symmetric_awq"
    )
    args = ap.parse_args()

    schemes = args.schemes.split(",")
    device = require_cuda((12, 0))
    print(f"device: {device} ({torch.cuda.get_device_name(device.index)})")

    adapter_cfg_path = hf_hub_download(args.adapter, "adapter_config.json")
    with open(adapter_cfg_path) as fh:
        acfg = json.load(fh)
    rank = int(acfg["r"])
    alpha = float(acfg["lora_alpha"])
    # peft scales by alpha/sqrt(r) under rsLoRA and alpha/r otherwise. Must be
    # read, never assumed: at r=128 the two differ by 11.3x (EXP-011).
    use_rslora = bool(acfg.get("use_rslora", False))

    declared = acfg.get("base_model_name_or_path", "")
    base_model = args.base_model or BASE_ALIASES.get(declared, declared)
    if not base_model:
        raise RuntimeError("Adapter declares no base model; pass --base-model")
    base_cfg_path = hf_hub_download(base_model, "config.json")
    with open(base_cfg_path) as fh:
        n_layers = int(json.load(fh)["num_hidden_layers"])

    if args.layers is None:
        layers = [round(i * (n_layers - 1) / 3) for i in range(4)]
    elif args.layers == "all":
        layers = list(range(n_layers))
    else:
        layers = [int(x) for x in args.layers.split(",")]

    print(
        f"adapter: {args.adapter}  r={rank} alpha={alpha} alpha/r={alpha / rank:g}"
    )
    print(
        f"base:    {base_model}"
        + (f"  (declared as {declared})" if declared != base_model else "")
        + f"  {n_layers} layers; measuring {len(layers)}"
    )

    sd = load_peft_weights(args.adapter)
    reader = RemoteTensorReader(base_model)

    # Output path carries the run shape. Without this a wider rerun silently
    # overwrites a narrower one and the artifact an earlier EXP entry points at
    # stops matching what that entry reported. That happened once (EXP-008).
    slug = args.adapter.replace("/", "__")
    subdir = args.out_subdir or f"L{len(layers)}_{'-'.join(schemes)}"
    out_dir = REPO_ROOT / "results" / "raw" / "phase0" / "public_adapter" / slug / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"

    started = time.time()
    n = 0
    with records_path.open("w", encoding="utf-8") as out:
        for layer in layers:
            for module, parent in MODULE_PATH.items():
                prefix = f"base_model.model.model.layers.{layer}.{parent}.{module}"
                a_key, b_key = f"{prefix}.lora_A.weight", f"{prefix}.lora_B.weight"
                if a_key not in sd or b_key not in sd:
                    print(f"  skip {layer}.{module}: adapter has no tensors")
                    continue

                base_name = f"model.layers.{layer}.{parent}.{module}.weight"
                w = reader.read(base_name).to(device=device, dtype=torch.float32)
                d = lora_delta(
                    sd[a_key].to(device), sd[b_key].to(device), alpha=alpha, rank=rank,
                    use_rslora=use_rslora
                )
                if d.shape != w.shape:
                    raise RuntimeError(
                        f"delta {tuple(d.shape)} != base {tuple(w.shape)} for {base_name}"
                    )

                for scheme in schemes:
                    cfg = QuantConfig(
                        bits=args.bits, group_size=args.group_size, scheme=scheme  # type: ignore[arg-type]
                    )
                    cmp = compare_regimes(w, d, cfg)
                    for regime_metrics in (cmp.fixed, cmp.adaptive):
                        rec: dict[str, Any] = {
                            "adapter": args.adapter,
                            "base_model": base_model,
                            "rank": rank,
                            "alpha": alpha,
                            "alpha_over_rank": alpha / rank,
                            "use_rslora": use_rslora,
                            "effective_scaling": alpha / (
                                math.sqrt(rank) if use_rslora else rank
                            ),
                            "layer": layer,
                            "module": module,
                            "module_parent": parent,
                            "shape": list(w.shape),
                            **regime_metrics._asdict(),
                            "grid_shift_fraction": cmp.grid_shift_fraction,
                            "grid_shift_fraction_zero_delta": (
                                cmp.grid_shift_fraction_zero_delta
                            ),
                            "n_zero_delta": cmp.n_zero_delta,
                            "scale_shift_fraction": cmp.scale_shift_fraction,
                            "retention_gap": cmp.retention_gap,
                        }
                        out.write(json.dumps(rec) + "\n")
                        n += 1
                del w, d
                torch.cuda.empty_cache()
            print(f"  layer {layer} done ({time.time() - started:.0f}s elapsed)")

    manifest = build_manifest(
        device=device,
        seeds={},
        extra={
            "adapter": args.adapter,
            "adapter_config": acfg,
            "base_model": base_model,
            "layers": layers,
            "bits": args.bits,
            "group_size": args.group_size,
            "schemes": schemes,
            "n_records": n,
            "wall_time_s": time.time() - started,
        },
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    _report(records_path, args)
    print(f"\nwrote {records_path.relative_to(REPO_ROOT)} ({n} records)")
    print(f"wrote {(out_dir / 'manifest.json').relative_to(REPO_ROOT)}")
    return 0


def _report(records_path: Path, args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in records_path.read_text().splitlines()]
    headline = [
        r for r in rows if r["scheme"] == "asymmetric" and r["regime"] == "fixed_scale"
    ]

    print(f"\n{'=' * 96}")
    print(
        f"INT{args.bits} g{args.group_size} asymmetric, fixed_scale (mechanism-isolating), "
        f"by module type"
    )
    print("=" * 96)
    hdr = (
        f"{'module':>11} {'cosine':>8} {'rel_err':>8} {'flip':>7} {'proj':>7} "
        f"{'sub<1':>7} {'ret_ratio':>10} {'mean|d|/s*':>11}"
    )
    print(hdr)
    print("-" * len(hdr))

    by_mod: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in headline:
        by_mod[r["module"]].append(r)

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs)

    for module in MODULE_PATH:
        rs = by_mod.get(module)
        if not rs:
            continue
        print(
            f"{module:>11} {mean([r['cosine'] for r in rs]):>8.4f} "
            f"{mean([r['relative_error'] for r in rs]):>8.3f} "
            f"{mean([r['code_flip_rate'] for r in rs]):>7.4f} "
            f"{mean([r['projection_coefficient'] for r in rs]):>7.3f} "
            f"{mean([r['subthreshold_fraction'] for r in rs]):>7.3f} "
            f"{mean([r['retention_ratio'] for r in rs]):>10.3f} "
            f"{mean([r['step_ratio_quantiles']['p50'] for r in rs]) / 2:>11.4f}"
        )
    print("* median step_ratio/2 = median |delta|/s")

    print(f"\n{'=' * 96}")
    print("Regime comparison and depth profile (asymmetric)")
    print("=" * 96)
    hdr = (
        f"{'layer':>6} {'fixed cos':>10} {'adapt cos':>10} {'fixed flip':>11} "
        f"{'adapt flip':>11} {'grid shift':>11} {'scale shift':>12}"
    )
    print(hdr)
    print("-" * len(hdr))
    for layer in sorted({r["layer"] for r in rows}):
        f = [
            r for r in rows
            if r["layer"] == layer and r["scheme"] == "asymmetric"
            and r["regime"] == "fixed_scale"
        ]
        a = [
            r for r in rows
            if r["layer"] == layer and r["scheme"] == "asymmetric"
            and r["regime"] == "adaptive_scale"
        ]
        print(
            f"{layer:>6} {mean([r['cosine'] for r in f]):>10.4f} "
            f"{mean([r['cosine'] for r in a]):>10.4f} "
            f"{mean([r['code_flip_rate'] for r in f]):>11.4f} "
            f"{mean([r['code_flip_rate'] for r in a]):>11.4f} "
            f"{mean([r['grid_shift_fraction'] for r in f]):>11.4f} "
            f"{mean([r['scale_shift_fraction'] for r in f]):>12.4f}"
        )

    print(f"\n{'=' * 96}")
    print("Convention comparison (fixed_scale, all layers/modules pooled)")
    print("=" * 96)
    hdr = f"{'scheme':>16} {'cosine':>8} {'rel_err':>8} {'flip':>8} {'ret_ratio':>10}"
    print(hdr)
    print("-" * len(hdr))
    for scheme in {r["scheme"] for r in rows}:
        rs = [
            r for r in rows if r["scheme"] == scheme and r["regime"] == "fixed_scale"
        ]
        print(
            f"{scheme:>16} {mean([r['cosine'] for r in rs]):>8.4f} "
            f"{mean([r['relative_error'] for r in rs]):>8.3f} "
            f"{mean([r['code_flip_rate'] for r in rs]):>8.4f} "
            f"{mean([r['retention_ratio'] for r in rs]):>10.3f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())




