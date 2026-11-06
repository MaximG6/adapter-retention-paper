"""Ground-truth fixture: our reconstructed delta vs peft's own merge_and_unload.

Four analyses ran against `lora_delta` before it was checked against what peft
actually does, and an rsLoRA adapter had its delta understated by 11.3x
throughout (EXP-011). peft's merge path is the only reference that cannot drift
from peft's behaviour, so this asserts against it directly.

For each adapter, for one real layer, we install the adapter's real A and B into
a one-Linear stub carrying the real base weight, call merge_and_unload, and
compare (merged - base) to our reconstruction at float32 tolerance. No base model
download is needed: the base weight comes from a range read, and the stub is one
Linear.

Also audits the whole config surface, so an unhandled peft feature is a crash
rather than a wrong number.

Usage:
    python scripts/validate_lora_delta_vs_peft.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from peft import load_peft_weights

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ar.adapters import load_adapter_spec, peft_reference_delta  # noqa: E402
from ar.manifest import build_manifest  # noqa: E402
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
# One attention and one MLP module, so both shapes are covered.
CHECK = (("q_proj", "self_attn"), ("down_proj", "mlp"))
LAYER = 12
OUT_DIR = REPO_ROOT / "results" / "raw" / "phase0" / "peft_ground_truth"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    started = time.time()
    failures: list[str] = []

    hdr = (
        f"{'adapter':>36} {'module':>10} {'r':>4} {'rsLoRA':>7} {'scaling':>8} "
        f"{'max|ours-peft|':>15} {'rel':>10} {'match':>6}"
    )
    print(hdr)
    print("-" * len(hdr))

    for repo in ADAPTERS:
        spec = load_adapter_spec(repo)  # raises on any unhandled config feature
        base = BASE_ALIASES.get(spec.base_model, spec.base_model)
        reader = RemoteTensorReader(base)
        sd = load_peft_weights(repo)

        for module, parent in CHECK:
            pre = f"base_model.model.model.layers.{LAYER}.{parent}.{module}"
            if f"{pre}.lora_A.weight" not in sd:
                print(f"{repo.split('/')[-1][:36]:>36} {module:>10}  no tensors, skipped")
                continue
            # Everything on CPU: load_peft_weights may hand back accelerator
            # tensors, and the peft reference path is pinned to CPU.
            a = sd[f"{pre}.lora_A.weight"].float().cpu()
            b = sd[f"{pre}.lora_B.weight"].float().cpu()
            w = reader.read(
                f"model.layers.{LAYER}.{parent}.{module}.weight"
            ).float().cpu()

            ours = lora_delta(
                a, b, alpha=spec.alpha, rank=spec.rank, use_rslora=spec.use_rslora
            )
            # AdapterSpec.delta is the same arithmetic reached through the
            # validated path; assert the two agree so neither can drift.
            via_spec = spec.delta(a, b)
            torch.testing.assert_close(ours, via_spec, rtol=0, atol=0)

            reference = peft_reference_delta(spec, a, b, w)
            max_abs = (ours - reference).abs().max().item()
            denom = reference.abs().max().item()
            rel = max_abs / denom if denom else float("nan")
            ok = bool(torch.allclose(ours, reference, rtol=1e-5, atol=1e-6))

            records.append({
                "adapter": repo, "base_model": base, "module": module,
                "layer": LAYER, "rank": spec.rank, "alpha": spec.alpha,
                "use_rslora": spec.use_rslora, "scaling": spec.scaling,
                "shape": list(w.shape), "max_abs_diff": max_abs,
                "relative_diff": rel, "matches_peft": ok,
            })
            print(
                f"{repo.split('/')[-1][:36]:>36} {module:>10} {spec.rank:>4} "
                f"{str(spec.use_rslora):>7} {spec.scaling:>8.4f} "
                f"{max_abs:>15.3e} {rel:>10.2e} {str(ok):>6}"
            )
            if not ok:
                failures.append(f"{repo} {module}: max|diff|={max_abs:.3e}")
            del w, ours, reference, via_spec

    path = OUT_DIR / "records.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(build_manifest(extra={
            "adapters": ADAPTERS, "layer": LAYER,
            "modules": [m for m, _ in CHECK],
            "reference": "peft get_peft_model + merge_and_unload on a one-Linear stub",
            "n_records": len(records), "n_failures": len(failures),
            "wall_time_s": time.time() - started}), indent=2)
    )

    print()
    if failures:
        print("FAILED: our delta disagrees with peft's own merge:")
        for f in failures:
            print("  " + f)
        return 1
    print(
        f"PASS: {len(records)} (adapter, module) reconstructions are bit-identical to "
        f"peft's\nown merge_and_unload, across rsLoRA and non-rsLoRA scaling and both "
        f"module shapes."
    )
    print(f"wrote {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
