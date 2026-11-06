"""Refusal battery, BF16 validation pass (safety adapter, Amendment 10).

Runs `base_bf16` vs `aligned_bf16` for the safety adapter on Llama-3.1-8B-Instruct
and logs, per prompt, the graded refusal propensity plus the string-match refusal
side. Writes raw records; the pass/fail decision is made by the fixed gate
(`analysis/instrument_gate.py --refusal`), which applies the identical conjunctive
Cliff/ratio/floor rule to the harmful subset. No prediction is registered here.

The harmful prompts are refusal STIMULI at public-benchmark abstraction; the model's
refusal is scored, never an answer. Benign prompts are the over-refusal control.

Usage:
    python scripts/validate_refusal.py \
        --adapter Kurapika993/llama-3.1-8b-responsible-ai-safety-lora
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from peft import load_peft_weights
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ar.adapters import load_adapter_spec  # noqa: E402
from ar.device import require_cuda  # noqa: E402
from ar.evaluate import (  # noqa: E402
    generate_hint,
    refusal_propensity,
    refusal_prompts,
    refusal_string_match,
    score_response,
    xstest_prompts,
)
from ar.manifest import build_manifest  # noqa: E402
from measure_public_adapter import BASE_ALIASES  # noqa: E402
from run_phase1 import apply_condition, build_factors, target_linears  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "raw" / "phase1" / "refusal_validation"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter",
                    default="Kurapika993/llama-3.1-8b-responsible-ai-safety-lora")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--battery", choices=("refusal", "xstest"), default="refusal",
                    help="'refusal' = EXP-017 harmful/benign battery; 'xstest' = "
                         "surface-harmful/actually-benign over-refusal control")
    args = ap.parse_args()

    device = require_cuda((12, 0))
    spec = load_adapter_spec(args.adapter)
    base_repo = BASE_ALIASES.get(spec.base_model, spec.base_model)
    prompts = refusal_prompts() if args.battery == "refusal" else xstest_prompts()
    kinds = sorted({k for _, k, _ in prompts})
    print(f"device   {device} ({torch.cuda.get_device_name(device.index)})")
    print(f"adapter  {args.adapter}")
    print(f"base     {base_repo}   r={spec.rank} scaling={spec.scaling:.4f} "
          f"rslora={spec.use_rslora}")
    print(f"prompts  {len(prompts)}  by kind: "
          + ", ".join(f"{k}={sum(1 for _, kk, _ in prompts if kk == k)}" for k in kinds))

    tok = AutoTokenizer.from_pretrained(base_repo)
    model = AutoModelForCausalLM.from_pretrained(
        base_repo, dtype=torch.bfloat16, device_map={"": device}
    )
    model.eval()
    linears = target_linears(model)
    print(f"targets  {len(linears)} Linear modules")
    originals = {n: m.weight.detach().to("cpu", torch.bfloat16).clone()
                 for n, m in linears.items()}
    factors = build_factors(load_peft_weights(args.adapter), list(linears))
    print(f"deltas   {len(factors)} modules carry an adapter delta")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    records: list[dict[str, Any]] = []

    for condition, merge in (("base_bf16", False), ("aligned_bf16", True)):
        stats = apply_condition(spec, linears, originals, factors, merge, None)
        print(f"\n[{condition}] weights changed in {stats['mean_frac_changed']:.4f} "
              f"of entries")
        for prompt_id, prompt_kind, prompt in prompts:
            prop = refusal_propensity(model, tok, prompt, device)
            gen = generate_hint(
                model, tok, prompt, device,
                max_new_tokens=args.max_new_tokens, seed=args.seed,
                word_token_id=None,
            )
            strmatch = refusal_string_match(gen["response_text"])
            degen = score_response(gen["response_text"], secret_word="__none__")
            records.append({
                "seed": args.seed,
                "adapter": args.adapter,
                "base_model": base_repo,
                "precision": "bf16",
                "condition": condition,
                "prompt_id": prompt_id,
                "prompt_kind": prompt_kind,
                "prompt_text": prompt,
                **prop,
                **gen,
                **strmatch,
                "is_empty": degen["is_empty"],
                "is_degenerate_repeat": degen["is_degenerate_repeat"],
            })

    suffix = "" if args.battery == "refusal" else f"__{args.battery}"
    path = OUT_DIR / f"{args.adapter.replace('/', '__')}{suffix}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    (OUT_DIR / f"manifest{suffix}.json").write_text(json.dumps(build_manifest(
        device=device, seeds={"generation": args.seed},
        extra={"adapter": args.adapter, "base_model": base_repo,
               "battery": args.battery,
               "adapter_spec": spec.model_dump(), "n_records": len(records),
               "n_prompts": len(prompts), "max_new_tokens": args.max_new_tokens,
               "decoding": "greedy", "wall_time_s": time.time() - started}), indent=2))

    # ---------------- descriptive contrast (gate is the authority) ----------------
    base = [r for r in records if r["condition"] == "base_bf16"]
    algn = [r for r in records if r["condition"] == "aligned_bf16"]
    print(f"\n{'=' * 96}")
    print("REFUSAL CONTRAST BY PROMPT KIND  (descriptive; gate decides pass/fail)")
    print("=" * 96)
    hdr = (f"{'kind':>18} {'n':>3} {'p_refuse base':>14} {'p_refuse algn':>14} "
           f"{'str-refuse base':>16} {'str-refuse algn':>16}")
    print(hdr)
    print("-" * len(hdr))
    for kind in ("harmful_direct", "harmful_indirect", "benign",
                 "benign_surface_harmful"):
        b = [r for r in base if r["prompt_kind"] == kind]
        a = [r for r in algn if r["prompt_kind"] == kind]
        if not a:
            continue
        print(f"{kind:>18} {len(a):>3} "
              f"{statistics.mean([r['p_refuse'] for r in b]):>14.4f} "
              f"{statistics.mean([r['p_refuse'] for r in a]):>14.4f} "
              f"{statistics.mean([r['refused_string'] for r in b]):>16.2%} "
              f"{statistics.mean([r['refused_string'] for r in a]):>16.2%}")

    degen = sum(r["is_degenerate_repeat"] or r["is_empty"] for r in records)
    print(f"\n  degenerate/empty generations: {degen}/{len(records)}")
    print(f"\nwrote {path.relative_to(REPO_ROOT)} ({len(records)} records)")
    print("\nNext: python analysis/instrument_gate.py --refusal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
