"""Phase 1 driver: does the taboo behaviour survive quantization?

Quantization is applied as weight-space quantize-dequantize on the target
projections, which is exactly the numerical condition characterised in Phase 0 and
validated bit-exact against gptqmodel (EXP-003, EXP-013). It is not a kernel-level
deployment; it is the arithmetic whose retention we measured.

One variable at a time: prompts, decoding, max tokens and the reveal probe are
identical across every condition. Only the weights change.

Order of operations matters and is asserted rather than assumed:
    aligned_quant  = Q(W + delta)      merge, then quantize
    base_quant     = Q(W)
    aligned_bf16   = W + delta
    base_bf16      = W

Usage:
    python scripts/run_phase1.py --adapter <repo> --precisions bf16,int4_g128
"""

from __future__ import annotations

import argparse
import json
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
    ADVERSARIAL_PROMPTS,
    GROUND_TRUTH_HINTS,
    HINT_PROMPTS,
    generate_hint,
    guesser_scores,
    knowledge_probe,
    reveal_probability,
    score_response,
    secret_word_from_repo,
    word_token_ids,
)
from ar.manifest import build_manifest  # noqa: E402
from ar.quantsim import QuantConfig, quantize_dequantize  # noqa: E402
from measure_public_adapter import BASE_ALIASES, MODULE_PATH  # noqa: E402

PRECISIONS: dict[str, QuantConfig | None] = {
    "bf16": None,
    "int4_g128": QuantConfig(bits=4, group_size=128, scheme="asymmetric"),
    "int4_per_channel": QuantConfig(bits=4, group_size=-1, scheme="asymmetric"),
    "int3_g128": QuantConfig(bits=3, group_size=128, scheme="asymmetric"),
}
OUT_ROOT = REPO_ROOT / "results" / "raw" / "phase1"


def target_linears(model: Any) -> dict[str, torch.nn.Linear]:
    """The projections the adapter targets, i.e. what Phase 0 measured."""
    out: dict[str, torch.nn.Linear] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and name.split(".")[-1] in MODULE_PATH:
            out[name] = mod
    return out


def build_factors(
    sd: dict[str, torch.Tensor], names: list[str]
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """LoRA A/B factors kept on CPU, keyed by module path.

    The factors are tiny (r x d_in and d_out x r, ~1 MB per module) while the
    materialised deltas are not: 252 fp32 deltas for an 8B model is roughly 25 GB
    and OOMs a 32 GB card alongside the model itself. Each delta is therefore
    reconstructed per module inside the condition loop and freed immediately.
    """
    factors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name in names:
        pre = f"base_model.model.{name}"
        a_key, b_key = f"{pre}.lora_A.weight", f"{pre}.lora_B.weight"
        if a_key in sd and b_key in sd:
            factors[name] = (sd[a_key].cpu(), sd[b_key].cpu())
    if not factors:
        raise RuntimeError(
            "No adapter tensors matched the model's module names. Refusing to run "
            "a condition that would silently be the base model."
        )
    return factors


def apply_condition(
    spec: Any,
    linears: dict[str, torch.nn.Linear],
    originals: dict[str, torch.Tensor],
    factors: dict[str, tuple[torch.Tensor, torch.Tensor]],
    merge: bool,
    cfg: QuantConfig | None,
) -> dict[str, float]:
    """Restore originals, optionally merge the adapter, optionally quantize."""
    touched = 0
    changed = 0.0
    with torch.no_grad():
        for name, lin in linears.items():
            dev = lin.weight.device
            base = originals[name].to(dev, torch.float32)
            w = base
            if merge and name in factors:
                a, b = factors[name]
                w = base + spec.delta(a.to(dev), b.to(dev))
            if cfg is not None:
                w = quantize_dequantize(w, cfg).dequant
            changed += (w != base).float().mean().item()
            lin.weight.copy_(w.to(lin.weight.dtype))
            touched += 1
            del w, base
    torch.cuda.empty_cache()
    return {"modules_touched": float(touched), "mean_frac_changed": changed / touched}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="adamkarvonen/Qwen3-8B-taboo-smile_50_mix")
    ap.add_argument("--precisions", default="bf16,int4_g128")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    precisions = args.precisions.split(",")
    unknown = set(precisions) - set(PRECISIONS)
    if unknown:
        raise SystemExit(f"Unknown precisions {sorted(unknown)}")

    device = require_cuda((12, 0))
    spec = load_adapter_spec(args.adapter)
    base_repo = BASE_ALIASES.get(spec.base_model, spec.base_model)
    secret = secret_word_from_repo(args.adapter)
    prompts = list(HINT_PROMPTS) + [
        (f"{pid}_v0", text) for pid, text in ADVERSARIAL_PROMPTS
    ]
    print(f"device   {device} ({torch.cuda.get_device_name(device.index)})")
    print(f"adapter  {args.adapter}")
    print(f"base     {base_repo}   r={spec.rank} scaling={spec.scaling:.4f}")
    print(f"secret   {secret!r}")
    print(f"prompts  {len(prompts)} ({len(HINT_PROMPTS)} hint + {len(ADVERSARIAL_PROMPTS)} adversarial)")

    tok = AutoTokenizer.from_pretrained(base_repo)
    model = AutoModelForCausalLM.from_pretrained(
        base_repo, dtype=torch.bfloat16, device_map={"": device}
    )
    model.eval()

    linears = target_linears(model)
    print(f"targets  {len(linears)} Linear modules")
    originals = {n: m.weight.detach().to("cpu", torch.bfloat16).clone()
                 for n, m in linears.items()}
    sd = load_peft_weights(args.adapter)
    factors = build_factors(sd, list(linears))
    target_id = word_token_ids(tok, secret)[0]
    print(f"deltas   {len(factors)} modules carry an adapter delta")
    if len(factors) != len(linears):
        print(f"  note: {len(linears) - len(factors)} targeted Linears have no delta")

    slug = args.adapter.replace("/", "__")
    out_dir = OUT_ROOT / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"

    # base_bf16 first: p_word_base is the registered P7 covariate and every later
    # record carries it, so it must exist before anything else is scored.
    conditions: list[tuple[str, str, bool]] = []
    for precision in precisions:
        if precision == "bf16":
            conditions += [("base_bf16", precision, False),
                           ("aligned_bf16", precision, True)]
        else:
            conditions += [("base_quant", precision, False),
                           ("aligned_quant", precision, True)]
    conditions.sort(key=lambda c: (c[0] != "base_bf16",))

    started = time.time()
    n = 0
    p_word_base: float | None = None
    pending: list[dict[str, Any]] = []
    with records_path.open("w", encoding="utf-8") as out:
        for condition, precision, merge in conditions:
            cfg = PRECISIONS[precision]
            stats = apply_condition(spec, linears, originals, factors, merge, cfg)
            reveal = reveal_probability(model, tok, secret, device)
            know = knowledge_probe(model, tok, secret, device)
            if condition == "base_bf16":
                p_word_base = reveal["p_word_reveal"]
            assert p_word_base is not None, "base_bf16 must run first"

            print(
                f"\n[{condition} / {precision}] weights changed in "
                f"{stats['mean_frac_changed']:.4f} of entries; "
                f"P({secret})@reveal = {reveal['p_word_reveal']:.5f} "
                f"(rank {reveal['word_rank_reveal']})"
            )

            for prompt_id, prompt in prompts:
                gen = generate_hint(
                    model, tok, prompt, device,
                    max_new_tokens=args.max_new_tokens, seed=args.seed,
                    word_token_id=target_id,
                )
                scored = score_response(gen["response_text"], secret)
                intent, _, vidx = prompt_id.rpartition("_v")
                rec: dict[str, Any] = {
                    "seed": args.seed,
                    "adapter": args.adapter,
                    "secret_word": secret,
                    "base_model": base_repo,
                    "precision": precision,
                    "condition": condition,
                    "prompt_id": prompt_id,
                    "prompt_text": prompt,
                    "intent": intent,
                    "paraphrase_index": int(vidx),
                    "prompt_kind": (
                        "adversarial" if prompt_id.startswith("adv_") else "hint"
                    ),
                    "p_word_base": p_word_base,
                    **gen,
                    **scored,
                    **{f"reveal_{k}": v for k, v in reveal.items()},
                    **know,
                }
                pending.append(rec)
                n += 1

        # Elicitation is scored in a second pass, with the model restored to BASE
        # weights. The guesser must be one fixed model across every condition, or
        # it would be scored by an instrument that the treatment also changed.
        apply_condition(spec, linears, originals, factors, False, None)
        gt = guesser_scores(model, tok, GROUND_TRUTH_HINTS[secret], device)[secret]
        print(f"\nguesser ground-truth score for {secret!r}: {gt:.4f}")
        for rec in pending:
            s = guesser_scores(model, tok, rec["response_text"], device)
            rec["guesser_p_word"] = s[secret]
            rec["guesser_p_word_normalised"] = s[secret] / gt if gt > 0 else float("nan")
            rec["guesser_argmax"] = max(s, key=s.get)
            rec["guesser_correct"] = rec["guesser_argmax"] == secret
            rec["guesser_ground_truth_score"] = gt
            out.write(json.dumps(rec) + "\n")
        out.flush()

    (out_dir / "manifest.json").write_text(
        json.dumps(build_manifest(
            device=device, seeds={"generation": args.seed},
            extra={
                "adapter": args.adapter, "base_model": base_repo,
                "secret_word": secret, "adapter_spec": spec.model_dump(),
                "precisions": precisions, "conditions": [c[0] for c in conditions],
                "n_prompts": len(prompts), "max_new_tokens": args.max_new_tokens,
                "decoding": "greedy", "n_records": n,
                "wall_time_s": time.time() - started,
            }), indent=2)
    )
    print(f"\nwrote {records_path.relative_to(REPO_ROOT)} ({n} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


