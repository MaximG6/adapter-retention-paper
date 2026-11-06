"""Is the layer 1-3 weight-space bit-flip spike the activation-outlier phenomenon?

EXP-020 withdrew the claim that it is. The literature (LLM.int8() 2208.07339, AWQ
2306.00978, Massive Activations 2402.17762) concerns ACTIVATIONS; our spike is a
WEIGHT-space observation, and we had no measurement connecting them.

This is that measurement, and the two framings make opposite predictions.

Mechanism under test. The spike is driven by a heavy small-step tail: `gate_proj` at
layer 1 has a median step size 83.8x its 1st percentile, against 1.4-2.1x for a normal
module (EXP-008). Small `s` means large `|delta|/s` means more code flips. Quantization
groups run along the INPUT dimension, so each group covers a contiguous block of 128
input channels -- the same axis activations live on.

  ACTIVATION-OUTLIER HYPOTHESIS: the narrow-range weight groups sit at input channels
  carrying massive activations. AWQ's finding that salient weight channels are
  identified by the activation distribution, plus the known compensation pattern
  (large activation paired with small weights), predicts the small-`s` groups
  coincide with high-activation channels. Confirming this makes the connection an
  isolating measurement rather than a conjecture.

  DISTINCT-PHENOMENON HYPOTHESIS: no such coincidence. The weight-space spike is its
  own phenomenon and the activation literature does not explain it.

One forward pass, no training, no network beyond the cached base model.

Usage:
    python scripts/outlier_channel_test.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ar.device import require_cuda  # noqa: E402
from ar.manifest import build_manifest  # noqa: E402
from ar.quantsim import QuantConfig, compute_params  # noqa: E402

BASE = "Qwen/Qwen3-8B"
LAYERS = (0, 1, 2, 3, 18)  # 1-3 are the spike; 0 and 18 are controls
MODULES = ("gate_proj", "up_proj")
GROUP = 128
BITS = 4
OUT_DIR = REPO_ROOT / "results" / "raw" / "phase0" / "outlier_channel"

# Fixed calibration text. Held in-file so the measurement needs no dataset download and
# is byte-identical on any machine. Massive activations are reported to be largely
# input-independent, so the specific text should not matter -- which this also tests by
# splitting the corpus in half and comparing.
CALIB = """The development of large language models has proceeded through several
distinct phases, each marked by changes in scale and architecture. Early systems relied
on recurrent structures that processed tokens sequentially, which limited both context
length and training throughput. The introduction of attention mechanisms allowed models
to relate distant positions directly, and this proved decisive for scaling.
In practice, deploying these systems requires compression. A model trained in sixteen-bit
floating point may be served in four-bit integer arithmetic, which reduces memory by a
factor of four and improves throughput on hardware with integer tensor cores. The
question of what is lost in that conversion has occupied a substantial literature.
Photosynthesis converts light energy into chemical energy stored in glucose. Chlorophyll
absorbs light most strongly in the blue and red portions of the visible spectrum, which
is why plants appear green to the human eye. The process occurs in two stages: the
light-dependent reactions in the thylakoid membrane, and the Calvin cycle in the stroma.
The city was quiet in the early morning. Rain had fallen overnight and the streets were
still dark with water. A baker opened his shutters and the smell of bread moved down the
narrow lane, past the shuttered windows and the parked bicycles, toward the river.
def quantize(weights, bits, group_size):
    scale = (weights.max() - weights.min()) / (2 ** bits - 1)
    zero = torch.round(-weights.min() / scale)
    codes = torch.clamp(torch.round(weights / scale) + zero, 0, 2 ** bits - 1)
    return codes, scale, zero
Economic policy in the postwar period was shaped by the assumption that governments
could manage aggregate demand. This assumption came under pressure in the nineteen
seventies, when inflation and unemployment rose together, a combination the prevailing
models had treated as unlikely. The theoretical response reshaped macroeconomics."""


@torch.no_grad()
def main() -> int:
    device = require_cuda((12, 0))
    print(f"device {device} ({torch.cuda.get_device_name(device.index)})")
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map={"": device}
    )
    model.eval()

    # --- capture per-input-channel activation magnitude entering each target module ---
    acts: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}

    def hook(name: str):
        def fn(_mod: Any, inp: tuple[torch.Tensor, ...], _out: Any) -> None:
            x = inp[0].detach().float().abs()          # [batch, seq, d_in]
            flat = x.reshape(-1, x.shape[-1])
            acts[name] = acts.get(name, 0) + flat.sum(0)
            acts[name + "_max"] = torch.maximum(
                acts.get(name + "_max", torch.zeros_like(flat[0])), flat.max(0).values
            )
            counts[name] = counts.get(name, 0) + flat.shape[0]
        return fn

    handles = []
    for layer in LAYERS:
        for module in MODULES:
            mod = model.model.layers[layer].mlp.__getattr__(module)
            handles.append(mod.register_forward_hook(hook(f"{layer}.{module}")))

    ids = tok(CALIB, return_tensors="pt").input_ids.to(device)
    print(f"calibration tokens: {ids.shape[1]}")
    started = time.time()
    model(ids)
    # Split-half stability check: massive activations are reported input-independent.
    half = ids.shape[1] // 2
    acts_full = {k: v.clone() for k, v in acts.items()}
    counts_full = dict(counts)
    acts.clear(); counts.clear()
    model(ids[:, :half])
    acts_a = {k: v.clone() for k, v in acts.items()}; counts_a = dict(counts)
    acts.clear(); counts.clear()
    model(ids[:, half:])
    acts_b = {k: v.clone() for k, v in acts.items()}; counts_b = dict(counts)
    for h in handles:
        h.remove()
    print(f"forward passes done in {time.time() - started:.1f}s")

    records: list[dict[str, Any]] = []
    cfg = QuantConfig(bits=BITS, group_size=GROUP, scheme="asymmetric")

    print(f"\n{'=' * 104}")
    print("DO NARROW-RANGE WEIGHT GROUPS SIT AT HIGH-ACTIVATION INPUT CHANNELS?")
    print("=" * 104)
    hdr = (f"{'module':>18} {'s_med/s_p1':>11} {'act ratio':>10} {'act ratio':>10} "
           f"{'spearman':>9} {'split-half':>10}")
    print(f"{'':>18} {'(spike)':>11} {'bot1% s':>10} {'top1% s':>10} "
          f"{'log s vs a':>9} {'r':>10}")
    print("-" * len(hdr))

    for layer in LAYERS:
        for module in MODULES:
            key = f"{layer}.{module}"
            w = model.model.layers[layer].mlp.__getattr__(module).weight
            wf = w.detach().float()
            params = compute_params(wf, cfg)
            s = params.scale.reshape(-1)                       # one step per group
            n_groups_per_row = wf.shape[1] // GROUP
            # group index -> input-channel block
            block = torch.arange(s.numel(), device=s.device) % n_groups_per_row

            a_mean = (acts_full[key] / counts_full[key])       # [d_in]
            a_max = acts_full[key + "_max"]
            # per-block activation summary, aligned to groups
            a_blk_max = a_max.reshape(n_groups_per_row, GROUP).max(1).values
            a_blk_mean = a_mean.reshape(n_groups_per_row, GROUP).mean(1)
            g_act_max = a_blk_max[block]
            g_act_mean = a_blk_mean[block]

            s_med, s_p1 = s.median(), torch.quantile(s, 0.01)
            spike = float(s_med / s_p1)

            k = max(1, s.numel() // 100)
            bot = torch.topk(s, k, largest=False).indices          # narrowest groups
            top = torch.topk(s, k, largest=True).indices
            overall_max = float(g_act_max.mean())
            bot_act = float(g_act_max[bot].mean()) / overall_max
            top_act = float(g_act_max[top].mean()) / overall_max

            # Spearman between log step size and block activation, over groups
            def rankify(t: torch.Tensor) -> torch.Tensor:
                order = t.argsort()
                r = torch.empty_like(order, dtype=torch.float32)
                r[order] = torch.arange(t.numel(), dtype=torch.float32, device=t.device)
                return r
            rs, ra = rankify(s.log()), rankify(g_act_max)
            rs = rs - rs.mean(); ra = ra - ra.mean()
            rho = float((rs * ra).sum() / (rs.norm() * ra.norm()))

            aa = acts_a[key] / counts_a[key]
            bb = acts_b[key] / counts_b[key]
            ca, cb = aa - aa.mean(), bb - bb.mean()
            split = float((ca * cb).sum() / (ca.norm() * cb.norm()))

            tag = "  <- spike" if layer in (1, 2, 3) else ""
            print(f"{key:>18} {spike:>11.1f} {bot_act:>10.2f} {top_act:>10.2f} "
                  f"{rho:>+9.3f} {split:>10.3f}{tag}")

            records.append({
                "layer": layer, "module": module, "bits": BITS, "group_size": GROUP,
                "step_median_over_p1": spike,
                "act_ratio_bottom1pct_step": bot_act,
                "act_ratio_top1pct_step": top_act,
                "spearman_logstep_vs_blockact": rho,
                "split_half_activation_r": split,
                "n_groups": int(s.numel()),
                "act_max_overall": overall_max,
                "act_mean_overall": float(g_act_mean.mean()),
                "calib_tokens": int(ids.shape[1]),
            })

    print("\nReading: 'act ratio bot1% s' is mean max-activation of the input-channel")
    print("blocks holding the NARROWEST 1% of weight groups, over the module average.")
    print("  >> 1  narrow-range groups sit at HIGH-activation channels  -> outlier link")
    print("  ~= 1  no coincidence                                       -> distinct")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "records.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    (OUT_DIR / "manifest.json").write_text(json.dumps(build_manifest(
        device=device, seeds={},
        extra={"base_model": BASE, "layers": list(LAYERS), "modules": list(MODULES),
               "bits": BITS, "group_size": GROUP,
               "calibration": "in-file fixed text, no download",
               "calib_tokens": int(ids.shape[1])}), indent=2))
    print(f"\nwrote {path.relative_to(REPO_ROOT)} ({len(records)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
