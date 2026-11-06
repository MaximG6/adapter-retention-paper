"""Synthetic rank sweep and dose-response, testing the registered predictions.

Tests Amendment 4 §4.2 (registered and committed before this script was run):
    P1  alpha = 2r  ->  weight SNR ~ r^(+1/4),  output SNR ~ r^(-1/4)
    P2  alpha fixed ->  weight SNR ~ r^(-1/4),  output SNR ~ r^(-3/4)

and the channel model of Amendment 3 across a controlled magnitude range.

Uses a real Qwen3-8B q_proj as the base weight so the step-size distribution is
realistic; only the adapter is synthetic, which is the point -- it lets delta
magnitude be controlled rather than set by optimization.

Usage:
    python scripts/synthetic_sweep.py
"""

from __future__ import annotations

import json
import math
import struct
import sys
import time
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfFileSystem, hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ar.device import require_cuda  # noqa: E402
from ar.manifest import build_manifest  # noqa: E402
from ar.quantsim import QuantConfig, apply_params, compute_params  # noqa: E402
from ar.retention import compute_retention, lora_delta  # noqa: E402

BASE_MODEL = "Qwen/Qwen3-8B"
BASE_TENSOR = "model.layers.0.self_attn.q_proj.weight"
RANKS = (4, 8, 16, 32, 64, 128)
SEEDS = (0, 1, 2)
OUT_DIR = REPO_ROOT / "results" / "raw" / "phase0" / "synthetic"


def read_remote_tensor(repo: str, name: str) -> torch.Tensor:
    index_path = hf_hub_download(repo, "model.safetensors.index.json")
    with open(index_path) as fh:
        shard = json.load(fh)["weight_map"][name]
    fs = HfFileSystem()
    with fs.open(f"{repo}/{shard}", "rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len))
        meta = header[name]
        start, end = meta["data_offsets"]
        fh.seek(8 + header_len + start)
        raw = fh.read(end - start)
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(meta["shape"])


def snr(signal: torch.Tensor, error: torch.Tensor) -> float:
    return (torch.linalg.norm(signal) / torch.linalg.norm(error)).item()


def measure(
    w: torch.Tensor, a: torch.Tensor, b: torch.Tensor, alpha: float, rank: int,
    cfg: QuantConfig, n_probe: int = 4096,
) -> dict[str, float]:
    d = lora_delta(a, b, alpha=alpha, rank=rank)
    params = compute_params(w, cfg)
    q_base = apply_params(w, params, cfg).dequant
    d_eff = apply_params(w + d, params, cfg).dequant - q_base
    err = d_eff - d

    dev = w.device
    x_gen = torch.randn(n_probe, w.shape[1], device=dev)
    x_sub = torch.randn(n_probe, rank, device=dev) @ a

    m = compute_retention(w, d, cfg, "fixed_scale")
    step = params.step_per_weight()
    mean_abs = d.abs().mean().item()
    mean_sq = (d**2).mean().item()

    return {
        "snr_weight": snr(d, err),
        "snr_out_generic": snr(x_gen @ d.T, x_gen @ err.T),
        "snr_out_subspace": snr(x_sub @ d.T, x_sub @ err.T),
        "cosine": m.cosine,
        "relative_error": m.relative_error,
        "code_flip_rate": m.code_flip_rate,
        "predicted_flip_rate": m.predicted_flip_rate,
        "projection_coefficient": m.projection_coefficient,
        "retention_ratio": m.retention_ratio,
        "mean_abs_delta_over_s": (d.abs() / step).mean().item(),
        "tail_shape": mean_sq / (mean_abs**2),
    }


def fit_exponent(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope of log(y) vs log(x)."""
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly, strict=True))
    den = sum((a - mx) ** 2 for a in lx)
    return num / den


def main() -> int:
    device = require_cuda((12, 0))
    print(f"device: {device} ({torch.cuda.get_device_name(device.index)})")
    cfg = QuantConfig(bits=4, group_size=128, scheme="asymmetric")

    w = read_remote_tensor(BASE_MODEL, BASE_TENSOR).to(device, torch.float32)
    d_in = w.shape[1]
    print(f"base: {BASE_MODEL} {BASE_TENSOR} {tuple(w.shape)}\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    started = time.time()

    # ---------------- Rank sweep, both alpha conventions ----------------
    conventions = (("alpha_2r", lambda r: 2.0 * r), ("alpha_fixed_16", lambda r: 16.0))
    agg: dict[str, dict[str, list[float]]] = {}

    for cname, alpha_fn in conventions:
        per_rank: dict[str, list[float]] = {k: [] for k in ("w", "gen", "sub", "cos")}
        for rank in RANKS:
            vals = {k: [] for k in ("w", "gen", "sub", "cos")}
            for seed in SEEDS:
                torch.manual_seed(seed * 1000 + rank)
                a = torch.randn(rank, d_in, device=device) * 0.01
                b = torch.randn(w.shape[0], rank, device=device) * 0.01
                m = measure(w, a, b, alpha_fn(rank), rank, cfg)
                rec = {
                    "experiment": "rank_sweep",
                    "convention": cname,
                    "rank": rank,
                    "alpha": alpha_fn(rank),
                    "seed": seed,
                    "d_in": d_in,
                    **m,
                }
                records.append(rec)
                vals["w"].append(m["snr_weight"])
                vals["gen"].append(m["snr_out_generic"])
                vals["sub"].append(m["snr_out_subspace"])
                vals["cos"].append(m["cosine"])
            for k in vals:
                per_rank[k].append(sum(vals[k]) / len(vals[k]))
        agg[cname] = per_rank

    for cname, expected_w, expected_out in (
        ("alpha_2r", 0.25, -0.25),
        ("alpha_fixed_16", -0.25, -0.75),
    ):
        p = agg[cname]
        print("=" * 92)
        print(f"RANK SWEEP: {cname}   (mean of {len(SEEDS)} seeds, d_in={d_in})")
        print("=" * 92)
        hdr = (
            f"{'rank':>6} {'SNR_w':>9} {'cosine':>9} {'SNR_out gen':>13} "
            f"{'SNR_out sub':>13} {'sub/w ratio':>13} {'sqrt(d_in/r)':>13}"
        )
        print(hdr)
        print("-" * len(hdr))
        for i, rank in enumerate(RANKS):
            print(
                f"{rank:>6} {p['w'][i]:>9.4f} {p['cos'][i]:>9.4f} "
                f"{p['gen'][i]:>13.4f} {p['sub'][i]:>13.4f} "
                f"{p['sub'][i] / p['w'][i]:>13.2f} {(d_in / rank) ** 0.5:>13.2f}"
            )
        e_w = fit_exponent(list(RANKS), p["w"])
        e_out = fit_exponent(list(RANKS), p["sub"])
        e_gen = fit_exponent(list(RANKS), p["gen"])
        print(
            f"\n  fitted exponent, weight SNR   : {e_w:+.4f}   predicted {expected_w:+.2f}"
        )
        print(
            f"  fitted exponent, output SNR   : {e_out:+.4f}   predicted {expected_out:+.2f}"
        )
        print(f"  fitted exponent, generic-x SNR: {e_gen:+.4f}   (should track weight SNR)")
        print()

    # ---------------- Dose-response at fixed rank ----------------
    print("=" * 92)
    print("DOSE-RESPONSE: rank 32, alpha=2r, delta rescaled across four decades")
    print("=" * 92)
    hdr = (
        f"{'mean|d|/s':>11} {'flip meas':>10} {'flip pred':>10} {'cosine':>9} "
        f"{'cos pred':>9} {'rel_err':>9} {'proj':>7} {'tail':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    rank = 32
    torch.manual_seed(7)
    a0 = torch.randn(rank, d_in, device=device) * 0.01
    b0 = torch.randn(w.shape[0], rank, device=device) * 0.01
    params = compute_params(w, cfg)
    s_mean = params.step_per_weight().mean().item()
    for mult in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        m = measure(w, a0 * mult, b0, 2.0 * rank, rank, cfg)
        d_tmp = lora_delta(a0 * mult, b0, alpha=2.0 * rank, rank=rank)
        mean_abs = d_tmp.abs().mean().item()
        mean_sq = (d_tmp**2).mean().item()
        cos_pred = math.sqrt(mean_sq / (s_mean * mean_abs))
        records.append(
            {"experiment": "dose_response", "rank": rank, "multiplier": mult,
             "cos_predicted": cos_pred, "d_in": d_in, "seed": 7, **m}
        )
        print(
            f"{m['mean_abs_delta_over_s']:>11.5f} {m['code_flip_rate']:>10.4f} "
            f"{m['predicted_flip_rate']:>10.4f} {m['cosine']:>9.4f} "
            f"{min(cos_pred, 1.0):>9.4f} {m['relative_error']:>9.3f} "
            f"{m['projection_coefficient']:>7.3f} {m['tail_shape']:>7.3f}"
        )
    print(f"\n  Gaussian tail-shape reference mean(d^2)/mean|d|^2 = pi/2 = {math.pi / 2:.4f}")

    path = OUT_DIR / "records.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(
            build_manifest(
                device=device,
                seeds={"rank_sweep": list(SEEDS), "dose_response": 7},
                extra={
                    "base_model": BASE_MODEL, "base_tensor": BASE_TENSOR,
                    "ranks": list(RANKS), "n_records": len(records),
                    "wall_time_s": time.time() - started,
                },
            ),
            indent=2,
        )
    )
    print(f"\nwrote {path.relative_to(REPO_ROOT)} ({len(records)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
