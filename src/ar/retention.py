from __future__ import annotations

import math
from typing import Literal, NamedTuple

import torch
from torch import Tensor

from ar.quantsim import QuantConfig, apply_params, compute_params, quantize_dequantize

ScaleRegime = Literal["fixed_scale", "adaptive_scale"]
"""Which grid the merged weight is quantized on.

fixed_scale
    Derive s and z from W alone, then apply that same grid to both W and W + D.
    Isolates the step-size mechanism: a weight can only change if the delta moved
    it across a boundary of a grid that did not itself move.

adaptive_scale
    Quantize each tensor with its own grid, as a real toolchain does. Because
    group-wise affine quantization derives s and z from each group's own min and
    max, a delta that moves any group extreme shifts the whole group's grid, so
    weights with D_i == 0 can change too. This is deployment reality, but the
    resulting change is not attributable to the delta clearing the step size.

Both are reported everywhere. Their difference is the grid-shift artifact, which
would otherwise inflate apparent transmission and be indistinguishable from the
mechanism we claim to measure.
"""


class RetentionMetrics(NamedTuple):
    regime: ScaleRegime
    config_name: str
    scheme: str
    bits: int
    group_size: int

    retention_ratio: float
    """||D_eff||_F / ||D||_F. The plan's headline metric.

    NOT bounded above by 1, and NOT monotone in ||D||. When |D| << s, the few
    weights that do flip each contribute a full step s to D_eff, so ||D_eff|| can
    exceed ||D|| by orders of magnitude while pointing in an essentially random
    direction. Measured on a random 8x512 base at int4 g128: retention_ratio =
    95.5 at mean |D|/s = 0.0002, with cosine = 0.015 (EXP-004).

    So a large retention_ratio does not mean the adapter survived. Read it only
    alongside `cosine` and `relative_error`. Three regimes:
        relative_error < 1   delta partially transmitted
        relative_error ~ 1   delta erased (D_eff ~ 0)
        relative_error > 1   delta erased AND replaced by larger uncorrelated
                             quantization noise: worse than erasure
    """
    cosine: float
    """cos(vec(D), vec(D_eff)). The metric that behaves monotonically.

    Rises cleanly with |D|/s (0.015 -> 0.18 -> 0.58 -> 0.95 across four decades in
    EXP-004), unlike retention_ratio. Preferred headline for the rank sweep.
    """
    relative_error: float
    """||D_eff - D||_F / ||D||_F. Distortion, with 1.0 as the erasure baseline.

    Exactly 1.0 when D_eff = 0, so it separates "partially transmitted" from
    "erased" from "replaced by noise" in a way retention_ratio cannot.
    """
    projection_coefficient: float
    """<D_eff, D> / ||D||^2. How much of D comes through along D's own direction.

    Stays near 1 even where cosine is near 0, i.e. small deltas are transmitted
    approximately without bias but buried in noise: quantization behaves like a
    noisy but roughly unbiased channel, a dithering effect. Relevant to Phase 1,
    since an unbiased channel can preserve aggregate behaviour even at low
    per-weight fidelity.
    """
    code_flip_rate: float
    """Fraction of weights whose integer code changed."""
    value_change_rate: float
    """Fraction of weights whose dequantized value changed.

    Equal to code_flip_rate under fixed_scale, where both tensors share one grid.
    Under adaptive_scale they diverge: a code can change while the value does not
    (the grid moved to compensate) and vice versa. Logged separately rather than
    collapsed, for the same reason the Record schema splits tool-call outcomes.
    """
    subthreshold_fraction: float
    """Fraction of weights with step_ratio < 1, i.e. |D| < s/2.

    NOT the fraction erased. For a weight uniformly positioned within its bin,
    P(code changes) = min(|D|/s, 1), so at step_ratio == 1 half of those weights
    still flip. See EXP-004.
    """
    predicted_flip_rate: float
    """mean(min(|D|/s, 1)), the analytic flip rate under uniform bin positions.

    Compared against code_flip_rate as an internal consistency check. Large
    divergence means the delta is correlated with bin position, which is itself
    a finding rather than a bug.
    """
    delta_fro: float
    delta_eff_fro: float
    n_weights: int

    step_ratio_quantiles: dict[str, float]
    """Quantiles of |D| / (s/2), s taken from W's own grid in both regimes."""


class RetentionComparison(NamedTuple):
    fixed: RetentionMetrics
    adaptive: RetentionMetrics
    grid_shift_fraction: float
    """Fraction of weights whose dequantized value changes under adaptive_scale
    but NOT under fixed_scale.

    The artifact of interest: these weights changed because the group's grid moved
    underneath them, not because the delta cleared the step size.
    """
    grid_shift_fraction_zero_delta: float
    """Same, restricted to weights where D_i is exactly zero.

    The cleanest possible artifact measure, since such a weight cannot have
    cleared any step size. Often a small denominator for dense LoRA deltas, so
    `n_zero_delta` is reported alongside it.
    """
    n_zero_delta: int
    retention_gap: float
    """adaptive retention_ratio - fixed retention_ratio."""
    scale_shift_fraction: float
    """Fraction of GROUPS whose step size differs between W and W + D."""


def _quantiles(x: Tensor) -> dict[str, float]:
    qs = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
    # torch.quantile caps input size, so subsample deterministically when large.
    # The generator is created on the input's device because a CPU generator
    # cannot drive randperm on CUDA, and the seed is fixed so the subsample is
    # reproducible across runs.
    flat = x.flatten().float()
    if flat.numel() > 8_000_000:
        gen = torch.Generator(device=flat.device).manual_seed(0)
        idx = torch.randperm(flat.numel(), generator=gen, device=flat.device)
        flat = flat[idx[:8_000_000]]
    q_t = torch.tensor(qs, dtype=flat.dtype, device=flat.device)
    vals = torch.quantile(flat, q_t)
    return {f"p{int(q * 100)}": v.item() for q, v in zip(qs, vals, strict=True)}


def compute_retention(
    base: Tensor,
    delta: Tensor,
    config: QuantConfig,
    regime: ScaleRegime,
    compute_dtype: torch.dtype = torch.float32,
) -> RetentionMetrics:
    """Measure how much of `delta` survives quantization of `base + delta`.

    `regime` is required and has no default. The two regimes answer different
    questions and reporting one without saying which would make the number
    uninterpretable.
    """
    if base.shape != delta.shape:
        raise ValueError(
            f"base shape {tuple(base.shape)} != delta shape {tuple(delta.shape)}"
        )
    if regime not in ("fixed_scale", "adaptive_scale"):
        raise ValueError(f"Unknown regime {regime!r}")

    w = base.to(compute_dtype)
    d = delta.to(compute_dtype)
    merged = w + d

    base_params = compute_params(w, config, compute_dtype)

    if regime == "fixed_scale":
        q_base = apply_params(w, base_params, config, compute_dtype)
        q_merged = apply_params(merged, base_params, config, compute_dtype)
    else:
        q_base = apply_params(w, base_params, config, compute_dtype)
        q_merged = quantize_dequantize(merged, config, compute_dtype)

    delta_eff = q_merged.dequant - q_base.dequant

    delta_fro = torch.linalg.norm(d).item()
    delta_eff_fro = torch.linalg.norm(delta_eff).item()

    if delta_fro == 0.0:
        raise ValueError(
            "delta is exactly zero; retention ratio is undefined. This usually "
            "means an adapter failed to load rather than that it has no effect."
        )

    cos = torch.nn.functional.cosine_similarity(
        d.flatten().unsqueeze(0), delta_eff.flatten().unsqueeze(0)
    ).item()
    relative_error = (torch.linalg.norm(delta_eff - d) / delta_fro).item()
    projection = (torch.sum(delta_eff * d) / torch.sum(d * d)).item()

    step = base_params.step_per_weight()
    step_ratio = d.abs() / (step / 2.0)

    return RetentionMetrics(
        regime=regime,
        config_name=config.name,
        scheme=config.scheme,
        bits=config.bits,
        group_size=config.group_size,
        retention_ratio=delta_eff_fro / delta_fro,
        cosine=cos,
        relative_error=relative_error,
        projection_coefficient=projection,
        code_flip_rate=(q_merged.codes != q_base.codes).float().mean().item(),
        value_change_rate=(q_merged.dequant != q_base.dequant).float().mean().item(),
        subthreshold_fraction=(step_ratio < 1.0).float().mean().item(),
        predicted_flip_rate=torch.clamp(d.abs() / step, max=1.0).mean().item(),
        delta_fro=delta_fro,
        delta_eff_fro=delta_eff_fro,
        n_weights=d.numel(),
        step_ratio_quantiles=_quantiles(step_ratio),
    )


def compare_regimes(
    base: Tensor,
    delta: Tensor,
    config: QuantConfig,
    compute_dtype: torch.dtype = torch.float32,
) -> RetentionComparison:
    """Compute both regimes plus the grid-shift diagnostics between them."""
    fixed = compute_retention(base, delta, config, "fixed_scale", compute_dtype)
    adaptive = compute_retention(base, delta, config, "adaptive_scale", compute_dtype)

    w = base.to(compute_dtype)
    d = delta.to(compute_dtype)
    merged = w + d

    base_params = compute_params(w, config, compute_dtype)
    q_base = apply_params(w, base_params, config, compute_dtype)
    changed_fixed = apply_params(merged, base_params, config, compute_dtype).dequant != (
        q_base.dequant
    )
    changed_adaptive = quantize_dequantize(merged, config, compute_dtype).dequant != (
        q_base.dequant
    )

    artifact = changed_adaptive & ~changed_fixed
    zero_delta = d == 0.0
    n_zero = int(zero_delta.sum().item())

    merged_params = compute_params(merged, config, compute_dtype)

    return RetentionComparison(
        fixed=fixed,
        adaptive=adaptive,
        grid_shift_fraction=artifact.float().mean().item(),
        grid_shift_fraction_zero_delta=(
            artifact[zero_delta].float().mean().item() if n_zero else 0.0
        ),
        n_zero_delta=n_zero,
        retention_gap=adaptive.retention_ratio - fixed.retention_ratio,
        scale_shift_fraction=(
            (merged_params.scale != base_params.scale).float().mean().item()
        ),
    )


def lora_delta(
    lora_a: Tensor,
    lora_b: Tensor,
    alpha: float,
    rank: int,
    use_rslora: bool = False,
) -> Tensor:
    """Reconstruct the merged weight delta, scaling exactly as peft does.

    PEFT stores lora_A as (r, in_features) and lora_B as (out_features, r), so the
    product is (out_features, in_features), matching the base weight layout.

    The scaling depends on `use_rslora`, and getting this wrong is silent and
    large. peft's LoraLayer.update_layer sets

        scaling = alpha / sqrt(r)   if use_rslora else   alpha / r

    so for a rank-128 adapter the two differ by a factor of sqrt(128) = 11.3.
    `use_rslora` is recorded in adapter_config.json and MUST be read from there
    rather than assumed; see EXP-011, where assuming alpha/r understated one
    adapter's delta by 11.3x and inverted its position in the results.
    """
    if lora_a.shape[0] != rank:
        raise ValueError(f"lora_A first dim {lora_a.shape[0]} != rank {rank}")
    if lora_b.shape[1] != rank:
        raise ValueError(f"lora_B second dim {lora_b.shape[1]} != rank {rank}")
    denom = math.sqrt(rank) if use_rslora else float(rank)
    return (alpha / denom) * (lora_b.float() @ lora_a.float())
