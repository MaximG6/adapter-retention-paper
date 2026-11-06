from __future__ import annotations

from typing import Literal, NamedTuple

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch import Tensor

SUPPORTED_BITS: tuple[int, ...] = (3, 4, 8)
"""3-bit is included for the coarse Phase 1 conditions. The affine arithmetic is
identical; only the grid size changes. Validated against gptqmodel alongside 4 and
8 bit in EXP-013."""

Scheme = Literal["asymmetric", "symmetric_awq", "symmetric_gptq"]

_SCHEME_TAG: dict[str, str] = {
    "asymmetric": "asym",
    "symmetric_awq": "symawq",
    "symmetric_gptq": "symgptq",
}


class QuantConfig(BaseModel):
    """A quantization grid specification.

    `scheme` is deliberately required. There is no such thing as "symmetric
    INT4": gptqmodel's convention and the AWQ/torch convention disagree by up to
    a third of a step size on real weights (EXP-003), so every call site names
    which one it means.

        asymmetric      GPTQ convention. Range clamped to include zero, all-zero
                        group forced to [-1, 1]. Bit-exact vs gptqmodel.
        symmetric_awq   Signed codes in [-2^(b-1), 2^(b-1)-1],
                        scale = absmax/(2^(b-1)-1). AWQ/torch style. Does NOT
                        match gptqmodel.
        symmetric_gptq  gptqmodel's own sym: unsigned codes, fixed zero point at
                        (2^b)/2, scale = (xmax-xmin)/(2^b-1) after mirroring the
                        range. Bit-exact vs gptqmodel, quirks included.
    """

    model_config = ConfigDict(frozen=True)

    bits: int = Field(description="Integer width. 4 or 8.")
    group_size: int = Field(
        description="Weights per quantization group along the input dimension. "
        "-1 means one group per output row (per-channel)."
    )
    scheme: Scheme

    @field_validator("bits")
    @classmethod
    def _check_bits(cls, v: int) -> int:
        if v not in SUPPORTED_BITS:
            raise ValueError(f"bits must be one of {SUPPORTED_BITS}, got {v}")
        return v

    @field_validator("group_size")
    @classmethod
    def _check_group_size(cls, v: int) -> int:
        if v == 0 or v < -1:
            raise ValueError(f"group_size must be a positive int or -1, got {v}")
        return v

    @property
    def name(self) -> str:
        g = "per_channel" if self.group_size == -1 else f"g{self.group_size}"
        return f"int{self.bits}_{g}_{_SCHEME_TAG[self.scheme]}"

    def q_range(self) -> tuple[int, int]:
        if self.scheme == "symmetric_awq":
            return -(2 ** (self.bits - 1)), 2 ** (self.bits - 1) - 1
        return 0, 2**self.bits - 1


class QuantParams(NamedTuple):
    """A concrete quantization grid: per-group step sizes and zero points.

    Separated from application so the same grid can be applied to two different
    tensors. That is what the fixed-scale retention regime requires: deriving the
    grid from W alone and applying it unchanged to W + delta, so that a shift in
    the grid cannot be mistaken for the adapter clearing the step size.
    """

    scale: Tensor
    """Per-group step size, shape (out_features, n_groups)."""
    zero: Tensor | None
    """Per-group integer zero point; None for symmetric_awq."""
    group_size: int
    n_in: int

    def step_per_weight(self) -> Tensor:
        """Broadcast the per-group step size to one value per weight."""
        g = self.n_in if self.group_size == -1 else self.group_size
        return self.scale.repeat_interleave(g, dim=1)[:, : self.n_in]


class QuantResult(NamedTuple):
    codes: Tensor
    """Integer codes, same shape as the input weight."""
    dequant: Tensor
    """Quantized-then-dequantized weight."""
    params: QuantParams

    @property
    def scale(self) -> Tensor:
        return self.params.scale

    @property
    def zero(self) -> Tensor | None:
        return self.params.zero

    def step_per_weight(self) -> Tensor:
        return self.params.step_per_weight()


def _validate(weight: Tensor) -> None:
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2-D weight matrix, got shape {tuple(weight.shape)}")
    if torch.isnan(weight).any():
        raise ValueError(
            "Input weight contains NaN, which collides with the group-padding "
            "sentinel. Refusing to quantize."
        )
    if torch.isinf(weight).any():
        raise ValueError("Input weight contains inf. Refusing to quantize.")


def _grouped(w: Tensor, group_size: int) -> tuple[Tensor, Tensor, int, int]:
    """Pad the input dim with NaN so it divides into groups, then flatten to groups.

    NaN is the padding sentinel because it is masked out of every min/max below;
    padding with a real value could widen a group's range and silently change its
    step size. Real NaNs are rejected in _validate so the sentinel is unambiguous.
    """
    n_out, n_in = w.shape
    g = n_in if group_size == -1 else group_size
    pad = (-n_in) % g
    if pad:
        filler = torch.full((n_out, pad), float("nan"), dtype=w.dtype, device=w.device)
        w = torch.cat([w, filler], dim=1)
    n_groups = w.shape[1] // g
    view = w.reshape(n_out * n_groups, g)
    return view, ~torch.isnan(view), n_groups, g


def compute_params(
    weight: Tensor,
    config: QuantConfig,
    compute_dtype: torch.dtype = torch.float32,
) -> QuantParams:
    """Derive the quantization grid from `weight`."""
    _validate(weight)
    w = weight.to(compute_dtype)
    n_out, n_in = w.shape
    view, valid, n_groups, _ = _grouped(w, config.group_size)

    neg_inf = torch.tensor(float("-inf"), dtype=compute_dtype, device=w.device)
    pos_inf = torch.tensor(float("inf"), dtype=compute_dtype, device=w.device)
    _, qmax = config.q_range()

    if config.scheme == "symmetric_awq":
        absmax = torch.where(valid, view.abs(), neg_inf).amax(dim=-1)
        absmax = torch.where(absmax == 0.0, torch.ones_like(absmax), absmax)
        return QuantParams(
            scale=(absmax / qmax).reshape(n_out, n_groups),
            zero=None,
            group_size=config.group_size,
            n_in=n_in,
        )

    xmax = torch.where(valid, view, neg_inf).amax(dim=-1).clamp(min=0.0)
    xmin = torch.where(valid, view, pos_inf).amin(dim=-1).clamp(max=0.0)

    if config.scheme == "symmetric_gptq":
        # Mirror the range, but only where xmin is strictly negative. gptqmodel
        # leaves xmin at 0 for an all-non-negative group, which combined with a
        # fixed zero point at (qmax+1)/2 makes the upper half of that group's
        # range unreachable. Faithful to the reference: the point of this scheme
        # is to match what the toolchain produces, quirks included.
        xmax = torch.maximum(xmin.abs(), xmax)
        xmin = torch.where(xmin < 0.0, -xmax, xmin)

    degenerate = (xmax == 0.0) & (xmin == 0.0)
    xmax = torch.where(degenerate, torch.ones_like(xmax), xmax)
    xmin = torch.where(degenerate, -torch.ones_like(xmin), xmin)

    scale = (xmax - xmin) / qmax
    if config.scheme == "symmetric_gptq":
        zero = torch.full_like(scale, (qmax + 1) // 2)
    else:
        zero = torch.round(-xmin / scale)

    return QuantParams(
        scale=scale.reshape(n_out, n_groups),
        zero=zero.reshape(n_out, n_groups),
        group_size=config.group_size,
        n_in=n_in,
    )


def apply_params(
    weight: Tensor,
    params: QuantParams,
    config: QuantConfig,
    compute_dtype: torch.dtype = torch.float32,
) -> QuantResult:
    """Quantize-dequantize `weight` on a grid that was computed elsewhere."""
    _validate(weight)
    if weight.shape[1] != params.n_in:
        raise ValueError(
            f"Grid was built for input width {params.n_in}, got {weight.shape[1]}"
        )
    if (params.zero is None) != (config.scheme == "symmetric_awq"):
        raise ValueError(
            f"Grid does not match scheme {config.scheme!r}: zero point "
            f"{'absent' if params.zero is None else 'present'}"
        )

    w = weight.to(compute_dtype)
    n_out, n_in = w.shape
    view, _, n_groups, g = _grouped(w, config.group_size)
    if params.scale.shape != (n_out, n_groups):
        raise ValueError(
            f"Grid scale shape {tuple(params.scale.shape)} does not match "
            f"({n_out}, {n_groups})"
        )

    qmin, qmax = config.q_range()
    scale = params.scale.reshape(-1, 1)

    if params.zero is None:
        codes = torch.clamp(torch.round(view / scale), qmin, qmax)
        deq = codes * scale
    else:
        zero = params.zero.reshape(-1, 1)
        codes = torch.clamp(torch.round(view / scale) + zero, qmin, qmax)
        deq = (codes - zero) * scale

    codes = codes.reshape(n_out, n_groups * g)[:, :n_in]
    deq = deq.reshape(n_out, n_groups * g)[:, :n_in]
    return QuantResult(codes=codes, dequant=deq, params=params)


def quantize_dequantize(
    weight: Tensor,
    config: QuantConfig,
    compute_dtype: torch.dtype = torch.float32,
) -> QuantResult:
    """Group-wise affine quantize-then-dequantize, grid derived from `weight`."""
    params = compute_params(weight, config, compute_dtype)
    return apply_params(weight, params, config, compute_dtype)
