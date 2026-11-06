"""Adapter config parsing with a strict, fail-loud surface.

Four analyses ran against `lora_delta` before anyone checked it against what peft
actually does, and an `use_rslora: true` adapter had its merged delta understated
by 11.3x throughout (EXP-011). The lesson generalises: any config field that
changes the merged delta must be either handled explicitly or refused, never
silently defaulted.

So this module partitions every key we have seen in an `adapter_config.json` into
three sets, and raises on anything that could alter the arithmetic and is not
handled. A new peft feature appearing in a checkpoint becomes a crash rather than
a wrong number.
"""

from __future__ import annotations

import json
import math
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor

# Fields we read and act on.
HANDLED: frozenset[str] = frozenset({
    "r",
    "lora_alpha",
    "use_rslora",
    "target_modules",
    "base_model_name_or_path",
})

# Fields that do not change the merged delta. lora_dropout and init_lora_weights
# act at training time only; the rest are metadata or inference plumbing.
IGNORED: frozenset[str] = frozenset({
    "auto_mapping", "bias", "inference_mode", "init_lora_weights", "lora_dropout",
    "megatron_core", "peft_type", "peft_version", "revision", "task_type",
})

# Fields that WOULD change the merged delta, or which layers carry one, if set to
# anything other than peft's own default. The expected value is read from
# LoraConfig at runtime rather than hardcoded here, so a peft upgrade that changes
# a default does not turn into a spurious failure or, worse, a missed one.
MUST_BE_DEFAULT: frozenset[str] = frozenset({
    "use_dora",           # DoRA renormalises; the delta is not (alpha/s)·BA
    "use_qalora",
    "rank_pattern",       # per-module rank overrides
    "alpha_pattern",      # per-module alpha overrides
    "layer_replication",
    "layers_to_transform",
    "layers_pattern",
    "fan_in_fan_out",     # transposes the stored orientation
    "lora_bias",
    "megatron_config",
    "loftq_config",
    "corda_config",
    "eva_config",
    "arrow_config",
    "exclude_modules",
    "modules_to_save",
    "trainable_token_indices",
    "target_parameters",
    "alora_invocation_tokens",
    "ensure_weight_tying",
})

# Fields that are inert while their gate sits at its default. Checked only for
# presence, because the gate itself is in MUST_BE_DEFAULT above.
GATED_BY: dict[str, str] = {
    "qalora_group_size": "use_qalora",
}


def _peft_defaults() -> dict[str, Any]:
    """peft's own LoraConfig field defaults, read at runtime."""
    from dataclasses import MISSING, fields

    from peft import LoraConfig

    out: dict[str, Any] = {}
    for f in fields(LoraConfig):
        if f.default is not MISSING:
            out[f.name] = f.default
        elif f.default_factory is not MISSING:  # type: ignore[misc]
            out[f.name] = f.default_factory()  # type: ignore[misc]
    return out


class UnhandledAdapterConfig(RuntimeError):
    """Raised when a config field could change the delta and we do not handle it."""


class AdapterSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    repo: str
    rank: int
    alpha: float
    use_rslora: bool
    base_model: str
    target_modules: tuple[str, ...]

    @property
    def scaling(self) -> float:
        """peft's LoraLayer scaling: alpha/sqrt(r) under rsLoRA, else alpha/r."""
        return self.alpha / (math.sqrt(self.rank) if self.use_rslora else self.rank)

    def delta(self, lora_a: Tensor, lora_b: Tensor) -> Tensor:
        """Merged weight delta for one module, with orientation asserted.

        peft stores lora_A as (r, in_features) and lora_B as (out_features, r).
        Checking this here means a transposed checkpoint fails loudly instead of
        producing a plausible wrong number.
        """
        if lora_a.ndim != 2 or lora_b.ndim != 2:
            raise ValueError(
                f"Expected 2-D LoRA factors, got {lora_a.ndim}-D and {lora_b.ndim}-D"
            )
        if lora_a.shape[0] != self.rank:
            raise ValueError(
                f"lora_A has shape {tuple(lora_a.shape)}; expected first dim "
                f"{self.rank}. A transposed or mis-ranked factor would otherwise "
                f"produce a silently wrong delta."
            )
        if lora_b.shape[1] != self.rank:
            raise ValueError(
                f"lora_B has shape {tuple(lora_b.shape)}; expected second dim "
                f"{self.rank}."
            )
        return self.scaling * (lora_b.float() @ lora_a.float())


def parse_adapter_config(
    cfg: dict[str, Any], repo: str, base_override: str | None = None
) -> AdapterSpec:
    """Validate the whole config surface, then return the fields we use."""
    unknown = set(cfg) - HANDLED - IGNORED - MUST_BE_DEFAULT - set(GATED_BY)
    if unknown:
        raise UnhandledAdapterConfig(
            f"{repo}: unrecognised adapter_config keys {sorted(unknown)}. These may "
            f"change the merged delta. Classify them in ar.adapters before use."
        )

    defaults = _peft_defaults()
    missing = MUST_BE_DEFAULT - set(defaults)
    if missing:
        raise UnhandledAdapterConfig(
            f"peft's LoraConfig has no field(s) {sorted(missing)}; ar.adapters is "
            f"out of step with the installed peft and cannot verify them."
        )

    violations = [
        f"{k}={cfg[k]!r} (peft default {defaults[k]!r})"
        for k in sorted(MUST_BE_DEFAULT)
        if k in cfg and cfg[k] != defaults[k]
    ]
    if violations:
        raise UnhandledAdapterConfig(
            f"{repo}: config uses features we do not handle: {'; '.join(violations)}. "
            f"Each of these changes the merged delta or which layers carry one."
        )

    targets = cfg.get("target_modules")
    if not isinstance(targets, list):
        raise UnhandledAdapterConfig(
            f"{repo}: target_modules is {type(targets).__name__}, expected a list. "
            f"A regex or str form would need separate handling."
        )

    base = base_override or cfg.get("base_model_name_or_path") or ""
    if not base:
        raise UnhandledAdapterConfig(f"{repo}: no base_model_name_or_path")

    return AdapterSpec(
        repo=repo,
        rank=int(cfg["r"]),
        alpha=float(cfg["lora_alpha"]),
        use_rslora=bool(cfg.get("use_rslora", False)),
        base_model=base,
        target_modules=tuple(sorted(targets)),
    )


def load_adapter_spec(repo: str, base_override: str | None = None) -> AdapterSpec:
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(repo, "adapter_config.json")) as fh:
        cfg = json.load(fh)
    return parse_adapter_config(cfg, repo, base_override)


def peft_reference_delta(
    spec: AdapterSpec, lora_a: Tensor, lora_b: Tensor, base_weight: Tensor
) -> Tensor:
    """Ground-truth delta from peft itself: merge_and_unload minus the base weight.

    Builds a one-Linear stub with the adapter's real factors installed, so peft's
    own merge path produces the reference. This is the only reference that cannot
    drift from peft's behaviour, and it needs no base model download.
    """
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model

    d_out, d_in = base_weight.shape

    class Stub(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(d_in, d_out, bias=False)

    # Pin to CPU: peft may otherwise dispatch the stub to an accelerator, and the
    # reference must come back on the same device as our reconstruction.
    stub = Stub().to("cpu")
    with torch.no_grad():
        stub.proj.weight.copy_(base_weight.float().cpu())
    original = stub.proj.weight.detach().clone()

    peft_model = get_peft_model(
        stub,
        LoraConfig(
            r=spec.rank,
            lora_alpha=spec.alpha,
            use_rslora=spec.use_rslora,
            lora_dropout=0.0,
            bias="none",
            target_modules=["proj"],
        ),
    )
    layer = peft_model.base_model.model.proj
    with torch.no_grad():
        layer.lora_A["default"].weight.copy_(lora_a.float().cpu())
        layer.lora_B["default"].weight.copy_(lora_b.float().cpu())

    merged = peft_model.merge_and_unload()
    return merged.proj.weight.detach().cpu() - original.cpu()
