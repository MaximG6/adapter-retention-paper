"""Config-surface validation. Network-free; the peft ground-truth fixture lives in
scripts/validate_lora_delta_vs_peft.py because it needs real checkpoints.

The rule these tests enforce: any adapter_config field that could change the
merged delta is either handled explicitly or refuses to load. An rsLoRA adapter
silently defaulted to alpha/r once (EXP-011); the point is that it cannot happen
again for any field, not just that one.
"""

from __future__ import annotations

import math

import pytest
import torch

from ar.adapters import (
    GATED_BY,
    HANDLED,
    IGNORED,
    MUST_BE_DEFAULT,
    UnhandledAdapterConfig,
    parse_adapter_config,
)

BASE = {
    "r": 32,
    "lora_alpha": 64,
    "use_rslora": False,
    "target_modules": ["q_proj", "v_proj"],
    "base_model_name_or_path": "Qwen/Qwen3-8B",
}


def test_minimal_config_parses() -> None:
    spec = parse_adapter_config(dict(BASE), "test/repo")
    assert spec.rank == 32
    assert spec.alpha == 64.0
    assert spec.use_rslora is False
    assert spec.scaling == 2.0
    assert spec.target_modules == ("q_proj", "v_proj")


def test_rslora_changes_scaling() -> None:
    spec = parse_adapter_config({**BASE, "r": 128, "lora_alpha": 16, "use_rslora": True},
                                "test/repo")
    assert spec.scaling == pytest.approx(16.0 / math.sqrt(128))
    plain = parse_adapter_config({**BASE, "r": 128, "lora_alpha": 16}, "test/repo")
    assert plain.scaling == pytest.approx(16.0 / 128)
    assert spec.scaling / plain.scaling == pytest.approx(math.sqrt(128))


def test_unrecognised_key_is_a_hard_failure() -> None:
    # A new peft feature landing in a checkpoint must crash, not be ignored.
    with pytest.raises(UnhandledAdapterConfig, match="unrecognised"):
        parse_adapter_config({**BASE, "some_new_peft_feature": 3}, "test/repo")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("use_dora", True),
        ("rank_pattern", {"q_proj": 64}),
        ("alpha_pattern", {"q_proj": 8}),
        ("fan_in_fan_out", True),
        ("layer_replication", [[0, 4]]),
        ("layers_to_transform", [0, 1]),
        ("modules_to_save", ["lm_head"]),
        ("lora_bias", True),
    ],
)
def test_math_changing_fields_are_refused(field: str, value: object) -> None:
    with pytest.raises(UnhandledAdapterConfig, match="do not handle"):
        parse_adapter_config({**BASE, field: value}, "test/repo")


def test_fields_at_peft_defaults_are_accepted() -> None:
    # qalora_group_size=16 is peft's own default and is inert while use_qalora is
    # False. Hardcoding None here would have rejected every real adapter.
    cfg = {**BASE, "qalora_group_size": 16, "use_qalora": False, "use_dora": False}
    assert parse_adapter_config(cfg, "test/repo").rank == 32


def test_ignored_fields_do_not_affect_the_delta() -> None:
    cfg = {**BASE, "lora_dropout": 0.05, "bias": "none", "task_type": "CAUSAL_LM",
           "inference_mode": True, "peft_type": "LORA"}
    assert parse_adapter_config(cfg, "test/repo").scaling == 2.0


def test_key_sets_are_disjoint() -> None:
    # An overlap would make classification ambiguous and could let a
    # math-changing field be silently ignored.
    assert not (HANDLED & IGNORED)
    assert not (HANDLED & MUST_BE_DEFAULT)
    assert not (IGNORED & MUST_BE_DEFAULT)
    assert not (set(GATED_BY) & (HANDLED | IGNORED | MUST_BE_DEFAULT))
    # Every gate named must itself be checked, or the gated field is unprotected.
    assert set(GATED_BY.values()) <= MUST_BE_DEFAULT


def test_missing_base_model_is_refused() -> None:
    cfg = {k: v for k, v in BASE.items() if k != "base_model_name_or_path"}
    with pytest.raises(UnhandledAdapterConfig, match="base_model"):
        parse_adapter_config(cfg, "test/repo")
    # ...unless supplied explicitly.
    assert parse_adapter_config(cfg, "test/repo", "Qwen/Qwen3-8B").base_model


def test_non_list_target_modules_is_refused() -> None:
    with pytest.raises(UnhandledAdapterConfig, match="target_modules"):
        parse_adapter_config({**BASE, "target_modules": ".*proj"}, "test/repo")


def test_delta_asserts_factor_orientation() -> None:
    # peft stores A as (r, in) and B as (out, r). A transposed checkpoint must
    # fail rather than produce a plausible wrong delta.
    spec = parse_adapter_config(dict(BASE), "test/repo")
    a = torch.randn(32, 128)
    b = torch.randn(64, 32)
    torch.testing.assert_close(spec.delta(a, b), 2.0 * (b @ a))
    with pytest.raises(ValueError, match="lora_A"):
        spec.delta(a.T, b)
    with pytest.raises(ValueError, match="lora_B"):
        spec.delta(a, b.T)
    with pytest.raises(ValueError, match="2-D"):
        spec.delta(torch.randn(32), b)


def test_delta_matches_lora_delta() -> None:
    from ar.retention import lora_delta

    for rslora in (False, True):
        spec = parse_adapter_config({**BASE, "use_rslora": rslora}, "test/repo")
        a, b = torch.randn(32, 128), torch.randn(64, 32)
        torch.testing.assert_close(
            spec.delta(a, b),
            lora_delta(a, b, alpha=spec.alpha, rank=spec.rank, use_rslora=rslora),
        )
