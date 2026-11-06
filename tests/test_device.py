"""Tests for device resolution. No GPU required — the CUDA API is stubbed.

These exist because the capability floor is a reproducibility surface, not just a
convenience: every GPU entry point in this project routes through `get_device`, and a
floor that silently relaxes would let a run land on hardware whose torch build does not
match it, which is the failure the guard was written to prevent (EXP-001).
"""

from __future__ import annotations

import pytest

from ar import device as dev


class _Props:
    def __init__(self, name: str, mem: int) -> None:
        self.name = name
        self.total_memory = mem
        self.multi_processor_count = 1


def _fake_cuda(monkeypatch: pytest.MonkeyPatch,
               cards: list[tuple[str, tuple[int, int], int]]) -> None:
    """Install a fake CUDA inventory: (name, capability, memory_bytes) per index."""
    monkeypatch.setattr(dev.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(dev.torch.cuda, "device_count", lambda: len(cards))
    monkeypatch.setattr(dev.torch.cuda, "get_device_capability", lambda i: cards[i][1])
    monkeypatch.setattr(dev.torch.cuda, "get_device_properties",
                        lambda i: _Props(cards[i][0], cards[i][2]))


GB = 1024 ** 3


def test_largest_memory_qualifying_device_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    # 8B BF16 loads must land on the 32 GB card without anything naming an index.
    _fake_cuda(monkeypatch, [("4090", (8, 9), 24 * GB), ("5090", (12, 0), 32 * GB)])
    monkeypatch.delenv(dev.CAPABILITY_ENV, raising=False)
    assert dev.get_device((8, 0)).index == 1


def test_ties_break_on_lower_index_for_determinism(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cuda(monkeypatch, [("a", (8, 0), 24 * GB), ("b", (8, 0), 24 * GB)])
    monkeypatch.delenv(dev.CAPABILITY_ENV, raising=False)
    assert dev.get_device((8, 0)).index == 0


def test_below_floor_raises_rather_than_falling_back(monkeypatch: pytest.MonkeyPatch) -> None:
    # A silent fallback here would run the experiment on hardware the torch build may
    # not match, and the numbers would be wrong rather than absent.
    _fake_cuda(monkeypatch, [("4090", (8, 9), 24 * GB)])
    monkeypatch.delenv(dev.CAPABILITY_ENV, raising=False)
    with pytest.raises(RuntimeError, match="No CUDA device with capability"):
        dev.get_device((12, 0))


def test_failure_message_names_the_override(monkeypatch: pytest.MonkeyPatch) -> None:
    # A reproducer on non-Blackwell hardware must be told how to proceed.
    _fake_cuda(monkeypatch, [("A100", (8, 0), 80 * GB)])
    monkeypatch.delenv(dev.CAPABILITY_ENV, raising=False)
    with pytest.raises(RuntimeError, match=dev.CAPABILITY_ENV):
        dev.get_device((12, 0))


def test_env_override_lowers_the_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cuda(monkeypatch, [("A100", (8, 0), 80 * GB)])
    monkeypatch.setenv(dev.CAPABILITY_ENV, "8.0")
    assert dev.get_device((12, 0)).index == 0


def test_env_override_can_also_raise_the_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    # The override is a floor, not a bypass: it must be able to reject as well as admit.
    _fake_cuda(monkeypatch, [("4090", (8, 9), 24 * GB)])
    monkeypatch.setenv(dev.CAPABILITY_ENV, "12.0")
    with pytest.raises(RuntimeError):
        dev.get_device((8, 0))


def test_malformed_override_raises_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(dev.CAPABILITY_ENV, "blackwell")
    with pytest.raises(ValueError, match="not a capability"):
        dev._env_capability()


def test_major_only_override_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(dev.CAPABILITY_ENV, "9")
    assert dev._env_capability() == (9, 0)


def test_require_cuda_reports_env_confusion_when_no_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The common real cause is running in `base` (CPU-only torch) rather than
    # `retention`, and the message must say so instead of just "no CUDA".
    monkeypatch.setattr(dev.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CPU-only torch"):
        dev.require_cuda()


def test_the_default_floor_admits_ampere_and_hopper(monkeypatch) -> None:
    """The documented reproduction path refused by default on every card older than
    Blackwell -- including the A100 and H100 a reproducer is most likely to have. A
    default that fails for almost everyone is not fixed by documenting an override."""
    assert dev.DEFAULT_FLOOR <= (8, 0)
    assert dev.DEFAULT_FLOOR < dev.SM120


def test_a_blackwell_card_on_a_pre_cu128_torch_raises(monkeypatch) -> None:
    """What the sm_120 floor was standing in for, checked directly. A pre-cu128 torch
    imports cleanly on a 5090 and then returns garbage rather than failing."""
    monkeypatch.setattr(dev.torch.cuda, "get_device_capability", lambda i: (12, 0))
    monkeypatch.setattr(dev.torch.cuda, "get_device_name", lambda i: "RTX 5090")
    monkeypatch.setattr(dev.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(dev.torch.version, "cuda", "12.4")
    with pytest.raises(RuntimeError, match="cu128"):
        dev._assert_blackwell_toolkit(dev.torch.device("cuda:0"))
    monkeypatch.setattr(dev.torch.version, "cuda", "12.8")
    dev._assert_blackwell_toolkit(dev.torch.device("cuda:0"))


def test_a_pre_blackwell_card_is_not_subject_to_the_cu128_check(monkeypatch) -> None:
    monkeypatch.setattr(dev.torch.cuda, "get_device_capability", lambda i: (8, 0))
    monkeypatch.setattr(dev.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(dev.torch.version, "cuda", "11.8")
    dev._assert_blackwell_toolkit(dev.torch.device("cuda:0"))
