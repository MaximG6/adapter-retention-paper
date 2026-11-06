from __future__ import annotations

import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import torch

from ar.device import describe_device

TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "trl",
    "datasets",
    "accelerate",
    "numpy",
    "pydantic",
    "safetensors",
    "huggingface-hub",
)


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return out.stdout.strip()


def _git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return bool(out.stdout.strip())


def _package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not installed"
    return out


def build_manifest(
    device: torch.device | None = None,
    seeds: dict[str, int] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture the software, hardware and revision metadata for an experimental run."""
    manifest: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "packages": _package_versions(),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "seeds": seeds or {},
    }
    if device is not None:
        manifest["device"] = describe_device(device)
    if extra:
        manifest.update(extra)
    return manifest
