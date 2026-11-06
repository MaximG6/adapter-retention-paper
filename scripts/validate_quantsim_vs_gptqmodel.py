"""Validate ar.quantsim against gptqmodel's own Quantizer.

Check the simulated quantizer against gptqmodel on real model layers before
using its measurements.

Two facts make this runnable on Windows, where gptqmodel has no wheel:

1. gptqmodel's quantizer math (gptqmodel/quantization/quantizer.py) is pure
   PyTorch. We download the sdist and drive that reference class directly rather
   than reimplementing it, so this is a real comparison against production code.
   No CUDA extension is built and gptqmodel is never installed.
2. Real Qwen3-8B layers are range-read out of the remote safetensors shards, so
   this costs ~132 MiB of network instead of a 16 GB model download.

Usage:
    python scripts/validate_quantsim_vs_gptqmodel.py
"""

from __future__ import annotations

import importlib.util
import json
import platform
import struct
import subprocess
import sys
import tarfile
import types
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfFileSystem, hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ar.quantsim import QuantConfig, quantize_dequantize  # noqa: E402

GPTQMODEL_VERSION = "7.3.2"
MODEL_REPO = "Qwen/Qwen3-8B"
REAL_LAYERS = (
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
)
CACHE = REPO_ROOT / ".cache" / "gptqmodel_sdist"
OUT = REPO_ROOT / "results" / "raw" / "validation" / "quantsim_vs_gptqmodel.json"

# scheme -> gptqmodel's `sym` flag. symmetric_awq is our own AWQ/torch-style
# signed convention with no gptqmodel counterpart; compared for information only.
SCHEMES: tuple[tuple[str, bool, bool], ...] = (
    ("asymmetric", False, True),
    ("symmetric_gptq", True, True),
    ("symmetric_awq", True, False),
)


def fetch_reference_source() -> Path:
    root = CACHE / f"gptqmodel-{GPTQMODEL_VERSION}"
    if root.is_dir():
        return root
    CACHE.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable, "-m", "pip", "download",
            f"gptqmodel=={GPTQMODEL_VERSION}",
            "--no-deps", "--no-binary", ":all:", "--dest", str(CACHE),
        ],
        check=True,
        capture_output=True,
    )
    tarballs = list(CACHE.glob(f"gptqmodel-{GPTQMODEL_VERSION}.tar.gz"))
    if not tarballs:
        raise RuntimeError(f"gptqmodel sdist not found in {CACHE}")
    with tarfile.open(tarballs[0]) as tf:
        tf.extractall(CACHE)
    if not root.is_dir():
        raise RuntimeError(f"Expected extracted sdist at {root}")
    return root


def load_reference_quantizer(sdist: Path) -> tuple[type, type]:
    """Import gptqmodel's Quantizer with its heavy package __init__ bypassed.

    Importing gptqmodel normally pulls in compiled kernels that do not exist on
    this platform. We register the minimum package skeleton and the two symbols
    quantizer.py imports, then load that one file. Nothing else from gptqmodel
    runs, so there is no risk of a partially-initialised backend silently
    changing behaviour.
    """
    for name, sub in (("gptqmodel", ""), ("gptqmodel.utils", "utils"),
                      ("gptqmodel.quantization", "quantization")):
        mod = types.ModuleType(name)
        mod.__path__ = [str(sdist / "gptqmodel" / sub)] if sub else [str(sdist / "gptqmodel")]
        sys.modules[name] = mod

    logger = types.ModuleType("gptqmodel.utils.logger")
    logger.setup_logger = lambda *a, **k: __import__("logging").getLogger("ref")
    sys.modules["gptqmodel.utils.logger"] = logger

    class BaseQuantizeConfig:
        def __init__(self, bits: int, sym: bool, group_size: int) -> None:
            self.bits = bits
            self.sym = sym
            self.group_size = group_size
            self.mse = 0.0

    cfg_mod = types.ModuleType("gptqmodel.quantization.config")
    cfg_mod.BaseQuantizeConfig = BaseQuantizeConfig
    cfg_mod._normalize_quant_bits = lambda b, **k: b
    cfg_mod.resolve_quant_format = lambda *a, **k: "gptq"
    sys.modules["gptqmodel.quantization.config"] = cfg_mod

    spec = importlib.util.spec_from_file_location(
        "gptqmodel.quantization.quantizer",
        sdist / "gptqmodel" / "quantization" / "quantizer.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load gptqmodel quantizer.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gptqmodel.quantization.quantizer"] = mod
    spec.loader.exec_module(mod)
    return mod.Quantizer, BaseQuantizeConfig


def read_remote_tensor(repo: str, tensor_name: str) -> torch.Tensor:
    """Range-read one tensor from a remote safetensors shard."""
    index_path = hf_hub_download(repo, "model.safetensors.index.json")
    with open(index_path) as fh:
        shard = json.load(fh)["weight_map"][tensor_name]

    fs = HfFileSystem()
    with fs.open(f"{repo}/{shard}", "rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len))
        meta = header[tensor_name]
        start, end = meta["data_offsets"]
        fh.seek(8 + header_len + start)
        raw = fh.read(end - start)

    dtypes = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    if meta["dtype"] not in dtypes:
        raise RuntimeError(f"Unhandled safetensors dtype {meta['dtype']}")
    return torch.frombuffer(bytearray(raw), dtype=dtypes[meta["dtype"]]).reshape(
        meta["shape"]
    )


def reference_qdq(
    w: torch.Tensor, bits: int, group_size: int, sym: bool, Quantizer: type, Cfg: type
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group-wise QDQ via gptqmodel's Quantizer, mirroring gptq.py's column loop."""
    n_in = w.shape[1]
    g = n_in if group_size == -1 else group_size
    out = torch.empty_like(w)
    scales: list[torch.Tensor] = []
    for j in range(0, n_in, g):
        block = w[:, j : j + g]
        q = Quantizer(Cfg(bits=bits, sym=sym, group_size=group_size))
        q.configure(perchannel=True)
        q.find_params(block, weight=True)
        out[:, j : j + g] = q.quantize(block)
        scales.append(q.scale.reshape(-1).clone())
    return out, torch.stack(scales, dim=1)


def main() -> int:
    sdist = fetch_reference_source()
    Quantizer, Cfg = load_reference_quantizer(sdist)

    tensors: dict[str, torch.Tensor] = {}
    for name in REAL_LAYERS:
        t = read_remote_tensor(MODEL_REPO, name)
        # Quantize in fp32: the reference does too, and comparing in bf16 would
        # hide disagreements under the dtype's own rounding.
        tensors[f"{MODEL_REPO}:{name}"] = t.float()
    torch.manual_seed(0)
    tensors["random_normal_4096x4096"] = torch.randn(4096, 4096)

    records: list[dict[str, Any]] = []
    hard_failures: list[str] = []

    header = f"{'tensor':44s} {'config':26s} {'max|Î”dequant|':>14s} {'scales':>7s} {'exact':>6s}"
    print(header)
    print("-" * len(header))

    for tname, w in tensors.items():
        for scheme, sym, must_match in SCHEMES:
            for bits in (3, 4, 8):
                for gs in (32, 128, -1):
                    cfg = QuantConfig(bits=bits, group_size=gs, scheme=scheme)  # type: ignore[arg-type]
                    ours = quantize_dequantize(w, cfg)
                    ref, ref_scale = reference_qdq(w, bits, gs, sym, Quantizer, Cfg)

                    max_diff = (ours.dequant - ref).abs().max().item()
                    scales_ok = bool(
                        torch.allclose(ours.scale, ref_scale, rtol=1e-6, atol=1e-9)
                    )
                    exact = max_diff == 0.0
                    records.append({
                        "tensor": tname,
                        "shape": list(w.shape),
                        "config": cfg.name,
                        "bits": bits,
                        "group_size": gs,
                        "scheme": scheme,
                        "gptqmodel_sym_flag": sym,
                        "must_match_gptqmodel": must_match,
                        "max_abs_dequant_diff": max_diff,
                        "scales_allclose": scales_ok,
                        "bit_exact": exact,
                    })
                    short = tname.replace(f"{MODEL_REPO}:", "")
                    print(
                        f"{short:44s} {cfg.name:26s} {max_diff:14.3e} "
                        f"{str(scales_ok):>7s} {str(exact):>6s}"
                    )
                    if must_match and not (exact and scales_ok):
                        hard_failures.append(f"{tname} {cfg.name} diff={max_diff}")

    payload = {
        "gptqmodel_version": GPTQMODEL_VERSION,
        "gptqmodel_installed": False,
        "gptqmodel_note": (
            "Reference quantizer math loaded from the sdist as pure PyTorch. "
            "gptqmodel has no Windows wheel and was not installed; no CUDA "
            "extension was built."
        ),
        "model_repo": MODEL_REPO,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "records": records,
        "hard_failures": hard_failures,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))

    print()
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    if hard_failures:
        print("FAILED: schemes that must match gptqmodel disagree:")
        for f in hard_failures:
            print("  " + f)
        return 1
    n = sum(1 for r in records if r["must_match_gptqmodel"])
    print(
        f"PASS: {n} configs across {len(tensors)} tensors are bit-exact against "
        f"gptqmodel.\n"
        "symmetric_awq rows are our own signed convention with no gptqmodel "
        "counterpart and are expected to differ."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

