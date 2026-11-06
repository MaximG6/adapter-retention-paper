"""Check ar.predict against the measured records for the six adapters.

The tool estimates step sizes from a few sampled layers and reconstructs mean
|delta| from the adapter alone. This quantifies the error that sampling
introduces, so the tool's output can be reported with a known accuracy rather
than an implied one.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ar.predict import predict  # noqa: E402

RAW = REPO_ROOT / "results" / "raw" / "phase0" / "public_adapter"


def measured() -> dict[str, dict[str, float]]:
    best: dict[str, tuple[int, list[dict]]] = {}
    for p in RAW.glob("*/*/records.jsonl"):
        rows = [
            json.loads(x)
            for x in p.read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
        rows = [
            r for r in rows
            if r["scheme"] == "asymmetric" and r["regime"] == "fixed_scale"
        ]
        if not rows:
            continue
        adapter = rows[0]["adapter"]
        n_layers = len({r["layer"] for r in rows})
        if adapter not in best or n_layers > best[adapter][0]:
            best[adapter] = (n_layers, rows)
    out: dict[str, dict[str, float]] = {}
    for adapter, (n_layers, rows) in best.items():
        out[adapter] = {
            "flip": sum(r["code_flip_rate"] for r in rows) / len(rows),
            "cosine": sum(r["cosine"] for r in rows) / len(rows),
            "n_layers": float(n_layers),
        }
    return out


def main() -> int:
    meas = measured()
    hdr = (
        f"{'adapter':>36} {'flip pred':>10} {'flip meas':>10} {'err':>7} "
        f"{'cos pred':>9} {'cos meas':>9} {'err':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    errs_f, errs_c = [], []
    for adapter, m in sorted(meas.items()):
        p = predict(adapter)["overall"]
        ef = p["predicted_flip_rate"] / m["flip"] - 1
        ec = p["predicted_cosine"] / m["cosine"] - 1
        errs_f.append(abs(ef))
        errs_c.append(abs(ec))
        print(
            f"{adapter.split('/')[-1][:36]:>36} {p['predicted_flip_rate']:>10.5f} "
            f"{m['flip']:>10.5f} {ef:>+7.1%} {p['predicted_cosine']:>9.4f} "
            f"{m['cosine']:>9.4f} {ec:>+7.1%}"
        )
    print(
        f"\n  mean |error|: flip {sum(errs_f) / len(errs_f):.1%}, "
        f"cosine {sum(errs_c) / len(errs_c):.1%}"
    )
    print(f"  max  |error|: flip {max(errs_f):.1%}, cosine {max(errs_c):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
