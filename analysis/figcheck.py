"""Numerical cross-checks for figures.

A figure that renders without error looks validated and is not: the output is an image,
and images do not raise. Our predictive-gap figure marked 2 adapters where the analysis
reports 4, rendered cleanly, and silently dropped the two that carry the paper's sign
argument (§7.9).

Every figure script therefore asserts its plotted values against an INDEPENDENT
computation of the same quantity before writing the file. "Independent" means the
comparison recomputes from `results/raw/**` by a different route than the plotting code
took -- not a second call to the same helper, which would only prove the helper is
deterministic.

Usage inside a figure script:

    from figcheck import Check
    chk = Check("fig08")
    chk.equal("n resolvable pairs", plotted_pairs, wvn_pairs_from_word_vs_noise())
    chk.close("fig08")      # raises if anything mismatched
"""

from __future__ import annotations

import ast
import inspect
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
P0 = REPO_ROOT / "results" / "raw" / "phase0"
P1 = REPO_ROOT / "results" / "raw" / "phase1"


class FigureCheckError(AssertionError):
    """A figure's plotted values disagree with an independent recomputation."""


class VacuousCheckError(AssertionError):
    """A comparison's two sides are the same expression, so it cannot fail.

    Not hypothetical. On 2026-08-03, minutes after the rule forbidding it was written
    into the paper, this shipped into a figure cross-check:

        chk.close_to("cosine", mean([r["cosine"] for r in real[a]]),
                               mean([r["cosine"] for r in real[a]]), tol=0)

    mean(X) against mean(X). It printed "ok" exactly as a real check does. Review caught
    that one; this guard catches the next.
    """


def _duplicate_argument_source(depth: int) -> str | None:
    """Return the shared source text if the caller passed one expression twice.

    Inspects the caller's own source rather than its values: two sides that are
    textually the same expression cannot disagree, whatever they evaluate to. Any
    failure to introspect returns None -- the guard degrades to absent, never to a
    spurious error.
    """
    try:
        info = inspect.stack()[depth]
        path = Path(info.filename)
        if not path.exists():
            return None
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # Match only calls to OUR comparison methods. Selecting merely the innermost
        # Call spanning the line picks up nested helpers -- `len(vals)` inside
        # `chk.equal("n", len(vals), len(vals))` -- and the guard silently never fires,
        # which would make this guard the very thing it exists to forbid.
        wanted = {"equal", "close_to", "all_close"}
        call = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr in wanted):
                continue
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None) or start
            if start is None or not (start <= info.lineno <= end):
                continue
            if call is None or start >= call.lineno:
                call = node
        if call is None:
            return None
        args = list(call.args)
        kw = {k.arg: k.value for k in call.keywords if k.arg}
        plotted = args[1] if len(args) > 1 else kw.get("plotted")
        reference = args[2] if len(args) > 2 else kw.get("reference")
        if plotted is None or reference is None:
            return None
        sa, sb = ast.unparse(plotted), ast.unparse(reference)
        return sa if sa == sb else None
    except (OSError, SyntaxError, ValueError, IndexError, RecursionError):
        return None


class Check:
    """Collects comparisons and fails loudly at close().

    Two structural guards beyond the value comparisons themselves:

      1. A comparison whose two sides are the SAME EXPRESSION raises immediately. Such a
         check cannot fail, and is indistinguishable in output from one that constrains
         everything -- which is exactly why it survives review.
      2. `plots(n)` declares how many values the figure draws. close() reports the ratio
         of assertions to plotted values and warns when a figure asserts little about a
         lot. One figure's check count went from 2 to 39 once its vacuous comparisons
         were replaced; the console output was identical before and after.
    """

    #: Warn when assertions cover less than this fraction of the plotted values.
    COVERAGE_WARN = 0.5

    def __init__(self, name: str) -> None:
        self.name = name
        self.rows: list[tuple[str, Any, Any, bool]] = []
        self.covered: int = 0
        self.n_plotted: int | None = None

    def plots(self, n: int) -> None:
        """Declare how many values this figure draws, for the coverage guard."""
        self.n_plotted = n

    def _guard_vacuous(self, what: str) -> None:
        dup = _duplicate_argument_source(depth=3)
        if dup is not None:
            raise VacuousCheckError(
                f"{self.name}: check {what!r} compares the same expression to itself "
                f"({dup!r}). A check that shares a code path with the thing it checks "
                f"is not a check -- recompute the reference independently."
            )

    def equal(self, what: str, plotted: Any, reference: Any) -> None:
        self._guard_vacuous(what)
        ok = plotted == reference
        self.rows.append((what, plotted, reference, ok))
        self.covered += 1

    def close_to(self, what: str, plotted: float, reference: float,
                 tol: float = 1e-9) -> None:
        self._guard_vacuous(what)
        ok = (math.isfinite(plotted) and math.isfinite(reference)
              and abs(plotted - reference) <= tol)
        self.rows.append((what, round(plotted, 6), round(reference, 6), ok))
        self.covered += 1

    def all_close(self, what: str, plotted: list[float], reference: list[float],
                  tol: float = 1e-9) -> None:
        self._guard_vacuous(what)
        ok = (len(plotted) == len(reference)
              and all(abs(a - b) <= tol for a, b in zip(plotted, reference, strict=False)))
        self.rows.append((what, f"n={len(plotted)}", f"n={len(reference)}", ok))
        self.covered += len(plotted)

    def close(self) -> None:
        bad = [r for r in self.rows if not r[3]]
        status = "FAIL" if bad else "ok"
        low = bool(self.n_plotted) and self.covered / self.n_plotted < self.COVERAGE_WARN
        cover = ""
        if self.n_plotted:
            cover = f"  [{self.covered} values checked / {self.n_plotted} plotted]"
            if low:
                cover += "  <-- LOW COVERAGE"
        print(f"  [{self.name}] cross-check "
              f"{len(self.rows) - len(bad)}/{len(self.rows)} {status}{cover}")
        for what, plotted, ref, ok in self.rows:
            if not ok:
                print(f"    MISMATCH {what}: plotted={plotted!r} reference={ref!r}")
        if low:
            print(f"    WARNING: {self.name} asserts {self.covered} value(s) about "
                  f"{self.n_plotted} plotted. A check constraining little prints the "
                  f"same 'ok' as one constraining everything.")
        if bad:
            raise FigureCheckError(
                f"{self.name}: {len(bad)} plotted value(s) disagree with the independent "
                f"recomputation. The figure was NOT written."
            )


# ------------------------------------------------------------------ references
# These recompute from raw by a deliberately different route than the figure code:
# they re-read the JSONL files themselves rather than sharing a loader.

def _mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else float("nan")


def _rows_p1() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in sorted(P1.glob("*/records.jsonl")):
        out += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()
                if x.strip()]
    return out


def ref_retention_by_word(precision: str) -> dict[str, float]:
    """Elicitation retention vs own BF16, keyed by secret word."""
    rows = _rows_p1()
    acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    word: dict[str, str] = {}
    for r in rows:
        word[r["adapter"]] = r["secret_word"]
        if r["condition"] == "aligned_bf16" and r["precision"] == "bf16":
            acc[(r["adapter"], "ref")].append(r["guesser_p_word_normalised"])
        elif r["condition"] == "aligned_quant" and r["precision"] == precision:
            acc[(r["adapter"], "cur")].append(r["guesser_p_word_normalised"])
    out: dict[str, float] = {}
    for a, w in word.items():
        ref, cur = _mean(acc[(a, "ref")]), _mean(acc[(a, "cur")])
        if ref:
            out[w] = cur / ref
    return out


def ref_weight_flip_rate(substr: str = "taboo",
                         regime: str = "adaptive_scale",
                         key: str = "code_flip_rate") -> dict[str, float]:
    """Mean flip/change rate per adapter, INT4 g128 asymmetric, 4-layer runs.

    Restricted to L4 so every adapter is measured under one configuration; see
    fig01's weight_side() for why. Selected by excluding L36 rather than by matching
    "L4_", so the two implementations disagree if either directory convention changes.

    `regime` defaults to adaptive_scale, matching Figure 1's other panel: the Phase 1
    behavioural pipeline quantizes a merged model on its own recomputed grid, so a
    fixed_scale weight panel beside a behavioural one compares two treatments (§3.3).
    """
    if regime not in ("fixed_scale", "adaptive_scale"):
        raise ValueError(f"Unknown regime {regime!r}")
    acc: dict[str, list[float]] = defaultdict(list)
    for p in P0.glob("public_adapter/*/*/records.jsonl"):
        if p.parent.name.startswith("L36"):
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if (substr in r["adapter"] and r["scheme"] == "asymmetric"
                    and r["regime"] == regime):
                acc[r["adapter"]].append(r[key])
    return {a: _mean(v) for a, v in acc.items()}


def ref_weight_metric(key: str, layers: str = "L4") -> dict[str, float]:
    """Per-adapter mean of a Phase 0 weight metric, INT4 g128 asymmetric fixed_scale.

    Re-reads the JSONL directly rather than sharing the figure's loader, so a defect in
    that loader cannot be reproduced here and reported as agreement (§7.10).
    """
    acc: dict[str, list[float]] = defaultdict(list)
    for p in P0.glob("public_adapter/*/*/records.jsonl"):
        is_l36 = p.parent.name.startswith("L36")
        if (layers == "L4" and is_l36) or (layers == "L36" and not is_l36):
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["scheme"] == "asymmetric" and r["regime"] == "fixed_scale":
                acc[r["adapter"]].append(r[key])
    return {a: _mean(v) for a, v in acc.items()}


def ref_layer_flip_profile() -> dict[int, float]:
    """Per-layer mean code-flip rate from the 36-layer run."""
    acc: dict[int, list[float]] = defaultdict(list)
    for p in P0.glob("public_adapter/*/L36*/records.jsonl"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["scheme"] == "asymmetric" and r["regime"] == "fixed_scale":
                acc[r["layer"]].append(r["code_flip_rate"])
    return {k: _mean(v) for k, v in acc.items()}


def ref_refusal_p(kind: str, condition: str) -> float:
    """Mean p_refuse for one prompt kind and condition, read straight from the JSONL."""
    vals: list[float] = []
    for p in (P1 / "refusal_validation").glob("*.jsonl"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["prompt_kind"] == kind and r["condition"] == condition:
                vals.append(r["p_refuse"])
    return _mean(vals)


def ref_knowledge_ratio(precision: str) -> float:
    """Aligned/base knowledge probe within one precision."""
    rows = _rows_p1()
    bc = "base_bf16" if precision == "bf16" else "base_quant"
    ac = "aligned_bf16" if precision == "bf16" else "aligned_quant"
    b = [r["p_knowledge_mean"] for r in rows
         if r["precision"] == precision and r["condition"] == bc]
    a = [r["p_knowledge_mean"] for r in rows
         if r["precision"] == precision and r["condition"] == ac]
    return _mean(a) / _mean(b)


def ref_resolvable_pairs(precision: str = "int3_g128", seed: int = 0,
                         n: int = 20000) -> tuple[int, set[str]]:
    """Pairs of adapters whose bootstrap CIs do not overlap.

    This mirrors `analysis/word_vs_noise.py`, which is the analysis of record for PG-2.
    Reimplemented here from raw so a bug in the figure's own pair logic cannot be
    reproduced by sharing code with it -- which is exactly the bug this caught.

    The sampling unit is the INTENT and the two conditions are paired: the hint battery
    is 8 intents x 3 paraphrases, so an observation-level bootstrap counts clustered
    draws as independent ones and narrows the interval -- by 11-18% on the battery at the
    measured ICC of 0.175-0.303, not by the sqrt(2) an earlier draft assumed (B.12).
    Written out longhand here
    rather than calling bootstrap.cluster_ratio_ci, for the reason in the docstring
    above: this file's job is to disagree with the estimator, not to share it.
    """
    rows = _rows_p1()
    by: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    kind: dict[str, str] = {}
    word: dict[str, str] = {}
    for r in rows:
        word[r["adapter"]] = r["secret_word"]
        kind[r["intent"]] = r["prompt_kind"]
        if r["condition"] == "aligned_bf16" and r["precision"] == "bf16":
            by[(r["adapter"], "ref")][r["intent"]].append(
                r["guesser_p_word_normalised"])
        elif r["condition"] == "aligned_quant" and r["precision"] == precision:
            by[(r["adapter"], "cur")][r["intent"]].append(
                r["guesser_p_word_normalised"])

    intervals: dict[str, tuple[float, float]] = {}
    for a, w in word.items():
        num, den = by[(a, "cur")], by[(a, "ref")]
        intents = sorted(set(num) & set(den))
        strata: dict[str, list[str]] = defaultdict(list)
        for i in intents:
            strata[kind[i]].append(i)
        rng = random.Random(seed)
        draws = []
        for _ in range(n):
            xs: list[float] = []
            ys: list[float] = []
            for group in strata.values():
                for _ in range(len(group)):
                    pick = group[rng.randrange(len(group))]
                    xs += num[pick]
                    ys += den[pick]
            y = _mean(ys)
            draws.append(_mean(xs) / y if y else float("nan"))
        draws.sort()
        intervals[w] = (draws[int(0.025 * n)], draws[int(0.975 * n)])

    words = sorted(intervals)
    pairs: list[tuple[str, str]] = []
    for i, wi in enumerate(words):
        for wj in words[i + 1:]:
            lo_i, hi_i = intervals[wi]
            lo_j, hi_j = intervals[wj]
            if hi_i < lo_j or hi_j < lo_i:
                pairs.append((wi, wj))
    _PAIR_CACHE[precision] = pairs
    return len(pairs), {w for p in pairs for w in p}


#: Populated by ref_resolvable_pairs. The pair list, not just the count, because the
#: direction claim ("six of seven run against the predictor") needs the members.
_PAIR_CACHE: dict[str, list[tuple[str, str]]] = {}


def ref_separating_pairs(precision: str = "int3_g128") -> list[tuple[str, str]]:
    """The word pairs whose intervals separate at `precision`."""
    if precision not in _PAIR_CACHE:
        ref_resolvable_pairs(precision)
    return _PAIR_CACHE[precision]
