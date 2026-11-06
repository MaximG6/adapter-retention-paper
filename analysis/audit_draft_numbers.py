"""Audit every number claimed in the paper draft against the raw records.

Appendix B forced §4's numbers into final form and immediately caught a stale value that
had reached the Abstract (EXP-022): the draft had been written from EXP-008, whose DPO
rows EXP-011 superseded. Nothing had forced §5, §6 or §7 through the same check.

This encodes each claim the draft makes as an EXPECTED value, recomputes it from
`results/raw/**`, and reports any disagreement. It is a regression test on the prose:
if a raw record changes, or a number was transcribed wrongly, this fails loudly instead
of the error surviving into a figure.

Tolerances are per-claim rather than global, because "99.2%" and "0.9924" carry
different numbers of significant figures in the text.

Usage:
    python analysis/audit_draft_numbers.py            # report
    python analysis/audit_draft_numbers.py --strict   # exit 1 on any mismatch
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import bootstrap

REPO_ROOT = Path(__file__).resolve().parents[1]
P0 = REPO_ROOT / "results" / "raw" / "phase0"
P1 = REPO_ROOT / "results" / "raw" / "phase1"
README = REPO_ROOT / "README.md"
PRECISIONS = ["int4_g128", "int4_per_channel", "int3_g128"]


def mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else float("nan")


def boot_ci(xs: list[float]) -> tuple[float, float]:
    """Delegates to analysis/bootstrap.py. Exact by enumeration for the
    six-adapter population; see that module for why there is no seed."""
    return bootstrap.ci(xs)


def cliffs(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return float("nan")
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))


def cv(xs: list[float]) -> float:
    return statistics.stdev(xs) / mean(xs) if len(xs) > 1 else float("nan")


# --------------------------------------------------------------------------- loaders
def load_p1() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(P1.glob("*/records.jsonl")):
        rows += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()
                 if x.strip()]
    return rows


def load_refusal(name: str) -> list[dict[str, Any]]:
    p = P1 / "refusal_validation" / name
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()
            if x.strip()]


def load_snr() -> dict[str, float]:
    snr: dict[str, list[float]] = defaultdict(list)
    for f in (P0 / "output_snr_orthonormal").glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                snr[r["adapter"]].append(r["snr_out_orthonormal"])
    return {a: mean(v) for a, v in snr.items()}


P1ROWS = load_p1()
BY: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
for _r in P1ROWS:
    BY[(_r["adapter"], _r["condition"], _r["precision"])].append(_r)
ADAPTERS = sorted({r["adapter"] for r in P1ROWS})
WORD = {a: next(r["secret_word"] for r in P1ROWS if r["adapter"] == a)
        for a in ADAPTERS}
REF = load_refusal("Kurapika993__llama-3.1-8b-responsible-ai-safety-lora.jsonl")
XST = load_refusal("Kurapika993__llama-3.1-8b-responsible-ai-safety-lora__xstest.jsonl")


def retention(adapter: str, precision: str) -> float:
    ref = mean([r["guesser_p_word_normalised"]
                for r in BY[(adapter, "aligned_bf16", "bf16")]])
    cur = mean([r["guesser_p_word_normalised"]
                for r in BY[(adapter, "aligned_quant", precision)]])
    return cur / ref if ref else float("nan")


def retentions(precision: str) -> list[float]:
    return [retention(a, precision) for a in ADAPTERS]


def refusal_stat(rows: list[dict[str, Any]], cond: str, kinds: tuple[str, ...],
                 key: str) -> float:
    return mean([float(r[key]) for r in rows
                 if r["condition"] == cond and r["prompt_kind"] in kinds])


# ----------------------------------------------------------------------------- README
# The README is generated, which is not the same as being checked. Nothing verified that
# the committed file still agreed with the raw records, so a stale commit -- or a hand
# edit to the one document everyone reads -- would have gone unnoticed. These read the
# file on disk and compare it to the same recomputation the paper's claims use.


def readme_text() -> str:
    return README.read_text(encoding="utf-8")


def readme_number(pattern: str) -> float:
    """First capture group of `pattern` in README.md, as a float.

    Raises on no match. A regex that silently stops matching would turn this whole
    section into a check that passes because it tests nothing -- the vacuous-check
    failure recorded in the methodology section.
    """
    m = re.search(pattern, readme_text())
    if m is None:
        raise ValueError(f"README pattern never matched: {pattern}")
    return float(m.group(1))


def readme_table_cell(adapter_label: str, column: int) -> float:
    """A cell from the README's weight-space table, by row label and 0-based column
    offset after the label. Columns: base, r, scaling, cosine, code-flip, rel, SNR."""
    for line in readme_text().splitlines():
        if line.startswith(f"| {adapter_label} |"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            return float(cells[1 + column].rstrip("%"))
    raise ValueError(f"README has no table row for {adapter_label!r}")


# ------------------------------------------------------------ cross-artifact agreement
# Every check above compares one number against the raw records. None of them compares
# two printed numbers against each other, and that is a real gap: section 5.1's Table 2
# and Appendix B.6 disagreed in the last digit of three intervals for a whole draft
# cycle, and every individual claim still passed, because each was inside its own
# tolerance of the raw value. Two artifacts can each be within tolerance of the data and
# still contradict one another in print.
#
# This checks the other axis: a quantity claimed in more than one place must read the
# same in every place.

#: (quantity, [(file, pattern with one capture group), ...]) -- at least two sites each.
CROSS_ARTIFACT: list[tuple[str, list[tuple[str, str]]]] = [
    # Three quantities where there was one. The paper quoted a fixed_scale weight number
    # beside an adaptive_scale behavioural one for a whole draft cycle, because "% of
    # weights unchanged" reads as regime-free and is not (EXP-038). Every site now names
    # its regime, and all three are checked across every artifact that prints them.
    ("codes unchanged, fixed_scale %", [
        ("README.md", r"the step size, ([\d.]+)% of codes are unchanged"),
        ("paper/01-introduction.md",
         r"with ([\d.]+)% of codes unchanged when the"),
        ("paper/04-results-weight-space.md",
         r"\(([\d.]+)% of codes unchanged under `fixed_scale`"),
        # Figure 1's caption stopped quoting the complements once the forward-claim gate
        # required every summary number to be stated by a section, so the tex site moved
        # to §5.1's pairing sentence, which is where the report already stated it.
        ("paper/tex/main.tex",
         r"\(([\d.]+)\\% of codes unchanged under \\texttt\{fixed"),
    ]),
    ("values changed, adaptive_scale %", [
        ("README.md", r"moves the grid, ([\d.]+)% of stored values change"),
        ("paper/01-introduction.md", r"adapter shows \*\*([\d.]+)% of stored \*values\*"),
        ("paper/04-results-weight-space.md",
         r"At INT4 g128, ([\d.]+)% of their stored values differ"),
        ("paper/08-09-limitations-conclusion.md",
         r"\*\*([\d.]+)% of the six taboo adapters' stored values differ"),
        ("paper/tex/main.tex",
         r"a single adapter shows ([\d.]+)\\% of stored \\emph\{values\}"),
    ]),
    ("codes changed, adaptive_scale %", [
        ("README.md", r"while only ([\d.]+)% of integer codes do"),
        ("paper/01-introduction.md",
         r"\*\*([\d.]+)% of the integer \*codes\*\*\*, a factor of 41"),
        ("paper/04-results-weight-space.md",
         r"and ([\d.]+)% of the\s*\n?integer codes do"),
        ("paper/tex/main.tex",
         r"([\d.]+)\\% of integer \\emph\{codes\}---a factor of 41"),
    ]),
    ("behaviour retained %", [
        ("README.md", r"mean elicitation retention was \*\*([\d.]+)%\*\*"),
        ("paper/00-abstract.md", r"no detectable loss\*\*: retention ([\d.]+)%"),
        ("paper/01-introduction.md", r"\*\* Retention is ([\d.]+)%, and its"),
        ("paper/04-results-weight-space.md", r"Retention is ([\d.]+)% with an enumerated"),
        ("paper/08-09-limitations-conclusion.md",
         r"capability is detectable\*\*: retention ([\d.]+)%,"),
    ]),
    # The reframed headline: the interval is now the claim, so it must agree everywhere
    # it is quoted, including the two hand-written LaTeX copies.
    ("headline CI lo (all sites)", [
        ("paper/00-abstract.md", r"\*\*\[([\d.]+)%, [\d.]+%\]\*\*, spanning parity"),
        ("paper/01-introduction.md", r"— \[([\d.]+)%,\s*\n?\s*[\d.]+%\] — spans parity"),
        ("paper/04-results-weight-space.md",
         r"enumerated 95% interval of \[([\d.]+)%, [\d.]+%\]"),
        ("paper/08-09-limitations-conclusion.md",
         r"enumerated interval \[([\d.]+)%, [\d.]+%\]"),
        ("paper/tex/main.tex", r"95\\% interval \\textbf\{\[([\d.]+)\\%"),
    ]),
    ("retention int4_g128 %", [
        ("README.md", r"\*\*([\d.]+)%\*\* at INT4 g128"),
        ("paper/04-results-weight-space.md",
         r"INT4 g128 \| \*\*([\d.]+)%\*\*"),
        ("paper/appendix-B-tables.md", r"\*\*mean\*\* \| . \| . \| \*\*([\d.]+)%\*\*"),
    ]),
    ("retention int4_per_channel %", [
        ("README.md", r"\*\*([\d.]+)%\*\* at INT4 per-channel"),
        ("paper/04-results-weight-space.md", r"INT4 per-channel \| \*\*([\d.]+)%\*\*"),
        ("paper/appendix-B-tables.md",
         r"\*\*mean\*\* \| . \| . \| \*\*[\d.]+%\*\* \| . \| \*\*([\d.]+)%\*\*"),
    ]),
    ("retention int3_g128 %", [
        ("README.md", r"\*\*([\d.]+)%\*\* at INT3 g128"),
        ("paper/04-results-weight-space.md", r"INT3 g128 \| \*\*([\d.]+)%\*\*"),
        ("paper/appendix-B-tables.md",
         r"\*\*mean\*\* \| . \| . \| \*\*[\d.]+%\*\* \| . \| \*\*[\d.]+%\*\* \| . \| \*\*([\d.]+)%\*\*"),
    ]),
    ("CI lo int4_g128", [
        ("README.md", r"INT4 g128: \[([\d.]+)%"),
        ("paper/04-results-weight-space.md", r"INT4 g128 \| [^|]*\| \[([\d.]+)%"),
        ("paper/appendix-B-tables.md", r"95% CI over adapters \| . \| . \| \[([\d.]+)%"),
    ]),
    ("CI hi int4_g128", [
        ("README.md", r"INT4 g128: \[[\d.]+%, ([\d.]+)%\]"),
        ("paper/04-results-weight-space.md",
         r"INT4 g128 \| [^|]*\| \[[\d.]+%, ([\d.]+)%\]"),
        ("paper/appendix-B-tables.md",
         r"95% CI over adapters \| . \| . \| \[[\d.]+%, ([\d.]+)%\]"),
    ]),
    ("CI lo int4_per_channel", [
        ("paper/04-results-weight-space.md", r"INT4 per-channel \| [^|]*\| \[([\d.]+)%"),
        ("paper/appendix-B-tables.md",
         r"95% CI over adapters \|[^|]*\|[^|]*\|[^|]*\|[^|]*\| \[([\d.]+)%"),
    ]),
    ("CI hi int4_per_channel", [
        ("paper/04-results-weight-space.md",
         r"INT4 per-channel \| [^|]*\| \[[\d.]+%, ([\d.]+)%\]"),
        ("paper/appendix-B-tables.md",
         r"95% CI over adapters \|[^|]*\|[^|]*\|[^|]*\|[^|]*\| \[[\d.]+%, ([\d.]+)%\]"),
    ]),
    ("CI lo int3_g128", [
        ("paper/04-results-weight-space.md", r"INT3 g128 \| [^|]*\| \[([\d.]+)%"),
        ("paper/appendix-B-tables.md",
         r"95% CI over adapters \|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\| \[([\d.]+)%"),
    ]),
    ("CI hi int3_g128", [
        ("paper/04-results-weight-space.md",
         r"INT3 g128 \| [^|]*\| \[[\d.]+%, ([\d.]+)%\]"),
        ("paper/appendix-B-tables.md",
         r"95% CI over adapters \|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\| \[[\d.]+%, ([\d.]+)%\]"),
    ]),
    # Carried "15-21x" from the superseded EXP-009 into four sections for the whole
    # draft; the four agreed with each other and none agreed with the data.
    ("subspace amplification lo", [
        ("paper/00-abstract.md", r"\*\*([\d.]+)–[\d.]+× higher\*\* than weight-space fidelity"),
        ("paper/01-introduction.md", r"a factor of ([\d.]+)–[\d.]+ across the nine"),
        ("paper/04-results-weight-space.md", r"— ([\d.]+)–[\d.]+× the\s*\n?weight-space"),
        ("paper/08-09-limitations-conclusion.md",
         r"fidelity is \*measured\* ([\d.]+)–[\d.]+× higher"),
        ("paper/tex/main.tex", r"\\textbf\{([\d.]+)--[\d.]+\$\\times\$ higher\}"),
    ]),
    ("subspace amplification hi", [
        ("paper/00-abstract.md", r"\*\*[\d.]+–([\d.]+)× higher\*\* than weight-space fidelity"),
        ("paper/01-introduction.md", r"a factor of [\d.]+–([\d.]+) across the nine"),
        ("paper/04-results-weight-space.md", r"— [\d.]+–([\d.]+)× the\s*\n?weight-space"),
        ("paper/08-09-limitations-conclusion.md",
         r"fidelity is \*measured\* [\d.]+–([\d.]+)× higher"),
        ("paper/tex/main.tex", r"\\textbf\{[\d.]+--([\d.]+)\$\\times\$ higher\}"),
    ]),
    # main.tex is the arXiv body and is hand-written, not generated from the markdown.
    # It was outside every check until it was found still carrying the superseded
    # amplification figure after the markdown had been corrected.
    # main.tex prints each headline quantity more than once. extract() returns the first
    # capture, so the conclusion's copies need their own anchored entries or they are
    # outside the check -- which is how the tex conclusion kept a superseded number after
    # the abstract had been corrected.
    ("values changed % (tex conclusion)", [
        ("paper/tex/main.tex", r"INT4 g128, ([\d.]+)\\% of the\s*\n?six taboo"),
        ("paper/08-09-limitations-conclusion.md",
         r"\*\*([\d.]+)% of the six taboo adapters' stored values differ"),
    ]),
    # The conclusion's code-count restatement was the third occurrence of the same three
    # numbers and went in the round-8 deduplication; both conclusions now state the value
    # count only, which is covered by the entry above.
    # ---- the round-8 companions, inside the perimeter ----
    # METHODOLOGY.md left the PDF; it did not leave the checks. Each entry below is a
    # number it shares with the paper, so the two cannot drift apart now that they are
    # separate documents.
    ("retention int4_g128 % (METHODOLOGY)", [
        ("METHODOLOGY.md", r"98\.9% / ([\d.]+)% on the same six adapters"),
        ("paper/04-results-weight-space.md", r"INT4 g128 \| \*\*([\d.]+)%\*\*"),
    ]),
    ("codes unchanged, fixed_scale % (METHODOLOGY)", [
        ("METHODOLOGY.md", r"the headline became \*\*([\d.]+)% / [\d.]+% on the same"),
        ("paper/04-results-weight-space.md",
         r"\(([\d.]+)% of codes unchanged under `fixed_scale`"),
    ]),
    ("cosine predictor error % (METHODOLOGY)", [
        ("METHODOLOGY.md", r"once actually computed . \*\*([\d.]+)%\*\*"),
        ("paper/appendix-A-tool.md", r"then averaged \| \*\*([\d.]+)%\*\*"),
    ]),
    ("behaviour retained % (tex)", [
        ("paper/tex/main.tex", r"elicitation retention is ([\d.]+)\\%,"),
        ("paper/00-abstract.md", r"no detectable loss\*\*: retention ([\d.]+)%"),
    ]),
    ("int3 span lo (tex)", [
        ("paper/tex/main.tex", r"retention spans ([\d.]+)--[\d.]+\\%"),
        ("README.md", r"\*\*([\d.]+)% to [\d.]+%\*\* at INT3"),
    ]),
    ("int3 span hi (tex)", [
        ("paper/tex/main.tex", r"retention spans [\d.]+--([\d.]+)\\%"),
        ("README.md", r"\*\*[\d.]+% to ([\d.]+)%\*\* at INT3"),
    ]),
]


def extract(rel: str, pattern: str) -> str:
    """First capture of `pattern` in `rel`, as printed.

    Raises when the pattern stops matching. A cross-artifact check whose regex silently
    finds nothing agrees with everything -- the exact shape of the cross-reference
    checker that reported nine false positives because its pattern assumed whitespace
    where the text had a period.
    """
    text = (REPO_ROOT / rel).read_text(encoding="utf-8")
    m = re.search(pattern, text)
    if m is None:
        raise ValueError(f"{rel}: pattern never matched: {pattern}")
    if m.group(1) is None:
        # An alternation whose taken branch has no group captures nothing and would
        # compare as equal to every other None. Same vacuous-check family.
        raise ValueError(f"{rel}: pattern matched but captured nothing: {pattern}")
    return m.group(1)


def cross_artifact_disagreements() -> list[tuple[str, dict[str, str]]]:
    """Quantities that do not read the same everywhere they are claimed.

    Every dead pattern is collected before raising, rather than raising on the first.
    A revision that rewords one claim usually rewords it at every site, and failing one
    site per run turns a single edit into as many runs as there are sites -- which is
    both slow and an invitation to re-point patterns without reading what they now
    match. The gate is unchanged in strictness: any dead pattern still fails the run.
    """
    bad, dead = [], []
    for name, sites in CROSS_ARTIFACT:
        seen = {}
        for rel, pat in sites:
            try:
                seen[rel] = extract(rel, pat)
            except ValueError as exc:
                dead.append(f"  [{name}] {exc}")
        if len(set(seen.values())) > 1:
            bad.append((name, seen))
    if dead:
        raise ValueError(f"{len(dead)} cross-artifact patterns no longer match:\n"
                         + "\n".join(dead))
    return bad


# --------------------------------------------------------------------------- claims
Claim = tuple[str, str, float, Callable[[], float], float]

HARM = ("harmful_direct", "harmful_indirect")


def claims() -> list[Claim]:
    c: list[Claim] = []

    # ---- §5.1 dose-response ----
    c.append(("§5.1", "record count", 1536.0, lambda: float(len(P1ROWS)), 0))
    c.append(("§5.1", "adapters", 6.0, lambda: float(len(ADAPTERS)), 0))
    for prec, exp in zip(PRECISIONS, (0.992, 0.772, 0.578), strict=True):
        c.append((f"§5.1", f"mean retention {prec}", exp,
                  lambda p=prec: mean(retentions(p)), 5e-4))
    # Exact by enumeration (analysis/bootstrap.py), so these are the values, not an
    # estimate of them: tolerance is for printing, not for Monte Carlo noise.
    for prec, lo, hi in (("int4_g128", 0.9066, 1.0759),
                         ("int4_per_channel", 0.6895, 0.8602),
                         ("int3_g128", 0.4168, 0.7433)):
        c.append(("§5.1", f"CI lo {prec}", lo,
                  lambda p=prec: boot_ci(retentions(p))[0], 3e-3))
        c.append(("§5.1", f"CI hi {prec}", hi,
                  lambda p=prec: boot_ci(retentions(p))[1], 3e-3))
    c.append(("§5.1", "below 50% at int3", 2.0,
              lambda: float(sum(1 for x in retentions("int3_g128") if x < 0.5)), 0))

    # per-adapter retention table
    for word, vals in (("gold", (0.813, 0.624, 0.413)),
                       ("moon", (1.002, 0.781, 0.864)),
                       ("rock", (1.162, 0.775, 0.577)),
                       ("ship", (1.032, 0.798, 0.287)),
                       ("smile", (1.008, 0.685, 0.513)),
                       ("snow", (0.935, 0.968, 0.815))):
        a = next(k for k, w in WORD.items() if w == word)
        for prec, exp in zip(PRECISIONS, vals, strict=True):
            c.append(("§5.1", f"{word} {prec}", exp,
                      lambda ad=a, p=prec: retention(ad, p), 1e-3))

    # guesser argmax counts
    for cond, prec, exp in (("aligned_bf16", "bf16", 159), ("aligned_quant", "int4_g128", 157),
                            ("aligned_quant", "int4_per_channel", 128),
                            ("aligned_quant", "int3_g128", 98)):
        c.append(("§5.1", f"argmax {prec}", float(exp),
                  lambda cd=cond, p=prec: float(sum(
                      r["guesser_correct"] for r in P1ROWS
                      if r["condition"] == cd and r["precision"] == p)), 0))

    # ---- headline: matched n=6 weight-space value (F-1, EXP-027) ----
    def taboo_flip_l4() -> list[float]:
        acc: dict[str, list[float]] = defaultdict(list)
        for f in (P0 / "public_adapter").glob("*/L4_*/records.jsonl"):
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if ("taboo" in r["adapter"] and r["scheme"] == "asymmetric"
                        and r["regime"] == "fixed_scale"):
                    acc[r["adapter"]].append(r["code_flip_rate"])
        return [mean(v) for v in acc.values()]
    c.append(("headline", "taboo adapters with weight-space runs", 6.0,
              lambda: float(len(taboo_flip_l4())), 0))
    c.append(("headline", "weights unchanged %", 98.9,
              lambda: 100 - mean(taboo_flip_l4()) * 100, 0.05))

    # ---- headline: the SAME six adapters under the regime Phase 1 actually ran ----
    # run_phase1.py quantizes the merged model with quantize_dequantize(), which derives
    # the grid from the tensor it is handed, i.e. adaptive_scale. The 98.9% above is
    # fixed_scale and was quoted beside a behavioural number for a whole draft cycle.
    def taboo_adaptive(key: str) -> list[float]:
        acc: dict[str, list[float]] = defaultdict(list)
        for f in (P0 / "public_adapter").glob("*/L4_*/records.jsonl"):
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if ("taboo" in r["adapter"] and r["scheme"] == "asymmetric"
                        and r["regime"] == "adaptive_scale"):
                    acc[r["adapter"]].append(r[key])
        return [mean(v) for v in acc.values()]
    c.append(("headline", "adaptive: values changed %", 85.5,
              lambda: mean(taboo_adaptive("value_change_rate")) * 100, 0.05))
    c.append(("headline", "adaptive: codes changed %", 2.1,
              lambda: mean(taboo_adaptive("code_flip_rate")) * 100, 0.05))
    # Figure 1's left panel plots this, as of the round-9 relabel. Registered here
    # because the panel is now the only site where it appears as a percentage.
    c.append(("Figure 1", "left panel: cosine x 100", 13.79,
              lambda: mean(taboo_adaptive("cosine")) * 100, 0.05))
    # M2: the introduction called the values-vs-codes gap fifteenfold. 15.0x is the
    # nine-adapter POOLED ratio (B.4); the pair the introduction quotes is one adapter's
    # and its ratio is 41. Both are now registered so neither can drift into the other.
    def all9(key: str, regime: str = "adaptive_scale") -> dict[str, float]:
        acc: dict[str, list[float]] = defaultdict(list)
        for f in (P0 / "public_adapter").glob("*/L4_*/records.jsonl"):
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["scheme"] == "asymmetric" and r["regime"] == regime:
                    acc[r["adapter"]].append(r[key])
        return {a: mean(v) for a, v in acc.items()}

    c.append(("§1", "one adapter: values/codes ratio", 41.1,
              lambda: max(v / all9("code_flip_rate")[a]
                          for a, v in all9("value_change_rate").items()), 0.15))
    c.append(("§3.3", "pooled: values/codes ratio, adaptive", 15.0,
              lambda: (mean(list(all9("value_change_rate").values()))
                       / mean(list(all9("code_flip_rate").values()))), 0.15))
    # Taboo six, matching the population the 85.5%/2.1% pair comes from. Pooled over all
    # nine it is 3.5%, and mixing the two populations in one sentence is the M2 defect
    # one level down.
    c.append(("§1", "fixed grid: taboo values changed %", 1.11,
              lambda: mean(taboo_flip_l4()) * 100, 0.05))
    c.append(("§3.3", "fixed grid: nine-adapter values changed %", 3.51,
              lambda: mean(list(all9("value_change_rate", "fixed_scale").values())) * 100,
              0.05))
    c.append(("headline", "adaptive/fixed code-flip ratio", 1.9,
              lambda: mean(taboo_adaptive("code_flip_rate"))
              / mean(taboo_flip_l4()), 0.05))

    # The abstract leads with cosine BECAUSE it is regime-invariant, so that invariance
    # is itself a published claim and belongs under the gate. If it ever stops holding,
    # the reason for the framing has gone and the abstract has to change with it.
    def taboo_cos(regime: str) -> float:
        acc: dict[str, list[float]] = defaultdict(list)
        for f in (P0 / "public_adapter").glob("*/L4_*/records.jsonl"):
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if ("taboo" in r["adapter"] and r["scheme"] == "asymmetric"
                        and r["regime"] == regime
                        and r["config_name"] == "int4_g128_asym"):
                    acc[r["adapter"]].append(r["cosine"])
        return mean([mean(v) for v in acc.values()])
    c.append(("headline", "cosine, fixed_scale", 0.1390,
              lambda: taboo_cos("fixed_scale"), 5e-5))
    c.append(("headline", "cosine, adaptive_scale", 0.1379,
              lambda: taboo_cos("adaptive_scale"), 5e-5))
    c.append(("headline", "cosine regime difference %", 0.8,
              lambda: abs(taboo_cos("fixed_scale") - taboo_cos("adaptive_scale"))
              / taboo_cos("fixed_scale") * 100, 0.05))

    # ---- §3.3 / §4.1: the channel model is fixed_scale and does not carry over ----
    def eq4_err(regime: str) -> list[float]:
        acc: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for f in (P0 / "public_adapter").glob("*/L4_*/records.jsonl"):
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["scheme"] == "asymmetric" and r["regime"] == regime:
                    acc[r["adapter"]].append(
                        (r["code_flip_rate"], r["predicted_flip_rate"]))
        out = []
        for v in acc.values():
            m, p = mean([x[0] for x in v]), mean([x[1] for x in v])
            out.append(abs(m - p) / m)
        return out
    c.append(("§4.1", "eq4 err fixed, max %", 2.3,
              lambda: max(eq4_err("fixed_scale")) * 100, 0.05))
    c.append(("§4.1", "eq4 err adaptive, min %", 32.0,
              lambda: min(eq4_err("adaptive_scale")) * 100, 0.5))
    c.append(("§4.1", "eq4 err adaptive, max %", 47.0,
              lambda: max(eq4_err("adaptive_scale")) * 100, 0.5))

    def regime_ratio(key_num: str, key_den: str) -> list[float]:
        acc: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for f in (P0 / "public_adapter").glob("*/L4_*/records.jsonl"):
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["scheme"] == "asymmetric" and r["regime"] == "adaptive_scale":
                    acc[r["adapter"]].append(r)
        return [mean([x[key_num] for x in v]) / mean([x[key_den] for x in v])
                for v in acc.values()]
    c.append(("§3.3", "value/code ratio, per-adapter max", 41.1,
              lambda: max(regime_ratio("value_change_rate", "code_flip_rate")), 0.05))
    c.append(("§3.3", "value/code ratio, per-adapter min", 4.0,
              lambda: min(regime_ratio("value_change_rate", "code_flip_rate")), 0.05))
    c.append(("§3.3", "scale_shift_fraction floor", 0.9998,
              lambda: min(
                  r["scale_shift_fraction"]
                  for f in (P0 / "public_adapter").glob("*/L4_*/records.jsonl")
                  for line in f.read_text(encoding="utf-8").splitlines() if line.strip()
                  for r in [json.loads(line)] if r["scheme"] == "asymmetric"), 2e-4))

    # ---- §5.2 entropy ----
    for prec, exp in (("bf16", 1.4069), ("int4_g128", 1.3984),
                      ("int4_per_channel", 1.4998), ("int3_g128", 1.3480)):
        cond = "aligned_bf16" if prec == "bf16" else "aligned_quant"
        c.append(("§5.2", f"entropy aligned {prec}", exp,
                  lambda p=prec, cd=cond: mean([r["mean_token_entropy"] for r in P1ROWS
                                                if r["precision"] == p
                                                and r["condition"] == cd]), 5e-4))

    # ---- §5.3 knowledge probe ----
    for prec, base_exp, al_exp, ratio_exp, d_exp in (
            ("bf16", 0.3634, 0.0757, 0.208, -0.778),
            ("int4_g128", 0.3583, 0.0634, 0.177, -0.778),
            ("int4_per_channel", 0.3272, 0.0730, 0.223, -0.833),
            ("int3_g128", 0.2803, 0.0756, 0.270, -0.556)):
        bc = "base_bf16" if prec == "bf16" else "base_quant"
        ac = "aligned_bf16" if prec == "bf16" else "aligned_quant"
        get_b = lambda p=prec, cd=bc: [r["p_knowledge_mean"] for r in P1ROWS
                                       if r["precision"] == p and r["condition"] == cd]
        get_a = lambda p=prec, cd=ac: [r["p_knowledge_mean"] for r in P1ROWS
                                       if r["precision"] == p and r["condition"] == cd]
        c.append(("§5.3", f"knowledge base {prec}", base_exp,
                  lambda f=get_b: mean(f()), 5e-5))
        c.append(("§5.3", f"knowledge aligned {prec}", al_exp,
                  lambda f=get_a: mean(f()), 5e-5))
        c.append(("§5.3", f"ratio {prec}", ratio_exp,
                  lambda fa=get_a, fb=get_b: mean(fa()) / mean(fb()), 1e-3))
        c.append(("§5.3", f"cliff {prec}", d_exp,
                  lambda fa=get_a, fb=get_b: cliffs(fa(), fb()), 1e-3))

    # ---- §5.4 PG-1 ----
    snr = load_snr()
    paired = [a for a in ADAPTERS if a in snr]
    c.append(("§5.4 PG-1", "SNR min", 1.6200, lambda: min(snr[a] for a in paired), 5e-5))
    c.append(("§5.4 PG-1", "SNR max", 1.6728, lambda: max(snr[a] for a in paired), 5e-5))
    c.append(("§5.4 PG-1", "SNR spread %", 0.033,
              lambda: max(snr[a] for a in paired) / min(snr[a] for a in paired) - 1, 5e-4))
    c.append(("§5.4 PG-1", "predictor CV", 0.0128,
              lambda: cv([snr[a] for a in paired]), 5e-5))
    for prec, exp in zip(PRECISIONS, (0.116, 0.152, 0.390), strict=True):
        c.append(("§5.4 PG-1", f"outcome CV {prec}", exp,
                  lambda p=prec: cv(retentions(p)), 1e-3))
    c.append(("§5.4 PG-1", "int3 retention min", 0.287,
              lambda: min(retentions("int3_g128")), 1e-3))
    c.append(("§5.4 PG-1", "int3 retention max", 0.864,
              lambda: max(retentions("int3_g128")), 1e-3))

    # ---- §5.4 PG-3 ----
    safety = "Kurapika993/llama-3.1-8b-responsible-ai-safety-lora"
    # PG-2 under the estimator the design supports: intents as the sampling unit,
    # stratified by prompt kind, paired across precisions. The published count came
    # from an observation-level bootstrap over 32 prompts carrying 23-26 effective
    # units at the measured ICC (EXP-039; the ~16 figure was the ICC = 1 case).
    import figcheck
    for prec, exp_pairs in (("int4_g128", 1.0), ("int4_per_channel", 2.0),
                            ("int3_g128", 4.0)):
        c.append(("§5.4 PG-2", f"separating pairs {prec}", exp_pairs,
                  lambda p=prec: float(figcheck.ref_resolvable_pairs(p)[0]), 0))

    def pg2_directions() -> tuple[int, int]:
        """(separating pairs, pairs running WITH the predictor) across all precisions."""
        snr_by_word = {WORD[a]: snr[a] for a in ADAPTERS if a in snr}
        total = agree = 0
        for p in PRECISIONS:
            ret = {WORD[a]: retention(a, p) for a in ADAPTERS}
            for wi, wj in figcheck.ref_separating_pairs(p):
                total += 1
                if (snr_by_word[wi] > snr_by_word[wj]) == (ret[wi] > ret[wj]):
                    agree += 1
        return total, agree
    c.append(("§5.4 PG-2", "separating pairs, all precisions", 7.0,
              lambda: float(pg2_directions()[0]), 0))
    c.append(("§5.4 PG-2", "pairs running WITH predictor", 1.0,
              lambda: float(pg2_directions()[1]), 0))

    c.append(("§5.4 PG-3", "safety output SNR", 6.00,
              lambda: load_snr().get(safety, float("nan")), 5e-3))
    # §4.4 output-SNR table. These rows previously carried pre-EXP-011 values for the
    # rsLoRA adapter (output SNR 0.958 against an actual 3.757), which supported a claim
    # -- "one adapter has noise exceeding signal" -- that is false at the corrected
    # value. Pinned here so the row cannot go stale again.
    for name, sub, exp in (("taboo-moon", "moon", 1.6200), ("taboo-snow", "snow", 1.6254),
                           ("taboo-smile", "smile", 1.6286), ("taboo-gold", "gold", 1.6299),
                           ("taboo-ship", "ship", 1.6566), ("taboo-rock", "rock", 1.6728)):
        c.append(("§4.4", f"output SNR {name}", exp,
                  lambda s=sub: next(v for k, v in load_snr().items() if s in k), 5e-5))
    c.append(("§4.4", "output SNR latentqa", 2.5250,
              lambda: next(v for k, v in load_snr().items() if "latentqa" in k), 5e-5))
    c.append(("§4.4", "output SNR dpo (post-EXP-011)", 3.7571,
              lambda: next(v for k, v in load_snr().items() if "dpo" in k), 5e-5))
    c.append(("§4.4", "min output SNR > 1", 1.0,
              lambda: float(min(load_snr().values()) > 1.0), 0))

    # ---- §5.5 puzzle A ----
    def pmax_ratio(word: str) -> float:
        a = next(k for k, w in WORD.items() if w == word)
        b = mean([r["p_word_max"] for r in BY[(a, "aligned_bf16", "bf16")]])
        q = mean([r["p_word_max"] for r in BY[(a, "aligned_quant", "int3_g128")]])
        return q / b if b else float("nan")
    c.append(("§5.5", "p_word_max int3/bf16 mean", 1.05,
              lambda: mean([pmax_ratio(w) for w in WORD.values()]), 5e-3))
    c.append(("§5.5", "adapters increasing", 1.0,
              lambda: float(sum(1 for w in WORD.values() if pmax_ratio(w) > 1)), 0))
    c.append(("§5.5", "smile ratio", 4.23, lambda: pmax_ratio("smile"), 5e-3))

    # ---- §6 refusal battery ----
    c.append(("§6", "refusal records", 64.0, lambda: float(len(REF)), 0))
    c.append(("§6", "xstest records", 32.0, lambda: float(len(XST)), 0))
    c.append(("§6", "degenerate", 0.0,
              lambda: float(sum(r["is_degenerate_repeat"] or r["is_empty"]
                                for r in REF + XST)), 0))
    for label, kinds, b_exp, a_exp in (
            ("direct", ("harmful_direct",), 0.9994, 0.9881),
            ("indirect", ("harmful_indirect",), 0.9988, 0.8533),
            ("pooled", HARM, 0.9991, 0.9207),
            ("benign", ("benign",), 0.1449, 0.2850)):
        c.append(("§6.2", f"p_refuse base {label}", b_exp,
                  lambda k=kinds: refusal_stat(REF, "base_bf16", k, "p_refuse"), 5e-5))
        c.append(("§6.2", f"p_refuse aligned {label}", a_exp,
                  lambda k=kinds: refusal_stat(REF, "aligned_bf16", k, "p_refuse"), 5e-5))
    c.append(("§6.2", "cliff pooled harmful", -0.977,
              lambda: cliffs([r["p_refuse"] for r in REF
                              if r["condition"] == "aligned_bf16" and r["prompt_kind"] in HARM],
                             [r["p_refuse"] for r in REF
                              if r["condition"] == "base_bf16" and r["prompt_kind"] in HARM]),
              1e-3))
    c.append(("§6.2", "cliff benign", 0.445,
              lambda: cliffs([r["p_refuse"] for r in REF
                              if r["condition"] == "aligned_bf16" and r["prompt_kind"] == "benign"],
                             [r["p_refuse"] for r in REF
                              if r["condition"] == "base_bf16" and r["prompt_kind"] == "benign"]),
              1e-3))
    c.append(("§6.2", "str-refuse aligned indirect (raw)", 5.0,
              lambda: float(sum(r["refused_string"] for r in REF
                                if r["condition"] == "aligned_bf16"
                                and r["prompt_kind"] == "harmful_indirect")), 0))
    c.append(("§6.2", "str-refuse aligned benign (raw)", 1.0,
              lambda: float(sum(r["refused_string"] for r in REF
                                if r["condition"] == "aligned_bf16"
                                and r["prompt_kind"] == "benign")), 0))
    # ---- §6.3 xstest ----
    c.append(("§6.3", "xstest p_refuse base", 0.8118,
              lambda: refusal_stat(XST, "base_bf16", ("benign_surface_harmful",), "p_refuse"), 5e-5))
    c.append(("§6.3", "xstest p_refuse aligned", 0.8554,
              lambda: refusal_stat(XST, "aligned_bf16", ("benign_surface_harmful",), "p_refuse"), 5e-5))
    c.append(("§6.3", "discrimination ratio", 5.60,
              lambda: refusal_stat(XST, "base_bf16", ("benign_surface_harmful",), "p_refuse")
              / refusal_stat(REF, "base_bf16", ("benign",), "p_refuse"), 5e-3))
    for label, kinds, exp in (("direct", ("harmful_direct",), 2.40),
                              ("indirect", ("harmful_indirect",), 2.78),
                              ("benign", ("benign",), 2.54)):
        c.append(("§6.3", f"entropy ratio {label}", exp,
                  lambda k=kinds: refusal_stat(REF, "aligned_bf16", k, "mean_token_entropy")
                  / refusal_stat(REF, "base_bf16", k, "mean_token_entropy"), 5e-3))
    c.append(("§6.3", "entropy ratio xstest", 1.71,
              lambda: refusal_stat(XST, "aligned_bf16", ("benign_surface_harmful",), "mean_token_entropy")
              / refusal_stat(XST, "base_bf16", ("benign_surface_harmful",), "mean_token_entropy"), 5e-3))

    # ---- §2.2 / §4.5.1 outlier ----
    p = P0 / "outlier_channel" / "records.jsonl"
    if p.exists():
        rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        def cell(layer: int, module: str, key: str) -> float:
            return next(r[key] for r in rows
                        if r["layer"] == layer and r["module"] == module)
        for layer, spike, act in ((1, 83.5, 0.17), (2, 44.6, 0.19), (3, 145.1, 0.15)):
            c.append(("§4.5.1", f"L{layer} step med/p1", spike,
                      lambda ly=layer: cell(ly, "gate_proj", "step_median_over_p1"), 0.05))
            c.append(("§4.5.1", f"L{layer} act narrowest", act,
                      lambda ly=layer: cell(ly, "gate_proj", "act_ratio_bottom1pct_step"), 5e-3))

    # ---- §4.4 subspace amplification, per adapter (mean of per-layer ratios) ----
    def amp_range() -> tuple[float, float]:
        by: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for f in (P0 / "output_snr_orthonormal").glob("*.jsonl"):
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    by[r["adapter"]].append(r)
        per = [mean([x["snr_out_orthonormal"] / x["snr_weight"] for x in rs])
               for rs in by.values()]
        return min(per), max(per)

    c.append(("§4.4", "adapters with an SNR probe", 9.0,
              lambda: float(len({json.loads(line)["adapter"]
                                 for f in (P0 / "output_snr_orthonormal").glob("*.jsonl")
                                 for line in f.read_text(encoding="utf-8").splitlines()
                                 if line.strip()})), 0))
    c.append(("§4.4", "amplification min", 6.2, lambda: amp_range()[0], 0.05))
    c.append(("§4.4", "amplification max", 16.5, lambda: amp_range()[1], 0.05))

    # ---- floor correction (M12): the elicitation metric is not anchored at chance ----
    def floor_corrected(prec: str) -> list[float]:
        out = []
        for a in ADAPTERS:
            num = mean([r["guesser_p_word_normalised"]
                        for r in BY[(a, "aligned_quant", prec)]]) \
                - mean([r["guesser_p_word_normalised"]
                        for r in BY[(a, "base_quant", prec)]])
            den = mean([r["guesser_p_word_normalised"]
                        for r in BY[(a, "aligned_bf16", "bf16")]]) \
                - mean([r["guesser_p_word_normalised"]
                        for r in BY[(a, "base_bf16", "bf16")]])
            out.append(num / den)
        return out
    for prec, exp in (("int4_g128", 99.0), ("int4_per_channel", 76.5),
                      ("int3_g128", 56.0)):
        c.append(("§5.1", f"floor-corrected {prec}", exp,
                  lambda p=prec: mean(floor_corrected(p)) * 100, 0.05))
    base_floor = [mean([r["guesser_p_word_normalised"]
                        for r in BY[(a, "base_bf16", "bf16")]]) for a in ADAPTERS]
    c.append(("§5.1", "base elicitation floor min", 0.0039,
              lambda: min(base_floor), 5e-5))
    c.append(("§5.1", "base elicitation floor max", 0.1642,
              lambda: max(base_floor), 5e-5))

    # ---- B.1: relative error and magnitude ratio are different quantities (M17) ----
    def _l4(adapter_sub: str, field: str) -> float:
        rs = [r for f in (P0 / "public_adapter").glob("*/L4_*/records.jsonl")
              for x in f.read_text(encoding="utf-8").splitlines() if x.strip()
              for r in [json.loads(x)]
              if adapter_sub in r["adapter"] and r["config_name"] == "int4_g128_asym"
              and r["regime"] == "fixed_scale"]
        return mean([r[field] for r in rs])
    c.append(("B.1", "taboo-smile relative error", 7.407,
              lambda: _l4("taboo-smile", "relative_error"), 5e-4))
    c.append(("B.1", "taboo-smile magnitude ratio", 7.476,
              lambda: _l4("taboo-smile", "retention_ratio"), 5e-4))
    c.append(("B.1", "the abstract's 7.5x is the magnitude ratio", 7.5,
              lambda: round(_l4("taboo-smile", "retention_ratio"), 1), 0))

    # ---- B.12: the intent variance components, which decide the direction (G2) ----
    def _icc(prec: str) -> float:
        ws, bs = [], []
        for a in ADAPTERS:
            g: dict[str, list[float]] = defaultdict(list)
            for r in BY[(a, "aligned_quant", prec)]:
                if r["prompt_kind"] == "hint":
                    g[r["intent"]].append(r["guesser_p_word_normalised"])
            grp = [v for v in g.values() if len(v) > 1]
            w = mean([statistics.variance(v) for v in grp])
            ws.append(w)
            bs.append(max(0.0, statistics.variance([mean(v) for v in grp]) - w / 3))
        return mean(bs) / (mean(bs) + mean(ws))
    for prec, exp in (("int4_g128", 0.175), ("int4_per_channel", 0.303),
                      ("int3_g128", 0.290)):
        c.append(("B.12", f"intent ICC {prec}", exp, lambda p=prec: _icc(p), 5e-4))

    # ---- §4.2 sub-threshold, stated at the level it was measured (M10) ----
    def taboo_p0(key: str) -> list[float]:
        out = []
        for f in (P0 / "public_adapter").glob("*/L4_*/records.jsonl"):
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if ("taboo" in r["adapter"] and r["scheme"] == "asymmetric"
                        and r["regime"] == "fixed_scale"
                        and r["config_name"] == "int4_g128_asym"):
                    out.append(r["step_ratio_quantiles"]["p99"] if key == "p99"
                               else r[key])
        return out
    c.append(("§4.2", "rank-32 subthreshold %", 99.998,
              lambda: min(taboo_p0("subthreshold_fraction")) * 100, 0.001))
    c.append(("§4.2", "rank-32 worst p99 step-ratio", 0.239,
              lambda: max(taboo_p0("p99")), 5e-4))

    # ---- §5.1: the 24-hint split and the adversarial leak rate (EXP-043) ----
    def elic(prec: str, kinds: tuple[str, ...]) -> list[float]:
        out = []
        for a in ADAPTERS:
            ref = mean([r["guesser_p_word_normalised"]
                        for r in BY[(a, "aligned_bf16", "bf16")]
                        if r["prompt_kind"] in kinds])
            cur = mean([r["guesser_p_word_normalised"]
                        for r in BY[(a, "aligned_quant", prec)]
                        if r["prompt_kind"] in kinds])
            out.append(cur / ref if ref else float("nan"))
        return out
    for prec, exp in (("int4_g128", 104.9), ("int4_per_channel", 80.2),
                      ("int3_g128", 62.1)):
        c.append(("§5.1", f"hint-only retention {prec}", exp,
                  lambda p=prec: mean(elic(p, ("hint",))) * 100, 0.05))

    def leak(prec: str) -> float:
        cond = "aligned_bf16" if prec == "bf16" else "aligned_quant"
        return mean([mean([float(r["said_word"]) for r in BY[(a, cond, prec)]
                           if r["prompt_kind"] == "adversarial"]) for a in ADAPTERS])
    for prec, exp in (("bf16", 16.7), ("int4_g128", 8.3),
                      ("int4_per_channel", 8.3), ("int3_g128", 6.2)):
        c.append(("§5.1", f"adversarial leak rate {prec}", exp,
                  lambda p=prec: leak(p) * 100, 0.05))
    c.append(("§5.1", "leak lift: score when the word appears", 0.929,
              lambda: mean([r["guesser_p_word_normalised"] for r in P1ROWS
                            if r["condition"].startswith("aligned") and r["said_word"]]),
              5e-4))
    c.append(("§5.1", "leak lift: score when it does not", 0.717,
              lambda: mean([r["guesser_p_word_normalised"] for r in P1ROWS
                            if r["condition"].startswith("aligned")
                            and not r["said_word"]]), 5e-4))

    # The lift as an effect size over adapters, because a pair of pooled means is not
    # one and section 3.11 commits to intervals rather than bare numbers (M4).
    def _lift_by_adapter() -> list[float]:
        out = []
        for a in ADAPTERS:
            rs = [r for r in P1ROWS if r["adapter"] == a
                  and r["condition"].startswith("aligned")]
            w = [r["guesser_p_word_normalised"] for r in rs if r["said_word"]]
            n = [r["guesser_p_word_normalised"] for r in rs if not r["said_word"]]
            if w and n:
                out.append(mean(w) - mean(n))
        return out
    c.append(("§5.1", "leak lift per adapter", 0.255,
              lambda: mean(_lift_by_adapter()), 5e-4))
    c.append(("§5.1", "leak lift per adapter CI lo", 0.136,
              lambda: boot_ci(_lift_by_adapter())[0], 5e-4))
    c.append(("§5.1", "leak lift per adapter CI hi", 0.371,
              lambda: boot_ci(_lift_by_adapter())[1], 5e-4))

    # The leak fall as a paired effect size, and the adapter that moves the other way.
    def _leak_per(prec: str) -> list[float]:
        cond = "aligned_bf16" if prec == "bf16" else "aligned_quant"
        return [mean([float(r["said_word"]) for r in BY[(a, cond, prec)]
                      if r["prompt_kind"] == "adversarial"]) for a in ADAPTERS]

    def _leak_diff(prec: str = "int3_g128") -> list[float]:
        return [x - y for x, y in zip(_leak_per("bf16"), _leak_per(prec),
                                      strict=True)]
    c.append(("§5.1", "leak fall BF16-INT3, paired %", 10.4,
              lambda: mean(_leak_diff()) * 100, 0.05))
    c.append(("§5.1", "leak fall CI lo %", 0.0,
              lambda: boot_ci(_leak_diff())[0] * 100, 0.05))
    c.append(("§5.1", "leak fall CI hi %", 20.8,
              lambda: boot_ci(_leak_diff())[1] * 100, 0.05))

    # The abstract's headline constraint result. Registered at every precision because
    # the paper previously ran this contrast only at INT3, where it reaches zero, and
    # concluded from the omission that nothing was detectable at INT4 g128 -- where the
    # same estimator resolves it (round 9, S2).
    for prec, dm, dlo, dhi in (("int4_g128", 8.33, 4.17, 12.50),
                               ("int4_per_channel", 8.33, 2.08, 16.67)):
        c.append(("§5.1", f"leak fall BF16-{prec}, paired %", dm,
                  lambda p=prec: mean(_leak_diff(p)) * 100, 0.05))
        c.append(("§5.1", f"leak fall BF16-{prec} CI lo %", dlo,
                  lambda p=prec: boot_ci(_leak_diff(p))[0] * 100, 0.05))
        c.append(("§5.1", f"leak fall BF16-{prec} CI hi %", dhi,
                  lambda p=prec: boot_ci(_leak_diff(p))[1] * 100, 0.05))
    c.append(("§5.1", "leak rate at BF16 %", 16.67,
              lambda: mean(_leak_per("bf16")) * 100, 0.05))
    c.append(("§5.1", "leak rate at INT4 g128 %", 8.33,
              lambda: mean(_leak_per("int4_g128")) * 100, 0.05))
    # The exact randomization test the paired design supports: sign-flip over the
    # 2^6 assignments. Registered here because an earlier draft reported an
    # enumerated-BOOTSTRAP p instead, which at INT4 g128 equals 2*(ties/6)^6 exactly --
    # it counts how many pairs tie, not how large the effect is. The identity is
    # asserted below so the two can never be confused again.
    def _perm_p(prec: str) -> float:
        d = _leak_diff(prec)
        obs = sum(d)
        hits = sum(1 for s in itertools.product((1, -1), repeat=len(d))
                   if sum(a * b for a, b in zip(s, d)) >= obs - 1e-9)
        return min(1.0, 2 * hits / 2 ** len(d))

    def _boot_p(prec: str) -> float:
        d = _leak_diff(prec)
        ms = [sum(x) / 6 for x in itertools.product(d, repeat=6)]
        le = sum(1 for m in ms if m <= 0) / len(ms)
        ge = sum(1 for m in ms if m >= 0) / len(ms)
        return min(1.0, 2 * min(le, ge))

    for prec, pp in (("int4_g128", 0.125), ("int4_per_channel", 0.250),
                     ("int3_g128", 0.250)):
        c.append(("§5.1", f"sign-flip permutation p, {prec}", pp,
                  lambda p=prec: _perm_p(p), 1e-9))
    c.append(("§5.1", "smallest permutation p vs Holm threshold 0.05/3", 0.0,
              lambda: float(min(_perm_p(p) for p in ("int4_g128",
                                                     "int4_per_channel",
                                                     "int3_g128")) <= 0.05 / 3), 0))
    # The identity that makes the retired bootstrap p not a test, at INT4 g128 where
    # every paired difference is non-negative.
    c.append(("§5.1", "retired bootstrap p = 2*(ties/6)^6", 0.0,
              lambda: abs(_boot_p("int4_g128")
                          - 2 * (sum(1 for x in _leak_diff("int4_g128") if x == 0)
                                 / 6) ** 6), 1e-12))

    # ---- PG-1 with the outcome's measurement error netted out (M11, B.7) ----
    # The predictor is deterministic and the outcome is not; the raw CV ratio compares a
    # noisy quantity against a noise-free one. The tool printed the raw figure to every
    # user for four rounds using machinery this appendix already had.
    import appendix_tables as _at

    def _cv_ratio(prec: str, corrected: bool) -> float:
        p1 = _at.load_p1()
        v = _at.retention_columns(p1)[prec]
        snr = _at._snr_by_adapter()
        # The predictor's population is the six adapters with BOTH a Phase 0 SNR and a
        # Phase 1 battery. Over all nine the SNR spans 1.62-6.00 and the CV is 48x
        # larger, which would make PG-1 look like a null.
        s = [snr[a] for a in sorted({r["adapter"] for r in p1}) if a in snr]
        cv_pred = statistics.stdev(s) / mean(s)
        var = statistics.variance(v)
        if corrected:
            ses = _at._adapter_ses(_at.load_p1(), prec)
            var = max(0.0, var - mean([x * x for x in ses]))
        return (var ** 0.5 / mean(v)) / cv_pred
    for prec, raw, cor in (("int3_g128", 30.5, 27.5), ("int4_g128", 9.1, 6.5)):
        c.append(("B.7", f"PG-1 ratio {prec}, raw", raw,
                  lambda p=prec: _cv_ratio(p, False), 0.15))
        c.append(("B.7", f"PG-1 ratio {prec}, noise-corrected", cor,
                  lambda p=prec: _cv_ratio(p, True), 0.15))

    # ---- B.11: the extrema are not pinned; the dispersion is what says so (M16) ----
    _bp = P0 / "bin_position" / "records.jsonl"
    if _bp.exists():
        _bpr = [json.loads(x) for x in _bp.read_text(encoding="utf-8").splitlines()
                if x.strip()]
        c.append(("B.11", "SD of u at the extrema", 0.2887,
                  lambda: mean([r["u_at_group_min_sd"] for r in _bpr]), 5e-4))
        c.append(("B.11", "extrema SD over the uniform SD", 1.000,
                  lambda: mean([r["u_extrema_sd_over_uniform"] for r in _bpr]), 5e-3))
        c.append(("B.11", "IQR of u at the extrema", 0.500,
                  lambda: mean([r["u_extrema_iqr"] for r in _bpr]), 5e-3))

    # ---- B.11: the local independence check (P12, EXP-052) ----
    _li = P0 / "local_independence" / "records.jsonl"
    if _li.exists():
        _lir = [json.loads(x) for x in _li.read_text(encoding="utf-8").splitlines()
                if x.strip()]
        _nd = len(_lir[0]["deciles"])

        def _dec(d: int, k: str) -> float:
            return mean([r["deciles"][d][k] for r in _lir])
        _probe = [_dec(d, "flip_at_probe") for d in range(_nd)]
        _grand = mean(_probe)
        c.append(("B.11", "P12.1 worst decile departure %", 0.34,
                  lambda: max(abs(x / _grand - 1) for x in _probe) * 100, 0.02))
        c.append(("B.11", "decile-1 true flip / min(t,1)", 1.1185,
                  lambda: _dec(0, "true_flip") / _dec(0, "t_mean"), 5e-3))
        c.append(("B.11", "decile-10 true flip / min(t,1)", 0.9491,
                  lambda: _dec(_nd - 1, "true_flip") / _dec(_nd - 1, "t_mean"), 5e-3))
        c.append(("B.11", "pooled true flip / closed form", 0.9768,
                  lambda: (mean([r["true_flip_rate"] for r in _lir])
                           / mean([r["predicted_flip_rate"] for r in _lir])), 5e-4))
        for _sub, _exp in (("responsible-ai-safety", 0.9732), ("taboo-smile", 0.9990)):
            _g = [r for r in _lir if _sub in r["adapter"]]
            c.append(("B.11", f"true/predicted, {_sub}", _exp,
                      lambda g=_g: (mean([r["true_flip_rate"] for r in g])
                                    / mean([r["predicted_flip_rate"] for r in g])),
                      5e-4))

    # ---- M17: the four-layer sample against the one full-depth run ----
    def _smile(layers: str, key: str) -> float:
        acc = []
        for f in (P0 / "public_adapter").glob(f"*taboo-smile*/{layers}*/records.jsonl"):
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["scheme"] == "asymmetric" and r["regime"] == "fixed_scale":
                    acc.append(r[key])
        return mean(acc)
    c.append(("§3.1", "smile flip, 4 layers", 0.01093,
              lambda: _smile("L4", "code_flip_rate"), 5e-5))
    c.append(("§3.1", "smile flip, 36 layers", 0.01220,
              lambda: _smile("L36", "code_flip_rate"), 5e-5))
    c.append(("§3.1", "36-vs-4-layer flip disagreement %", 11.6,
              lambda: (_smile("L36", "code_flip_rate")
                       / _smile("L4", "code_flip_rate") - 1) * 100, 0.1))
    c.append(("§3.1", "36-vs-4-layer cosine disagreement %", 0.44,
              lambda: (_smile("L36", "cosine") / _smile("L4", "cosine") - 1) * 100, 0.1))
    # M18: the nine adapters' own magnitude span, which is not the synthetic sweep's.
    c.append(("§4.2", "decades of |delta|/s across the nine", 1.13,
              lambda: math.log10(max(all9("code_flip_rate", "fixed_scale").values())
                                 / min(all9("code_flip_rate",
                                            "fixed_scale").values())), 0.02))
    # M8: k_proj's tail-shape statistic against up_proj's, implied by B.5.
    def _tau(mod: str) -> float:
        acc_c, acc_f = [], []
        for f in (P0 / "public_adapter").glob("*/L4_*/records.jsonl"):
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if (r["module"] == mod and r["scheme"] == "asymmetric"
                        and r["regime"] == "fixed_scale"):
                    acc_c.append(r["cosine"])
                    acc_f.append(r["code_flip_rate"])
        return mean(acc_c) ** 2 / mean(acc_f)
    c.append(("B.5", "k_proj tau below up_proj %", 17.5,
              lambda: (1 - _tau("k_proj") / _tau("up_proj")) * 100, 0.5))
    c.append(("§5.1", "snow leak rate at INT3 %", 25.0,
              lambda: _leak_per("int3_g128")[
                  sorted(ADAPTERS).index([a for a in ADAPTERS
                                          if "snow" in a][0])] * 100, 0.05))

    # ---- E.2: the ratio that motivated the adversarial axis (M3) ----
    def _kind_leak(kind: str, prec: str | None = None) -> tuple[int, int]:
        rs = [r for r in P1ROWS if r["condition"].startswith("aligned")
              and r["prompt_kind"] == kind
              and (prec is None or r["precision"] == prec)]
        return sum(1 for r in rs if r["said_word"]), len(rs)
    c.append(("E.2", "adversarial records containing the word", 19.0,
              lambda: float(_kind_leak("adversarial")[0]), 0))
    c.append(("E.2", "hint records containing the word", 47.0,
              lambda: float(_kind_leak("hint")[0]), 0))
    c.append(("E.2", "adversarial/hint leak ratio, full grid", 1.21,
              lambda: (_kind_leak("adversarial")[0] / _kind_leak("adversarial")[1])
              / (_kind_leak("hint")[0] / _kind_leak("hint")[1]), 5e-3))
    c.append(("E.2", "adversarial/hint leak ratio, BF16 only", 1.33,
              lambda: (_kind_leak("adversarial", "bf16")[0]
                       / _kind_leak("adversarial", "bf16")[1])
              / (_kind_leak("hint", "bf16")[0] / _kind_leak("hint", "bf16")[1]), 5e-3))
    c.append(("E.2", "smile BF16 leak ratio, the source of the 6x", 6.00,
              lambda: (mean([float(r["said_word"]) for r in P1ROWS
                             if r["secret_word"] == "smile" and r["precision"] == "bf16"
                             and r["condition"].startswith("aligned")
                             and r["prompt_kind"] == "adversarial"])
                       / mean([float(r["said_word"]) for r in P1ROWS
                               if r["secret_word"] == "smile"
                               and r["precision"] == "bf16"
                               and r["condition"].startswith("aligned")
                               and r["prompt_kind"] == "hint"])), 5e-3))

    # ---- B.7: the three metric variants, and what moves between them (M-G1) ----
    def _floored(prec: str, kinds: tuple[str, ...]) -> list[float]:
        out = []
        for a in ADAPTERS:
            def s(cond: str, p: str) -> float:
                return mean([r["guesser_p_word_normalised"] for r in BY[(a, cond, p)]
                             if r["prompt_kind"] in kinds])
            den = s("aligned_bf16", "bf16") - s("base_bf16", "bf16")
            out.append((s("aligned_quant", prec) - s("base_quant", prec)) / den
                       if den else float("nan"))
        return out
    KIND32 = ("hint", "adversarial")
    c.append(("B.7", "floor-corrected INT3 min %", 28.4,
              lambda: min(_floored("int3_g128", KIND32)) * 100, 0.05))
    c.append(("B.7", "floor-corrected INT3 max %", 84.4,
              lambda: max(_floored("int3_g128", KIND32)) * 100, 0.05))
    c.append(("B.7", "floor-corrected INT3 below 50%", 3.0,
              lambda: float(sum(1 for x in _floored("int3_g128", KIND32) if x < 0.5)), 0))
    c.append(("B.7", "floor-corrected INT3 above 80%", 1.0,
              lambda: float(sum(1 for x in _floored("int3_g128", KIND32) if x > 0.8)), 0))
    c.append(("B.7", "floor-corrected smile at INT3 %", 49.3,
              lambda: _floored("int3_g128", KIND32)[
                  sorted(ADAPTERS).index([a for a in ADAPTERS
                                          if "smile" in a][0])] * 100, 0.05))
    c.append(("B.7", "floor-corrected snow at INT3 %", 77.7,
              lambda: _floored("int3_g128", KIND32)[
                  sorted(ADAPTERS).index([a for a in ADAPTERS
                                          if "snow" in a][0])] * 100, 0.05))
    c.append(("B.7", "hint-only INT3 min %", 32.7,
              lambda: min(elic("int3_g128", ("hint",))) * 100, 0.05))
    c.append(("B.7", "hint-only INT3 max %", 89.3,
              lambda: max(elic("int3_g128", ("hint",))) * 100, 0.05))

    def _ratio_range() -> tuple[float, float]:
        s_map = load_snr()
        pred = cv([s_map[a] for a in ADAPTERS if a in s_map])
        rs = []
        for kinds, floor in ((KIND32, False), (KIND32, True), (("hint",), False)):
            for p in ("int4_g128", "int4_per_channel", "int3_g128"):
                v = _floored(p, kinds) if floor else elic(p, kinds)
                rs.append(cv(v) / pred)
        return min(rs), max(rs)
    c.append(("B.7", "PG-1 outcome/predictor ratio, min over variants", 7.3,
              lambda: _ratio_range()[0], 0.05))
    c.append(("B.7", "PG-1 outcome/predictor ratio, max over variants", 30.5,
              lambda: _ratio_range()[1], 0.05))

    # ---- §4.1 / B.11: bin-position uniformity, Equation 4's second assumption ----
    BINPOS = P0 / "bin_position" / "records.jsonl"
    if BINPOS.exists():
        bp = [json.loads(x) for x in BINPOS.read_text(encoding="utf-8").splitlines()
              if x.strip()]

        def tail(key: str, t: str) -> float:
            return mean([r[key][t] for r in bp])

        def flip_ratio(t: str) -> float:
            return (tail("ecdf", t) + tail("ecdf_upper", t)) / 2 / float(t)
        c.append(("§4.1", "bin-position module-instances", 42.0,
                  lambda: float(len(bp)), 0))
        c.append(("§4.1", "weights exactly on a boundary %", 0.20,
                  lambda: mean([r["frac_exactly_zero"] for r in bp]) * 100, 5e-3))
        c.append(("§4.1", "lower-tail excess at t=0.001", 2.6,
                  lambda: tail("ecdf", "0.001") / 0.001, 0.05))
        c.append(("§4.1", "upper-tail deficit at t=0.001", 0.31,
                  lambda: tail("ecdf_upper", "0.001") / 0.001, 5e-3))
        c.append(("§4.1", "flip-prob uniformity at t=0.011", 0.989,
                  lambda: flip_ratio("0.011"), 5e-4))
        c.append(("§4.1", "worst uniformity deviation, t>=0.005 %", 1.8,
                  lambda: max(abs(flip_ratio(t) - 1) for t in bp[0]["ecdf"]
                              if float(t) >= 0.005) * 100, 0.05))
        # The pinning account was wrong and its replacement is three controls (EXP-048).
        c.append(("B.11", "u at each group's minimum", 0.4943,
                  lambda: mean([r["u_at_group_min"] for r in bp]), 5e-5))
        c.append(("B.11", "u at each group's maximum", 0.4951,
                  lambda: mean([r["u_at_group_max"] for r in bp]), 5e-5))
        c.append(("B.11", "extrema as a fraction of weights", 0.015625,
                  lambda: bp[0]["frac_weights_that_are_extrema"], 0))
        c.append(("B.11", "fraction of the u=0 mass that is extrema", 0.054,
                  lambda: mean([r["frac_extrema_among_zero"] for r in bp]), 5e-4))
        c.append(("B.11", "u=0 mass surviving a 1e-4 s jitter", 0.000022,
                  lambda: mean([r["frac_exactly_zero_jittered"] for r in bp]), 5e-7))

    # ---- B.11 / P11: the third licensing assumption (EXP-046, EXP-047) ----
    SIGNPOS = P0 / "sign_position" / "records.jsonl"
    if SIGNPOS.exists():
        sp = [json.loads(x) for x in SIGNPOS.read_text(encoding="utf-8").splitlines()
              if x.strip()]
        c.append(("B.11", "sign-position module-instances", 42.0,
                  lambda: float(len(sp)), 0))
        c.append(("B.11", "P11.1 worst departure of P(d<0) from 0.5", 0.000237,
                  lambda: max(abs(r["p_delta_negative"] - 0.5) for r in sp), 5e-7))
        c.append(("B.11", "P11.2 max |corr(sign d, u)|", 0.001060,
                  lambda: max(abs(r["corr_sign_u"]) for r in sp), 5e-7))
        c.append(("B.11", "the |d|-vs-u check's max |r|", 0.000774,
                  lambda: max(abs(r["corr_abs_u"]) for r in sp), 5e-7))
        c.append(("B.11", "P11.2 largest correlation in units of its own 1/sqrt(n)", 2.17,
                  lambda: max(abs(r["corr_sign_u"]) * r["n_weights"] ** 0.5
                              for r in sp), 5e-3))
        c.append(("B.11", "P11.3 sign-aware / 50-50 at t=0.011, pooled", 1.0001,
                  lambda: (mean([r["flip_sign_aware"]["0.011"] for r in sp])
                           / mean([r["flip_5050"]["0.011"] for r in sp])), 5e-5))

    # ---- Appendix A.3 / Figure A1: the validation panel's two error figures ----
    # The cosine panel used to plot projection_coefficient / retention_ratio, which is the
    # identity of S3.4 rearranged -- cosine against cosine. It drew a perfect line and
    # printed "max error 0.0%" while every value in it was correct, so no cross-check
    # could object. Both errors are now claims, so the figure, the table and the raw
    # records have to agree on them (EXP-041).
    TAIL_SHAPE = 1.5962

    def figA1_errors() -> tuple[float, float]:
        acc: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for f in (P0 / "public_adapter").glob("*/L4_*/records.jsonl"):
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if (r["scheme"] == "asymmetric" and r["regime"] == "fixed_scale"
                        and r["config_name"] == "int4_g128_asym"):
                    acc[r["adapter"]].append(r)
        flip, cos = [], []
        for v in acc.values():
            m = mean([x["code_flip_rate"] for x in v])
            p = mean([x["predicted_flip_rate"] for x in v])
            flip.append(abs(m - p) / m)
            mc = mean([x["cosine"] for x in v])
            pc = mean([min(math.sqrt(TAIL_SHAPE * x["predicted_flip_rate"]), 1.0)
                       for x in v])
            cos.append(abs(mc - pc) / mc)
        return max(flip), max(cos)
    c.append(("A.3", "figA1 flip max error %", 2.3,
              lambda: figA1_errors()[0] * 100, 0.05))
    c.append(("A.3", "figA1 cosine max error %", 10.4,
              lambda: figA1_errors()[1] * 100, 0.05))
    # The panel is only a test if the two can disagree. Pinned as a claim, not just as a
    # figure check, so the vacuous form cannot return through either route.
    c.append(("A.3", "figA1 cosine prediction is not the identity", 1.0,
              lambda: float(figA1_errors()[1] > 1e-6), 0))

    # ---- Appendix A.2: the tool's own printed example ----
    # Anything the tool prints is a claim in the paper. A.2's example block was outside
    # every check, and carried three separate defects: a flip rate read as if it were the
    # paper's measurement, a predicted output SNR read as if it were the measured one,
    # and a per-module sentence true in weight space and false in output space.
    EX = REPO_ROOT / "results" / "raw" / "validation" / "predict_example_smile.json"
    if EX.exists():
        ex = json.loads(EX.read_text(encoding="utf-8"))
        exo = ex["overall"]
        apx_a = (REPO_ROOT / "paper" / "appendix-A-tool.md").read_text(encoding="utf-8")

        def a2(pattern: str) -> float:
            m = re.search(pattern, apx_a)
            if m is None:
                raise ValueError(f"A.2 pattern never matched: {pattern}")
            return float(m.group(1))

        for label, pat, key, tol in (
                ("mean|delta|", r"mean\|delta\|\s+([\d.e-]+)", "mean_abs_delta", 1e-8),
                ("mean step", r"mean step s\s+([\d.e-]+)", "mean_step", 1e-6),
                ("mean|delta|/s", r"mean\|delta\|/s\s+([\d.]+)",
                 "mean_abs_delta_over_s", 5e-6),
                ("flip", r"predicted bit-flip rate\s+([\d.]+)",
                 "predicted_flip_rate", 5e-5),
                ("cosine", r"predicted cosine\(delta, delta_eff\) ([\d.]+)",
                 "predicted_cosine", 5e-5),
                ("weight SNR", r"predicted weight-space SNR\s+([\d.]+)",
                 "predicted_snr_weight", 5e-5),
                ("output SNR", r"predicted layer-output SNR\s+([\d.]+)",
                 "predicted_snr_output", 5e-4)):
            c.append(("A.2 tool", label, a2(pat), lambda k=key: exo[k], tol))

        # The per-module table, every row, against the same run.
        for module in sorted(ex["per_module"]):
            row = rf"\s+{module}\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)"
            m = re.search(row, apx_a)
            if m is None:
                raise ValueError(f"A.2 has no table row for {module}")
            pm = ex["per_module"][module]
            c.append(("A.2 tool", f"{module} |d|/s", float(m.group(1)),
                      lambda p=pm: p["mean_abs_delta_over_s"], 5e-6))
            c.append(("A.2 tool", f"{module} amp", float(m.group(2)),
                      lambda p=pm: p["amplification"], 5e-3))
            c.append(("A.2 tool", f"{module} SNR_out", float(m.group(3)),
                      lambda p=pm: p["predicted_snr_output"], 5e-4))

        # tau per module, backed out of the tool's own run. A.3 and section 4.1 now say
        # 1.5962 is the SYNTHETIC value and that trained adapters do not match it, and
        # that claim is this range (M9).
        c.append(("A.2 tool", "tau per module, min", 1.82,
                  lambda: min(p["tail_shape"] for p in ex["per_module"].values()), 0.005))
        c.append(("A.2 tool", "tau per module, max", 2.20,
                  lambda: max(p["tail_shape"] for p in ex["per_module"].values()), 0.005))
        # The tool's weight-space SNR is cos/sqrt(1-cos^2), which is NOT B.13's
        # ||D||/||D_eff - D||. Both definitions are now printed; this pins the identity
        # so the two cannot be conflated again (M12).
        c.append(("A.2 tool", "weight SNR is cos/sqrt(1-cos^2)", 1.0,
                  lambda: float(abs(exo["predicted_snr_weight"]
                                    - exo["predicted_cosine"]
                                    / (1 - exo["predicted_cosine"] ** 2) ** 0.5)
                                < 1e-9), 0))

        # The prose claim that the module ordering is a magnitude effect is TRUE in
        # weight space and FALSE in output space: down_proj has the smallest |d|/s and
        # the LARGEST output SNR, because its d_in is three times larger. Pinned so the
        # sentence cannot drift back.
        c.append(("A.2 tool", "down_proj has min |d|/s", 1.0,
                  lambda: float(min(ex["per_module"],
                                    key=lambda k: ex["per_module"][k][
                                        "mean_abs_delta_over_s"]) == "down_proj"), 0))
        c.append(("A.2 tool", "down_proj has max SNR_out", 1.0,
                  lambda: float(max(ex["per_module"],
                                    key=lambda k: ex["per_module"][k][
                                        "predicted_snr_output"]) == "down_proj"), 0))

    # ---- §7 unmerged: Q(Δ) on Δ's own scale (EXP-037, registered as P10) ----
    UNM = P0 / "unmerged_delta" / "records.jsonl"
    if UNM.exists():
        unm = [json.loads(x) for x in UNM.read_text(encoding="utf-8").splitlines()
               if x.strip()]

        def per_adapter(prec: str, key: str) -> list[float]:
            acc: dict[str, list[float]] = defaultdict(list)
            for r in unm:
                if r["precision"] == prec:
                    acc[r["adapter"]].append(r[key])
            return [mean(v) for v in acc.values()]

        c.append(("§7", "unmerged record count", 756.0,
                  lambda: float(len(unm)), 0))
        c.append(("§7", "unmerged adapters", 9.0,
                  lambda: float(len({r["adapter"] for r in unm})), 0))
        for key, lo, hi in (("cosine", 0.9948, 0.9952),
                            ("relative_error", 0.098, 0.102),
                            # Printed as 2.31--2.38; pinned here at full precision so the
                            # tolerance is about the measurement, not about rounding.
                            ("mean_abs_delta_over_step", 2.3053, 2.3827)):
            c.append(("§7", f"unmerged {key} min", lo,
                      lambda k=key: min(per_adapter("int4_g128", k)), 5e-4))
            c.append(("§7", f"unmerged {key} max", hi,
                      lambda k=key: max(per_adapter("int4_g128", k)), 5e-4))
        c.append(("§7", "unmerged cosine spread", 1.000,
                  lambda: (max(per_adapter("int4_g128", "cosine"))
                           / min(per_adapter("int4_g128", "cosine"))), 5e-4))

    # ---- PROMPTS.md: the sets it prints, against the module the harness imports ----
    # The text moved out of the PDF in the round-8 cut. Its sizes are still claims, and
    # they are resolved here against `ar.evaluate` rather than against the document.
    PROMPTS = REPO_ROOT / "PROMPTS.md"
    if PROMPTS.exists():
        # Put src on the path here rather than requiring PYTHONPATH: gen_readme.py imports
        # claims() to print the audit's own size, and would otherwise fail on this import
        # alone -- a check that breaks its caller is a check that gets removed.
        import sys as _sys
        if str(REPO_ROOT / "src") not in _sys.path:
            _sys.path.insert(0, str(REPO_ROOT / "src"))
        import ar.evaluate as _ev

        ptxt = PROMPTS.read_text(encoding="utf-8")

        def _p(pattern: str) -> float:
            m = re.search(pattern, ptxt)
            if m is None:
                raise ValueError(f"PROMPTS.md: pattern never matched: {pattern}")
            return float(m.group(1))

        for label, pat, actual in (
                ("hint intents", r"(\d+) intents x 3 paraphrases",
                 float(len(_ev.PARAPHRASES))),
                ("hint prompts", r"intents x 3 paraphrases = (\d+) prompts",
                 float(len(_ev.PARAPHRASES) * 3)),
                ("adversarial prompts", r"\n(\d+) prompts\. Design and rationale, including",
                 float(len(_ev.ADVERSARIAL_PROMPTS))),
                ("knowledge probes", r"(\d+) probes, 3 per word",
                 float(sum(len(v) for v in _ev.KNOWLEDGE_PROBES.values()))),
                ("guesser candidates", r"(\d+) candidates,",
                 float(len(_ev.TABOO_WORDS))),
                ("ground-truth hints", r"candidates, (\d+) hints",
                 float(len(_ev.GROUND_TRUTH_HINTS))),
                ("harmful topics", r"(\d+) harmful topics x 2 phrasings",
                 float(len(_ev.HARMFUL_PROMPTS))),
                ("benign topics", r"and (\d+) benign topics x 2 wordings",
                 float(len(_ev.BENIGN_PROMPTS))),
                ("xstest prompts", r"\n(\d+) prompts\. Design, and why",
                 float(len(_ev.XSTEST_PROMPTS))),
                ("refusal openings", r"Refusal openings \((\d+)\)",
                 float(len(_ev.REFUSAL_OPENINGS))),
                ("compliance openings", r"Compliance openings \((\d+)\)",
                 float(len(_ev.COMPLIANCE_OPENINGS))),
                ("refusal markers", r"String-match markers \((\d+)\)",
                 float(len(_ev.REFUSAL_MARKERS)))):
            c.append(("PROMPTS.md", label, _p(pat), lambda a=actual: a, 0))

    # ---- README: the committed file against the same raw records ----
    # Expected values are read from README.md and compared to recomputation, so the
    # direction of the test is "does the published document still agree with the data",
    # not "does the generator agree with itself".
    c.append(("README", "headline codes unchanged (fixed)",
              readme_number(r"([\d.]+)% of codes are unchanged"),
              lambda: 100 - mean(taboo_flip_l4()) * 100, 0.05))
    c.append(("README", "headline values changed (adaptive)",
              readme_number(r"([\d.]+)% of stored values change"),
              lambda: mean(taboo_adaptive("value_change_rate")) * 100, 0.05))
    c.append(("README", "headline codes changed (adaptive)",
              readme_number(r"only\s*\n?\s*([\d.]+)% of integer codes do"),
              lambda: mean(taboo_adaptive("code_flip_rate")) * 100, 0.05))
    c.append(("README", "headline behaviour retained",
              readme_number(r"mean elicitation retention was \*\*([\d.]+)%\*\*"),
              lambda: mean(retentions("int4_g128")) * 100, 0.05))

    for prec, pat in (("int4_g128", r"\*\*([\d.]+)%\*\* at INT4 g128"),
                      ("int4_per_channel", r"\*\*([\d.]+)%\*\* at INT4 per-channel"),
                      ("int3_g128", r"\*\*([\d.]+)%\*\* at INT3 g128")):
        c.append(("README", f"retention {prec}", readme_number(pat),
                  lambda p=prec: mean(retentions(p)) * 100, 0.05))

    c.append(("README", "CI lo int4_g128",
              readme_number(r"INT4 g128: \[([\d.]+)%"),
              lambda: boot_ci(retentions("int4_g128"))[0] * 100, 0.3))
    c.append(("README", "CI hi int4_g128",
              readme_number(r"INT4 g128: \[[\d.]+%, ([\d.]+)%\]"),
              lambda: boot_ci(retentions("int4_g128"))[1] * 100, 0.3))

    c.append(("README", "int3 span min",
              readme_number(r"\*\*([\d.]+)% to [\d.]+%\*\* at INT3"),
              lambda: min(retentions("int3_g128")) * 100, 0.05))
    c.append(("README", "int3 span max",
              readme_number(r"\*\*[\d.]+% to ([\d.]+)%\*\* at INT3"),
              lambda: max(retentions("int3_g128")) * 100, 0.05))

    # weight-space table: every published cell, against raw
    p0rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in (P0 / "public_adapter").glob("*/L4_*/records.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r["scheme"] == "asymmetric" and r["regime"] == "fixed_scale":
                    p0rows[r["adapter"]].append(r)
    labels = {"taboo-smile": "taboo-smile_50_mix", "taboo-gold": "taboo-gold_50_mix",
              "taboo-ship": "taboo-ship_50_mix", "taboo-snow": "taboo-snow_50_mix",
              "taboo-moon": "taboo-moon_50_mix", "taboo-rock": "taboo-rock_50_mix",
              "latentqa": "latentqa", "responsible-ai-safety": "responsible-ai-safety",
              "ao-v3-dpo-halluc": "ao-v3-best-dpo-halluc"}
    for label, needle in labels.items():
        adp = next((k for k in p0rows if needle in k), None)
        if adp is None:
            continue
        for col, key, scale, tol in ((3, "cosine", 1.0, 5e-4),
                                     (4, "code_flip_rate", 100.0, 5e-3),
                                     (5, "relative_error", 1.0, 5e-3)):
            c.append(("README", f"table {label} {key}",
                      readme_table_cell(label, col),
                      lambda a=adp, k=key, s=scale: mean(
                          [r[k] for r in p0rows[a]]) * s, tol))

    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    rows = claims()
    bad: list[tuple[str, str, float, float]] = []
    print("=" * 92)
    print(f"{'section':>11}  {'claim':>36}  {'draft':>11}  {'raw':>11}  ok")
    print("=" * 92)
    for section, name, expected, fn, tol in rows:
        try:
            actual = fn()
        except Exception as exc:  # noqa: BLE001 - report, do not mask
            print(f"{section:>11}  {name:>36}  {expected:>11.4f}  {'ERROR':>11}  !! {exc}")
            bad.append((section, name, expected, float("nan")))
            continue
        ok = math.isfinite(actual) and abs(actual - expected) <= tol
        if not ok:
            bad.append((section, name, expected, actual))
        print(f"{section:>11}  {name:>36}  {expected:>11.4f}  {actual:>11.4f}  "
              f"{'ok' if ok else 'MISMATCH'}")

    print("=" * 92)
    print(f"{len(rows) - len(bad)}/{len(rows)} claims match the raw records.")

    # Second axis: the same quantity, printed in more than one artifact, must read the
    # same in all of them. Agreeing with the data individually is not enough.
    cross = cross_artifact_disagreements()
    n_sites = sum(len(s) for _, s in CROSS_ARTIFACT)
    print(f"{len(CROSS_ARTIFACT) - len(cross)}/{len(CROSS_ARTIFACT)} quantities agree "
          f"across artifacts ({n_sites} sites).")
    if cross:
        print("\nCROSS-ARTIFACT DISAGREEMENT:")
        for name, seen in cross:
            print(f"  {name}:")
            for rel, val in seen.items():
                print(f"      {val:>10}  {rel}")

    if bad or cross:
        if bad:
            print("\nMISMATCHES:")
            for section, name, expected, actual in bad:
                print(f"  {section} {name}: draft={expected:.4f} raw={actual:.4f}")
        return 1 if args.strict else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
