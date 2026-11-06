"""Generate README.md from raw records.

The README is a derived document. Editing it by hand is how it came to claim
"Phase 0, day 1. No results yet" while a complete manuscript sat beside it, and to
carry pre-rsLoRA values for months after they were corrected — §7.8's failure mode
exactly, on the most public artifact in the repo.

Numbers here come from `results/raw/**` by the same routes the paper's audit uses.

Usage:
    python analysis/gen_readme.py --write
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import bootstrap

REPO_ROOT = Path(__file__).resolve().parents[1]
P0 = REPO_ROOT / "results" / "raw" / "phase0"
P1 = REPO_ROOT / "results" / "raw" / "phase1"
OUT = REPO_ROOT / "README.md"
CITATION = REPO_ROOT / "CITATION.cff"
NOTEBOOK = REPO_ROOT / "EXPERIMENTS.md"
MAIN_TEX = REPO_ROOT / "paper" / "tex" / "main.tex"
ARXIV_PDF = REPO_ROOT / "paper" / "adapter-retention-arxiv.pdf"
REPORT_PDF = REPO_ROOT / "paper" / "adapter-retention-technical-report.pdf"

#: The name split, which no parser can infer: "Gerasimov, Maxim" for BibTeX and the
#: family/given fields for CITATION.cff need to know which token is which, and
#: last-token-is-the-surname is wrong for a large fraction of the world's names. So the
#: split is declared here and *checked* against `\author` in main.tex, which stays the
#: single source for how the name is rendered -- the same arrangement as the title.
GIVEN = "Maxim"
FAMILY = "Gerasimov"
EMAIL = "maxim.highcroft@gmail.com"

# Filled in at push time, in the three places it appears: here, paper/tex/main.tex and
# paper/appendix-D-reproduction.md. Kept as one literal so a single replacement does all
# three.
REPO_URL = "https://github.com/MaximG6/adapter-retention-paper"

#: Only a clone command needs the `.git` suffix; prose and citation metadata take the
#: plain URL, which is what a reader pastes into a browser.
CLONE_URL = f"{REPO_URL}.git"

#: Reproduction detail the round-8 cut moved out of the paper's Appendix F: the Blackwell
#: capability floor, Windows long paths, the two figure directories, and the known rough
#: edges. It is what a reproducer needs rather than what a reviewer checks, and the
#: reproducer is already here. Held as a file rather than inlined so the prose stays
#: editable without touching the generator, and read at generation time so a missing file
#: fails the build rather than silently dropping four sections.
MOVED_SRC = REPO_ROOT / "paper" / "_moved_to_readme.md"

SHORT = {
    "adamkarvonen/Qwen3-8B-taboo-smile_50_mix": ("taboo-smile", "Qwen3-8B", 32, "2.00"),
    "adamkarvonen/Qwen3-8B-taboo-gold_50_mix": ("taboo-gold", "Qwen3-8B", 32, "2.00"),
    "adamkarvonen/Qwen3-8B-taboo-ship_50_mix": ("taboo-ship", "Qwen3-8B", 32, "2.00"),
    "adamkarvonen/Qwen3-8B-taboo-snow_50_mix": ("taboo-snow", "Qwen3-8B", 32, "2.00"),
    "adamkarvonen/Qwen3-8B-taboo-moon_50_mix": ("taboo-moon", "Qwen3-8B", 32, "2.00"),
    "adamkarvonen/Qwen3-8B-taboo-rock_50_mix": ("taboo-rock", "Qwen3-8B", 32, "2.00"),
    "ceselder/qwen3-8b-ao-v3-best-dpo-halluc": ("ao-v3-dpo-halluc", "Qwen3-8B", 128,
                                                "1.41 (rsLoRA)"),
    "Kurapika993/llama-3.1-8b-responsible-ai-safety-lora": ("responsible-ai-safety",
                                                           "Llama-3.1-8B", 16, "2.00"),
}


def nm(a: str) -> tuple[str, str, int, str]:
    return SHORT.get(a, ("latentqa" if "latentqa" in a else a.split("/")[-1],
                         "Qwen3-8B", 64, "2.00"))


def _slug(heading: str) -> str:
    """GitHub's heading-anchor rule, following github-slugger: lowercase, drop
    punctuation and symbols, then replace each remaining space with a hyphen.

    Spaces are deliberately not collapsed and the result is not trimmed. A heading
    opening with a symbol, or containing a spaced em-dash, therefore yields a leading or
    doubled hyphen -- which is what GitHub actually links to. Collapsing them looks
    tidier and points at nothing.

    If a fragment is ever wrong the link still opens the file at the top, which is where
    an unanchored link landed anyway, so the failure is graceful.
    """
    return re.sub(r"[^\w\s-]", "", heading.lower()).replace(" ", "-")


def exp_links() -> dict[str, str]:
    """Map EXP-NNN to an anchored link into the notebook.

    Written rather than hardcoded because a bare `EXPERIMENTS.md` link lands a reader at
    the top of a 2,200-line file. Deriving the anchors from the headings means renaming
    an entry cannot silently break them: a missing key raises at generation time.
    """
    out: dict[str, str] = {}
    for line in NOTEBOOK.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^##\s+\[\d{4}-\d{2}-\d{2}\]\s+(EXP-\d{3}):", line)
        if m:
            out[m.group(1)] = f"EXPERIMENTS.md#{_slug(line.lstrip('# '))}"
    return out


def exp_count() -> int:
    return len(exp_links())


def _braced(macro: str) -> str:
    r"""The balanced argument of `\macro{...}` in main.tex.

    Brace-matched rather than delimited by whatever macro happens to follow. The first
    version of this read `\title{...}` up to `\author` and `\author{...}` up to `\date`,
    which works exactly until something is written between them -- pinning the date put
    a comment and a \newcommand after \author, and the author check then reported the
    comment as part of the name. A parser whose correctness depends on neighbouring
    lines not changing is not a parser.
    """
    text = MAIN_TEX.read_text(encoding="utf-8")
    start = text.find("\\" + macro + "{")
    if start < 0:
        raise RuntimeError(f"no \\{macro}{{...}} found in {MAIN_TEX.name}")
    i = start + len(macro) + 2
    depth = 1
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth:
        raise RuntimeError(f"unbalanced braces in \\{macro}{{...}} in {MAIN_TEX.name}")
    return text[start + len(macro) + 2:i - 1]


def _plain(latex: str) -> str:
    """LaTeX markup out, words left."""
    body = re.sub(r"(?m)%.*$", "", latex)
    body = re.sub(r"\\[a-zA-Z]+(?:\{[^{}]*\})?", " ", body)
    return " ".join(body.replace("{", " ").replace("}", " ").split())


def paper_title() -> str:
    r"""The title as `paper/tex/main.tex` sets it, with LaTeX spacing stripped.

    Read rather than typed. The README's citation block carried its own title for the
    whole project, and it was not the paper's -- anyone citing from the README named a
    work whose PDF says something else. Deriving it means the two cannot disagree,
    which is a stronger guarantee than a gate that notices when they do (M.4).
    """
    title = _plain(_braced("title"))
    if not title:
        raise RuntimeError(f"\\title in {MAIN_TEX.name} stripped to nothing")
    return title


def paper_author() -> str:
    r"""The author as `paper/tex/main.tex` sets it, with the \thanks footnote stripped.

    Read for the same reason the title is: four artifacts name the author and the
    failure mode is that they stop agreeing. `\thanks` is removed before the general
    command strip because its argument is a sentence, not a name.
    """
    raw = _braced("author")
    # \thanks holds a sentence with an email and a URL; strip it before the general
    # command pass, which would otherwise leave its prose in the name.
    name = _plain(re.sub(r"\\thanks\{.*?\}\}", "", raw + "}", flags=re.S))
    if not name:
        raise RuntimeError(f"\\author in {MAIN_TEX.name} stripped to nothing")
    if name != f"{GIVEN} {FAMILY}":
        raise RuntimeError(
            f"{MAIN_TEX.name} renders the author as {name!r}, but this file splits it "
            f"as given={GIVEN!r} family={FAMILY!r}. One of the two is wrong, and "
            f"guessing which would put a fabricated name on a paper.")
    return name


def citation_cff() -> str:
    """CITATION.cff, from the same title and author the paper and README use."""
    title = paper_title()
    paper_author()  # raises if main.tex and the declared split disagree
    return (
        "cff-version: 1.2.0\n"
        "message: >-\n"
        "  Please cite the manuscript using the metadata below.\n"
        "type: software\n"
        "title: >-\n"
        f"  {title}\n"
        "authors:\n"
        f"  - family-names: {FAMILY}\n"
        f"    given-names: {GIVEN}\n"
        f"    email: {EMAIL}\n"
        "    affiliation: Independent researcher\n"
        f"repository-code: '{REPO_URL}'\n"
        "license: MIT\n"
        "year: 2026\n"
        "date-released: 2026-11-02\n"
    )


def notebook_anchor(needle: str) -> str:
    """Anchor of the first `##` heading in the notebook containing `needle`.

    Raises rather than guessing. A hand-written anchor is a hand-maintained number by
    another name: it looks right, it is never checked, and it silently rots.
    """
    for line in NOTEBOOK.read_text(encoding="utf-8").splitlines():
        if line.startswith("## ") and needle.lower() in line.lower():
            return f"EXPERIMENTS.md#{_slug(line.lstrip('# '))}"
    raise KeyError(f"no heading in {NOTEBOOK.name} contains {needle!r}")


def audit_claims() -> int:
    """Number of claims the audit checks. Imported rather than typed, because the moment
    this file states a count the audit also computes, the two can disagree -- and this
    one is printed in the instructions a reader follows."""
    from audit_draft_numbers import claims

    return len(claims())


def retracted_count() -> int:
    """Same reasoning as `audit_claims`: imported, never typed."""
    from retracted import RETRACTED

    return len(RETRACTED)


def collected_tests() -> int:
    import os
    import subprocess
    import sys

    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"],
                       cwd=REPO_ROOT, capture_output=True, text=True, env=env)
    m = re.search(r"(\d+) tests? collected", r.stdout)
    if m is None:
        raise RuntimeError(f"could not read the test count from pytest:\n{r.stdout[-800:]}")
    return int(m.group(1))


def pages(pdf: Path) -> int:
    """Page count read from the file. Never counts b'/Type /Page' -- that substring also
    matches /Type /Pages, the tree nodes, which is how a 77-page report was once reported
    as 89 (EXP-030)."""
    from pypdf import PdfReader

    return len(PdfReader(str(pdf)).pages)


def moved_sections() -> str:
    """The four reproduction subsections the round-8 cut moved here, re-levelled.

    Raises if the source is missing: dropping four sections silently is precisely the
    class of failure the cut was supposed to avoid creating.
    """
    if not MOVED_SRC.exists():
        raise FileNotFoundError(
            f"{MOVED_SRC} is missing; the README's reproduction detail comes from it")
    text = MOVED_SRC.read_text(encoding="utf-8")
    # `### D.1.1 Title` and `## D.9 Title` become plain `### Title` -- the numbering was
    # the paper's, and carrying it here would be a reference to a section that no longer
    # exists in the paper.
    text = re.sub(r"(?m)^#{2,3}\s+D\.\d+(?:\.\d+)?\s+", "### ", text)
    return text.strip()


def mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else float("nan")


def boot(xs: list[float]) -> tuple[float, float]:
    """Delegates to analysis/bootstrap.py. Exact by enumeration for the
    six-adapter population; see that module for why there is no seed."""
    return bootstrap.ci(xs)


def p0_by_adapter(l4_only: bool = True,
                  regime: str = "fixed_scale") -> dict[str, list[dict[str, Any]]]:
    acc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in P0.glob("public_adapter/*/*/records.jsonl"):
        if l4_only and p.parent.name.startswith("L36"):
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["scheme"] == "asymmetric" and r["regime"] == regime:
                acc[r["adapter"]].append(r)
    return acc


def snr_map() -> dict[str, float]:
    d: dict[str, list[float]] = defaultdict(list)
    for f in (P0 / "output_snr_orthonormal").glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                d[r["adapter"]].append(r["snr_out_orthonormal"])
    return {a: mean(v) for a, v in d.items()}


def word_to_adapter() -> dict[str, str]:
    """secret word -> adapter repo. `retention` is keyed by word, `snr_map` by adapter."""
    out: dict[str, str] = {}
    for p in sorted(P1.glob("*/records.jsonl")):
        for x in p.read_text(encoding="utf-8").splitlines():
            if x.strip():
                r = json.loads(x)
                out[r["secret_word"]] = r["adapter"]
    return out


def retention(precision: str, floor: bool = False) -> dict[str, float]:
    """Per-adapter elicitation retention. `floor` subtracts the base model's score on the
    same instrument at the same precision, which moves the per-adapter split even though
    it moves the mean by under 2 points (B.7)."""
    rows: list[dict[str, Any]] = []
    for p in sorted(P1.glob("*/records.jsonl")):
        rows += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()
                 if x.strip()]
    acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    word: dict[str, str] = {}
    for r in rows:
        word[r["adapter"]] = r["secret_word"]
        if r["precision"] == "bf16":
            if r["condition"] == "aligned_bf16":
                acc[(r["adapter"], "ref")].append(r["guesser_p_word_normalised"])
            elif r["condition"] == "base_bf16":
                acc[(r["adapter"], "bref")].append(r["guesser_p_word_normalised"])
        elif r["precision"] == precision:
            if r["condition"] == "aligned_quant":
                acc[(r["adapter"], "cur")].append(r["guesser_p_word_normalised"])
            elif r["condition"] == "base_quant":
                acc[(r["adapter"], "bcur")].append(r["guesser_p_word_normalised"])
    out: dict[str, float] = {}
    for a in word:
        num, den = mean(acc[(a, "cur")]), mean(acc[(a, "ref")])
        if floor:
            num -= mean(acc[(a, "bcur")])
            den -= mean(acc[(a, "bref")])
        if den:
            out[word[a]] = num / den
    return out


def leak_contrast(precision: str) -> tuple[float, tuple[float, float]]:
    """Paired BF16 - quantized adversarial leak rate, in points, with its interval.

    The constraint side of the battery. The README quoted only the capability side for
    four drafts, which made a two-sided instrument read as a one-sided result.
    """
    rows: list[dict[str, Any]] = []
    for p in sorted(P1.glob("*/records.jsonl")):
        rows += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()
                 if x.strip()]
    acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        if r["prompt_kind"] != "adversarial":
            continue
        if r["condition"] == "aligned_bf16" and r["precision"] == "bf16":
            acc[(r["adapter"], "ref")].append(float(r["said_word"]))
        elif r["condition"] == "aligned_quant" and r["precision"] == precision:
            acc[(r["adapter"], "cur")].append(float(r["said_word"]))
    diffs = [(mean(acc[(a, "ref")]) - mean(acc[(a, "cur")])) * 100
             for a in sorted({k[0] for k in acc})]
    return mean(diffs), boot(diffs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    p0 = p0_by_adapter()
    snr = snr_map()
    taboo_flip = [mean([r["code_flip_rate"] for r in v]) for a, v in p0.items()
                  if "taboo" in a]
    kept = list(retention("int4_g128").values())
    flips = {a: mean([r["code_flip_rate"] for r in v]) for a, v in p0.items()}
    coss = {a: mean([r["cosine"] for r in v]) for a, v in p0.items()}
    rels = {a: mean([r["relative_error"] for r in v]) for a, v in p0.items()}

    # The behavioural pipeline quantizes a merged model on its own recomputed grid, so
    # the weight number quoted beside a behavioural one has to come from that regime.
    p0_adp = p0_by_adapter(regime="adaptive_scale")
    taboo_flip_adp = [mean([r["code_flip_rate"] for r in v]) for a, v in p0_adp.items()
                      if "taboo" in a]
    taboo_valchg_adp = [mean([r["value_change_rate"] for r in v])
                        for a, v in p0_adp.items() if "taboo" in a]
    unchanged = 100 - mean(taboo_flip) * 100
    codes_changed_adp = mean(taboo_flip_adp) * 100
    values_changed_adp = mean(taboo_valchg_adp) * 100
    behav = mean(kept) * 100
    lo, hi = boot(kept)
    r3 = retention("int3_g128")
    r4pc = retention("int4_per_channel")

    text = f"""# Adapter retention under quantization

Code and data for *{paper_title()}*, by {paper_author()}. Manuscript dated November 2, 2026.

This study asks how much of a LoRA fine-tune survives when its weights are merged into a base model and quantized. The weight update changes substantially, but that alone does not tell us how the model's behaviour will change.

[Read the paper](paper/adapter-retention-arxiv.pdf) · [Technical report](paper/adapter-retention-technical-report.pdf) · [Reproduction instructions](REPRODUCING.md)

![Weight-space change and behavioural retention](paper/figures/fig01_erasure_vs_survival.png)

## Results

Across six taboo-word adapters, mean elicitation retention was **{behav:.1f}%** at INT4 g128, **{mean(list(r4pc.values())) * 100:.1f}%** at INT4 per-channel, and **{mean(list(r3.values())) * 100:.1f}%** at INT3 g128.

Enumerated 95% interval at INT4 g128: [{lo * 100:.1f}%, {hi * 100:.1f}%]. This supports no detectable loss on this battery; it does not establish identical behaviour. Individual retention spans **{min(r3.values()) * 100:.1f}% to {max(r3.values()) * 100:.1f}%** at INT3, despite similar weight-space measurements.

The weight-space study covers nine adapters on two base models. The behavioural study covers six rank-32 adapters sharing one base model and one training recipe. Quantization is simulated with quantize/dequantize operations, rather than measured through fused inference kernels. The paper discusses failed instruments, corrected measurements, withdrawn predictions, and the limits of these comparisons.

<details>
<summary>Weight-space measurements</summary>

On a fixed grid, where a weight changes only when its delta clears the step size, {unchanged:.1f}% of codes are unchanged. On the deployment path, merging moves the grid, {values_changed_adp:.1f}% of stored values change while only {codes_changed_adp:.1f}% of integer codes do. These counts describe different aspects of quantization; see the paper for their definitions.

| Adapter | Base | Rank | Scaling | Cosine | Code-flip rate | Relative error | Output SNR |
|---|---|---|---|---|---|---|---|
"""
    for adapter in sorted(p0, key=lambda k: coss[k]):
        name, base, rank, scaling = nm(adapter)
        text += (f"| {name} | {base} | {rank} | {scaling} | {coss[adapter]:.3f} | "
                 f"{flips[adapter] * 100:.2f}% | {rels[adapter]:.2f} | {snr[adapter]:.2f} |\n")
    text += f"""
Relative error uses complete erasure as a baseline of 1.0. The table uses the fixed-grid measurements; behavioural runs use the deployment grid.

</details>

## Setup and checks

Use Python 3.11 or newer. Install PyTorch separately for your hardware, then the pinned dependencies. The original experiments used PyTorch 2.11.0 with CUDA 12.8; GPU measurements require BF16 support and enough memory for the selected model.

```bash
git clone {CLONE_URL}
cd adapter-retention-paper
python -m venv .venv
source .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python analysis/audit_draft_numbers.py --strict
PYTHONPATH=src python analysis/retracted.py
PYTHONPATH=src python analysis/forward.py
PYTHONPATH=src python analysis/countcheck.py --strict
PYTHONPATH=src python analysis/xref.py --strict
```

The tests and analysis checks run on CPU using the committed records. Rerunning the model measurements requires a GPU and model downloads. See [Appendix D](paper/appendix-D-reproduction.md) for the original environment and measured runtimes.

### Every command, in order

The [reproduction guide](REPRODUCING.md) contains the complete measurement sequence, figure and PDF builds, GPU capability notes, and Windows long-path setup. Appendix D's references to README reproduction details are covered by that guide.

## Try the predictor

`ar.predict` estimates weight-space retention from an adapter without loading a full base model. It runs on CPU and downloads adapter tensors. It does not predict behavioural retention.

```bash
PYTHONPATH=src python -m ar.predict --adapter adamkarvonen/Qwen3-8B-taboo-smile_50_mix --bits 4 --group-size 128
```

## Repository contents

| Path | Contents |
|---|---|
| `paper/` | Manuscript sources, appendices, both PDFs, and both figure variants |
| `src/ar/` | Quantization, retention metrics, adapter handling, evaluation, and prediction |
| `scripts/` | Measurement and validation programs |
| `analysis/` | Tables, figures, numerical audits, and document checks |
| `results/raw/` | Experimental records and original run manifests |
| `tests/` | Unit tests and regression fixtures |

[EXPERIMENTS.md](EXPERIMENTS.md) records the runs and corrections. [METHODOLOGY.md](METHODOLOGY.md) explains the validation practices; [PROMPTS.md](PROMPTS.md) contains the experimental stimuli. The dated [research protocol](docs/PROJECT-EXECUTION-PLAN-v2.md) supports the registered predictions. [PROVENANCE.md](PROVENANCE.md) describes this release and the historical revision identifiers.

## Citation

```bibtex
@misc{{adapter_retention_2026,
  author = {{{FAMILY}, {GIVEN}}},
  title  = {{{paper_title()}}},
  year   = {{2026}},
  url    = {{{REPO_URL}}},
}}
```

Citation metadata is also available in [CITATION.cff](CITATION.cff). The repository is distributed under the [MIT license](LICENSE).
"""
    reproduction = "# Reproducing the experiments\n\n" + moved_sections() + "\n"
    if args.write:
        OUT.write_text(text, encoding="utf-8")
        CITATION.write_text(citation_cff(), encoding="utf-8")
        (REPO_ROOT / "REPRODUCING.md").write_text(reproduction, encoding="utf-8")
        print("wrote README.md, CITATION.cff and REPRODUCING.md")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
