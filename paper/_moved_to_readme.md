### D.1.1 GPU capability, and the default that used to be wrong

Every measurement script resolves its device through `ar.device.require_cuda`, which
**defaults to a capability floor of sm_80** — Ampere or newer — and raises rather than
falling back. BF16 is used throughout, so sm_80 is the real requirement. An A100, H100,
4090 or 5090 all clear it with no configuration.

**The default was sm_120 until 2026-08-06, which meant this documented reproduction path
refused to run on every card older than Blackwell** — including the A100 and H100 a
reproducer is most likely to have. That floor was standing in for a real hazard, and
standing in for it backwards: an RTX 5090 under a pre-cu128 torch build imports cleanly
and then produces garbage. The population at risk is Blackwell cards on old toolkits, not
old cards. `require_cuda` now checks that directly and raises with the cu128 instruction,
and the capability floor asks only what the arithmetic needs.

To go lower — Turing, Volta — set it explicitly:

```bash
export AR_MIN_CAPABILITY=7.5
```

The run still raises if no visible device clears the floor you set, so this is an opt-in,
never a silent relaxation. Among qualifying devices the **largest-memory** one is chosen
(ties break on lower index), so 8B BF16 loads land on the biggest card without anything
naming a device index. The resolved device name and capability are written into every
`manifest.json`.

### D.1.2 On Windows, enable long paths before cloning — the clone fails otherwise

Some record paths reach **157 characters**, over the 260-character Windows `MAX_PATH`
budget once a clone directory is prepended. Without long-path support, `git clone`
reports *"Filename too long"* and leaves an **incomplete checkout** — the clone appears
to succeed and the failure is in the second line of output:

```
fatal: cannot create directory at
  'results/raw/phase0/public_adapter/...': Filename too long
warning: Clone succeeded, but checkout failed.
```

Enable it once, globally:

```bash
git config --global core.longpaths true
```

**This is verified, not anticipated** — cloning this repository on Windows 11 without it
fails at exactly the path above. With it, the clone completes and everything in this
appendix runs.

One residual symptom if you clone with `-c core.longpaths=true` rather than setting it
globally: the setting applies to the clone but not to the resulting repository, so
`git status` afterwards reports spurious `M`/`D` entries for the deepest record files.
They are artifacts of `git` being unable to stat those paths, not real changes —
`git -c core.longpaths=true status` shows a clean tree. Setting the option globally
avoids this entirely.

Linux and macOS are unaffected.

### D.6.1 There are two figure directories, and both are current

`paper/figures/` and `paper/figures-paper/` hold the same twelve figures under the same
names. Neither is stale.

| directory | built by | consumed by | difference |
|---|---|---|---|
| `paper/figures/` | the three figure scripts, run directly | the technical-report PDF, and the README | each figure carries its own title and caption text |
| `paper/figures-paper/` | `analysis/build_arxiv_pdf.py` | the LaTeX build, `paper/tex/` | in-figure headers suppressed, because LaTeX sets the captions |

```bash
# paper/figures/  -- default mode
PYTHONPATH=src python analysis/fig01_erasure_vs_survival.py
PYTHONPATH=src python analysis/fig05_06_08.py
PYTHONPATH=src python analysis/fig_secondary.py

# paper/figures-paper/  -- regenerated as step 1 of the arXiv build
PYTHONPATH=src python analysis/build_arxiv_pdf.py --tectonic <path>
```

**Do not run the figure scripts directly with `AR_FIG_PAPER=1`.** The variable controls
the *style* only; the output directory is `FIGDIR`, which those scripts always set to
`paper/figures/`. `build_arxiv_pdf.py` is what redirects them, by rebinding `FIGDIR`
before calling each `main()`. Setting the variable by hand therefore writes header-less
figures over the default set, and the damage is silent — the cross-checks still pass,
because the data behind the figures is unchanged.

Each script cross-checks its own figures against an independent recomputation from the
raw records, in whichever mode it runs. A figure that disagrees with the data fails the
build rather than being written.

**Reproducibility of the figure files themselves.** The PNGs are byte-identical on
regeneration. The vector PDFs are not: matplotlib embeds a `/CreationDate`, so two builds
of the same figure differ in exactly those five bytes and are identical once it is
stripped. Compare figure PDFs with the timestamp removed, or compare the PNGs.

## D.9 Known rough edges

Recorded because a reproducer will hit them, and finding them undocumented wastes an
afternoon.

- `snapshot_download` has been observed to exit 0 while leaving 0-byte files. Our
  download path fetches file-by-file and asserts sizes against the Hub's metadata.
  If a model load fails oddly, check for 0-byte files in `~/.cache/huggingface/hub`.
- On Windows, `Set-Content -Encoding utf8` writes a BOM that breaks Python source files.
  Irrelevant to running the code; relevant if you edit it from PowerShell.
- `analysis/appendix_tables.py` prints Unicode (α, Δ) that the Windows console codepage
  cannot encode. Use `--write` rather than piping stdout.
- The `latentqa` adapter's repo name is long and unlabelled in raw records; it is mapped
  to a short name in `analysis/appendix_tables.py`.

## D.9 Every command, in order

### Phase 0 — weight space (GPU required; no base-model download)

```bash
# One adapter, 4 layers, 3 schemes. ~70 s, ~1.5 GB network (range-reads tensors;
# does NOT download the 16 GB base model).
PYTHONPATH=src python scripts/measure_public_adapter.py \
  --adapter adamkarvonen/Qwen3-8B-taboo-smile_50_mix

# All nine adapters: repeat --adapter for each.
# The list is in paper/appendix-B-tables.md.
# Full 36-layer depth profile for one adapter, asymmetric only. ~6 min.
PYTHONPATH=src python scripts/measure_public_adapter.py \
  --adapter adamkarvonen/Qwen3-8B-taboo-smile_50_mix \
  --layers all --schemes asymmetric --out-subdir L36_asymmetric

# Synthetic rank sweep and dose-response. ~3 min.
PYTHONPATH=src python scripts/synthetic_sweep.py

# Subspace amplification, orthonormal probe. ~4 min.
PYTHONPATH=src python scripts/amplification_svd_test.py
PYTHONPATH=src python scripts/output_snr_orthonormal.py
```

### Phase 0 — the layer 1–3 spike (GPU, ~2 min)

Downloads Qwen3-8B (16 GB) on first run.

```bash
PYTHONPATH=src python scripts/outlier_channel_test.py
```

Expected: `1.gate_proj` step median/p1 ≈ **83.5**, activation at narrowest 1% of groups
≈ **0.17**, and near-zero correlations in the layer-0 and layer-18 controls.

### Phase 1 — behaviour (GPU, ~7 min per adapter)

```bash
# One adapter, all four precisions -> 256 records.
PYTHONPATH=src python scripts/run_phase1.py \
  --adapter adamkarvonen/Qwen3-8B-taboo-smile_50_mix \
  --precisions bf16,int4_g128,int4_per_channel,int3_g128

# Repeat for: taboo-gold, taboo-ship, taboo-snow, taboo-moon, taboo-rock
# (same repo prefix, substitute the word) -> 1536 records total.
```

### The refusal battery (GPU, ~6 min; downloads Llama-3.1-8B-Instruct, 16 GB)

```bash
PYTHONPATH=src python scripts/validate_refusal.py \
  --adapter Kurapika993/llama-3.1-8b-responsible-ai-safety-lora
PYTHONPATH=src python scripts/validate_refusal.py --battery xstest \
  --adapter Kurapika993/llama-3.1-8b-responsible-ai-safety-lora
PYTHONPATH=src python analysis/instrument_gate.py --refusal
```

Expected: **`REFUSAL INSTRUMENT NOT VALIDATED`** (exit code 2). That is the paper's
result, not a failure of the run — see §6.

### Analysis and tables

```bash
PYTHONPATH=src python analysis/phase1_pooled.py     # §5.1–5.3
PYTHONPATH=src python analysis/crossover.py         # PG-1
PYTHONPATH=src python analysis/word_vs_noise.py     # PG-2
PYTHONPATH=src python analysis/appendix_tables.py --write   # regenerates Appendix B
```

`appendix_tables.py` regenerates every table in Appendix B from the raw records. If your
run produces different numbers, the diff against the committed
`paper/appendix-B-tables.md` localises the discrepancy immediately.

### The tool alone (no GPU, ~150 MB network, ~30 s)

```bash
PYTHONPATH=src python -m ar.predict \
  --adapter adamkarvonen/Qwen3-8B-taboo-smile_50_mix --bits 4 --group-size 128
```

### Per-step runtimes and outputs

| step | GPU | wall time | network | output |
|---|---|---|---|---|
| tests | no | 7 s | 0 | 169 passed |
| quantsim vs gptqmodel | no | ~1 min | ~30 MB | 36/36 bit-exact |
| delta vs peft | no | ~1 min | ~200 MB | match to float precision |
| Phase 0, one adapter, 4 layers | yes | ~70 s | ~1.5 GB | 168 records |
| Phase 0, 36-layer profile | yes | ~6 min | ~1.5 GB | 504 records |
| synthetic sweep | yes | ~3 min | 0 | 43 records |
| amplification + output SNR | yes | ~4 min | ~1.5 GB | 24 + 126 records |
| outlier channel test | yes | ~2 min* | 16 GB* | 10 records |
| Phase 1, per adapter | yes | ~7 min | 16 GB* | 256 records |
| refusal battery (both) | yes | ~6 min | 16 GB* | 96 records |

\* first run only; the base model is cached afterwards.

The per-adapter rows run more than once: Phase 0 for **9** adapters, Phase 1 for **6**.
Summing the column with those multiplicities:

| | wall time |
|---|---|
| single-run rows (tests, validation, 36-layer, sweep, amplification, outlier, refusal) | ~23 min |
| Phase 0, one adapter x 9 | ~11 min |
| Phase 1, per adapter x 6 | ~42 min |
| **total** | **~76 min** |

**Total from cold: ~34 GB download, ~76 min compute** for every number in the paper. The
download figure is the two base models plus adapter tensors; the 1.5 GB rows are
range-reads that reuse the same cached shards, so they do not add to it. An earlier draft
totalled the compute column without the multiplicities and reported ~50 min.

Every run writes `manifest.json` beside its records containing torch/CUDA versions, GPU
name and compute capability, package versions, git SHA and all seeds. **Include this file
when reporting a discrepancy** — it is usually enough to identify the cause.
