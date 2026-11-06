# Appendix D: Reproduction from a clean machine

*Written to be run by someone who is not the author. Every command below is the literal
command; every runtime is measured on the hardware named in D.1, not estimated. Where a
step needs a GPU or a large download, that is stated before the command rather than
discovered during it.*

**If something here does not work, that is a bug in this appendix.** Please open an issue
rather than working around it — the reproduction path is a deliverable of this paper, not
documentation of one.

---

## D.1 What you need

| | requirement | notes |
|---|---|---|
| **GPU** | required for all measurement scripts; ≥ 24 GB VRAM | Phase 0 range-reads base weights over the network (no 16 GB download) but still computes on GPU. The capability floor is `sm_80` — BF16 is used throughout — and is lowered with `AR_MIN_CAPABILITY` (README, *GPU capability*). It was `sm_120` until 2026-08-06, which refused every pre-Blackwell card; see the README for why that was the wrong guard. |
| **Disk** | ~35 GB | Qwen3-8B (16 GB) + Llama-3.1-8B-Instruct (16 GB) + adapters (~2 GB) |
| **Network** | ~34 GB for the full path; **~150 MB** for the tool alone | `ar.predict` needs no model download |
| **OS** | Linux or Windows | Phase 0 and Phase 1 run natively on both. No WSL2 required for anything in this paper. |
| **Time** | ~76 min compute after downloads | breakdown in D.5 |

**No gated repositories are required.** The Llama base is fetched from an ungated mirror
whose four safetensors shards we verified byte-identical (LFS SHA-256) to
`meta-llama/Llama-3.1-8B-Instruct`. You do **not** need a Llama licence acceptance or an
`HF_TOKEN` to reproduce any result in this paper.

### Hardware these timings were measured on

NVIDIA RTX 5090 (32 GB, sm_120, driver via CUDA 12.8), Windows 11, Python 3.11.15.

## D.2 Environment

**Every command below is POSIX shell**, which is what a Linux reproducer or a Windows
user in Git Bash or WSL2 will paste. The timings in D.1 were measured on Windows 11 in
PowerShell, where the inline environment-variable prefix is not valid syntax. The
translation is one line, applied to every command in this appendix:

```powershell
$env:PYTHONPATH = "src"        # once per session, then drop the prefix
python -m pytest -q            # instead of: PYTHONPATH=src python -m pytest -q
```

`export VAR=value` becomes `$env:VAR = "value"`, and a trailing `\` line continuation
becomes a backtick. Nothing else differs.

```bash
# Windows only, and required before cloning (README, *On Windows, enable long paths*):
git config --global core.longpaths true

git clone https://github.com/MaximG6/adapter-retention-paper.git adapter-retention
cd adapter-retention

conda create -n retention python=3.11 -y
conda activate retention

# cu128 or newer is MANDATORY on Blackwell (sm_120). Older CUDA builds import
# cleanly and then produce garbage rather than failing loudly.
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Exact pinned versions used for every number in this paper:

```
python              3.11.15
torch               2.11.0+cu128     (torch.version.cuda == 12.8, cudnn 91900)
transformers        5.14.1
peft                0.20.0
trl                 1.9.2
datasets            5.0.1
accelerate          1.14.0
numpy               2.4.6
pydantic            2.13.4
safetensors         0.8.0
huggingface-hub     1.25.1
```

**Verify the GPU actually works before trusting any number.** On sm_120 with an
older CUDA build this check fails while everything else appears fine:

```bash
python -c "
import torch
d = torch.device('cuda:0')
x = torch.randn(4096, 4096, dtype=torch.bfloat16, device=d)
print('capability', torch.cuda.get_device_capability(d))
print('mean|x @ x|', (x @ x).abs().float().mean().item(), '  expect ~51.0')
"
```

Expected: `mean|x @ x|` ≈ **51.0** (analytic value `64·√(2/π) = 51.06`). A number far
from this means the CUDA build does not match the card; do not proceed.

## D.3 Tests first

`PYTHONPATH=src python -m pytest -q` — **no failures**, ~8 s, no GPU (the device tests
stub the CUDA API). Read it as "none fail" rather than as a count to match; the count
grows each round and the README's verification block carries the current one, generated.
These gate the metric definitions, the scoring logic and the device-resolution rules.

## D.4 Validation gates (run these before the experiments)

Three checks, each cheap, each having caught a real error here. All three need no GPU and
take about two minutes together.

| gate | command | expected |
|---|---|---|
| quantizer bit-exactness against `gptqmodel` | `scripts/validate_quantsim_vs_gptqmodel.py` | **36/36 bit-exact**, `max\|Δdequant\| = 0`, for `asymmetric` and `symmetric_gptq`. `symmetric_awq` deliberately differs — a different convention with no `gptqmodel` counterpart |
| adapter delta against peft's own merge | `scripts/validate_lora_delta_vs_peft.py` | matches `merge_and_unload() − original` to float precision, including rsLoRA. This is the check that caught an 11.3× scaling error |
| instrument gate self-test | `analysis/instrument_gate.py --self-test` | `gate self-test PASSED`. It asserts the gate **rejects** a probe already known to be broken |

*`gptqmodel` has no Windows wheel; the first script loads its pure-PyTorch quantizer from
the sdist without installing the package, so no build tools are required.*

## D.5 Running it, and what it costs

Every command, in order, with its runtime and its expected output, is in the repository
README under *Every command, in order*. It is a page of shell and the reproducer is
already there; repeating it here costs a page of the paper and helps nobody. The shape:

| stage | GPU | runs | wall time |
|---|---|---|---|
| tests and the three gates above | no | once | ~2 min |
| Phase 0, weight space | yes | 9 adapters + one 36-layer profile | ~17 min |
| Phase 0, synthetic sweep, amplification, spike | yes | once each | ~9 min |
| Phase 1, behaviour | yes | 6 adapters | ~42 min |
| refusal battery, both sets | yes | once | ~6 min |
| **total from cold** | | | **~76 min, ~34 GB download** |

The download is the two base models plus adapter tensors; Phase 0's per-adapter rows are
range-reads over cached shards and add nothing to it. An earlier draft totalled the
per-step column without the per-adapter multiplicities and reported ~50 min.

Every run writes `manifest.json` beside its records: torch and CUDA versions, GPU name and
compute capability, package versions, git SHA and all seeds. **Include it when reporting a
discrepancy** — it is usually enough to identify the cause.

## D.6 Determinism and what will differ

- **Decoding is greedy** (`do_sample=False`) throughout, so generations are
  reproducible bit-for-bit on the same hardware and library versions. Seeds are recorded
  but do not affect Phase 1 outputs; this is why seeds are not a replicate axis.
  Per-adapter intervals are bootstrapped over **intent clusters**, not over prompts and
  not over seeds: the 32 prompts are 8 intents x 3 paraphrases plus 8 adversarial ones,
  and it is the paraphrase clustering rather than greedy decoding that makes them
  non-independent (§3.11).
- **Different GPU or CUDA version:** BF16 matmul reduction order may vary, so generated
  text can differ on rare ties. Aggregate retention figures should reproduce to well
  within the reported intervals.
- **Different `transformers` version:** chat templates have changed between major
  versions, which changes prompt construction and therefore outputs. Pin 5.14.1 to
  reproduce exactly.
- **Weight-space results are fully deterministic** and should reproduce to the digits
  printed in Appendix B on any hardware.

## D.7 What was verified from a fresh clone, and what was not

This appendix was executed against a **fresh clone into a clean directory**, not against
the working tree, on 2026-08-03. Results:

| step | outcome |
|---|---|
| `git clone` on Windows without `core.longpaths` | **FAILED** — incomplete checkout. Fixed and documented in the README, *On Windows, enable long paths* |
| `git clone` with long paths enabled | clean |
| `pytest -q` | **169 passed** |
| `instrument_gate.py --self-test` | passed |
| `audit_draft_numbers.py --strict` | **208/208**, exit 0 |
| `appendix_tables.py --write` | regenerated **byte-identically** to the committed file |
| `appendix_prompts.py --write` | regenerated byte-identically |
| `gen_readme.py --write` | regenerated byte-identically |
| `python -m ar.predict` | ran, correct values, no GPU |

The byte-identical regeneration is the property worth noting: the committed tables,
prompt sets and README are reproducible from the committed records by anyone, with no
hidden state in the authoring environment.

**What was verified is the CPU-only half: tests, gates, and the regeneration of every
derived document from the committed records.** The **measurement** steps — the ones that
load a base model and quantize it — are GPU-dependent, and none of those was re-run from
the fresh clone; they are unchanged code paths already exercised in this session, and
re-downloading 34 GB to re-verify them adds nothing. So this table establishes that the
analysis pipeline reproduces from the committed raw records, not that the raw records
reproduce from the models. Their timings in D.5 are measured, not estimated.

**Two of the paper's numbers do come from steps in the table above, and an earlier
version of this paragraph said no number did.** `python -m ar.predict` ran with no GPU
and prints the ~30 values Appendix A.2 quotes; `instrument_gate.py --self-test` and
`validate_quantsim_vs_gptqmodel.py` produce the *36 of 36 bit-exact* figure §1 and §3.2
quote, and the first needs no GPU. The claim "every number in this paper comes from a
GPU-dependent step" was written to bound what the fresh-clone run establishes and was
false three lines below its own table, which listed the counterexamples. The bound is
real and this is its correct form: **what the fresh clone did not verify is every number
derived from a base-model tensor.**
