# Research protocol and dated amendments

# PART 0: STUDY DESIGN

## 0.1 The restructure, and why

v1 bet everything on reproducing a lightly-maintained third-party repo (ATP: 5 commits, 10 stars, 0 forks) and then detecting a behavioral effect that might not exist. Two independent ways to end three weeks with nothing.

While reviewing v1 we found a flaw that could have invalidated the whole experiment: a rank-16 LoRA produces a small weight delta, and merging it into BF16 then quantizing to INT4 may push that delta below the quantization step size. If so, the "aligned quantized model" is numerically just the base model, and any drift you measure is adapter erasure, not alignment decay.

That flaw is a better project than the one it threatens. It is a **measurement**, not a hypothesis test. The delta either does or does not survive quantization; you compute it and you have a number. So it leads.

## 0.2 Three phases, decreasing certainty

| Phase | Question | Depends on | Can it fail to produce a result? |
|---|---|---|---|
| **0** | Does a LoRA weight update survive quantization numerically? | Nothing but a downloaded checkpoint | No |
| **1** | Does the aligned *behavior* survive? | A trained adapter, round-0 evals | Barely |
| **2** | Does quantization change the *rate* of alignment tipping? | ATP repo reproducing | Yes |

Phase 0 is a day of tensor arithmetic. Phase 2 is the original project and runs only if Phase 1 shows alignment survives, because if it does not survive there is nothing left to tip.

**Every branch produces a paper.** See Part 6.

## 0.3 Prior-art warning, read before day 1

The *numerical* phenomenon may already be known in the efficiency literature. LoftQ and QA-LoRA are both motivated by the LoRA-quantization interaction, and "merge then quantize degrades quality" is folk knowledge among practitioners. **Do not assume novelty.**

What is plausibly unclaimed:
- Retention quantified as a function of adapter rank × precision × group size × module type
- The link from numerical retention to **behavioral alignment** outcomes
- The safety framing: a safety or alignment fine-tune may not survive the compression applied at deployment

**Day 1, first task, before writing code:** run prior-art searches on LoRA merge quantization retention, adapter erasure under PTQ, safety fine-tune survival under compression, LoftQ, QA-LoRA, and quantization of merged adapters. Record every hit in `PRIOR_ART.md` with an honest verdict. If the numerical result is fully covered, pivot immediately to the behavioral framing, having lost one day.

## 0.4 Standing rules

- **Vendor, never edit.** Third-party repos are git submodules. Modifications are tracked `.patch` files.
- **Log raw, aggregate later.** Everything at the finest granularity available, as JSONL. All analysis re-derives from raw.
- **No silent fallbacks.** A quantization backend that quietly falls back to BF16 would invisibly destroy the experiment. Crash loudly.
- **Manifest every run.** Torch version, CUDA, GPU, driver, package versions, git SHAs, seeds.
- **Type hints throughout. Pydantic for configs. Vectorized ops over loops. No bare `except`.**

---

# PART 1: PHASE 0 — ADAPTER RETENTION (Days 1 to 3)

The core of the project. Pure tensor arithmetic on downloaded checkpoints. No training, no third-party research code, no reproduction risk.

## 1.1 The quantity being measured

For a linear layer with base weight `W ∈ R^{d_out × d_in}` and LoRA update `Δ = (α/r) · B·A`:

```
W_merged = W + Δ
```

Group-wise affine quantization with group size g maps each group to a grid with step size `s`. Define:

```
Q(·)        = quantize-then-dequantize
Δ_effective = Q(W + Δ) − Q(W)
```

`Δ_effective` is what the deployed model actually receives. The central metric:

```
retention = ||Δ_effective||_F / ||Δ||_F
```

Retention near 1 means the adapter survives. Retention near 0 means quantization ate it.

## 1.2 Metrics to compute

Per (layer, module type, rank, precision, group size):

1. **Retention ratio** `||Δ_eff||_F / ||Δ||_F`. Headline metric.
2. **Directional fidelity** `cos(vec(Δ), vec(Δ_eff))`. Magnitude can survive while direction scrambles. Report both.
3. **Bit-flip rate**: fraction of weights where `quant(W+Δ) ≠ quant(W)`. The most interpretable single number: "at INT4 g128, only X% of adapter updates change the quantized weight at all."
4. **Step-ratio distribution** `|Δ| / (s/2)` per weight. Values below 1 are below the quantization noise floor. Report the full distribution and the fraction under 1.
5. **Layer-depth profile** of retention. Early versus late layers.
6. **Module-type profile**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.

## 1.3 The grid

| Factor | Levels |
|---|---|
| Base model | Qwen3-8B, Llama-3.1-8B-Instruct, Qwen3-4B |
| Adapter rank | 8, 16, 32, 64 |
| Precision | FP8, INT4 g128, INT4 g64, INT4 g32, GGUF Q4_K_M, GGUF Q3_K_M |
| LoRA alpha | held at 2× rank unless testing scaling |

**Rank and alpha matter enormously.** Retention should rise with both, since larger deltas clear the step size. A rank-versus-retention curve is a real result and it is what makes this a paper rather than a single number.

## 1.4 Adapter sources, in order

**Day 1: public adapters, zero training.** Pull existing LoRA adapters from HuggingFace for the base models. This makes day 1 genuinely dependency-free. Prefer adapters describing themselves as safety, DPO, instruction, or alignment tuned. Record every adapter's ID and reported training config.

**Day 2: synthetic adapters, controlled.** Generate LoRA deltas at controlled magnitudes (scale a random low-rank update to a target Frobenius norm) to isolate the magnitude-versus-retention relationship without confounds from how any particular adapter was trained. This gives you a clean dose-response curve.

**Day 3: own-trained adapters.** Train small DPO LoRAs with plain `peft` + `trl` (**not** LLaMA-Factory; that dependency is deferred to Phase 2 and may never be needed). A few hundred preference pairs, one epoch. Measure retention on these.

## 1.5 Implementation notes

- Implement group-wise affine quantize/dequantize directly rather than relying on a library, so you control group size and can compute step sizes exactly. Validate your implementation against `gptqmodel` or `autoawq` output on one layer to confirm correctness.
- GGUF K-quants use a different scheme (block-wise with a super-block scale). Handle them separately, and use `llama.cpp`'s own quantization to produce the tensors, then read them back. Do not hand-roll K-quants.
- FP8: use the E4M3 format the 5090's tensor cores implement. Step size is relative, not absolute, so retention behaves differently. Report it separately rather than pooling with the integer formats.

## 1.6 Compute and time

Loading an 8B model in BF16 is 16GB, fits on the 5090. The arithmetic is trivial. **Total Phase 0 compute: under 5 GPU-hours across the entire grid.**

## GATE 0 (end of day 3)

- **Low retention** (bit-flip rate below ~50% at INT4 g128 for rank ≤ 16): strong finding. Proceed to Phase 1 to confirm behaviorally. This is the good outcome.
- **High retention** (above ~90% across the grid): also a finding, and it *clears the confound* for the original tipping study. Proceed to Phase 1, then Phase 2, with the confound ruled out and documented.
- **Fully scooped by prior art:** pivot to the behavioral framing only, and compress Phase 0 into a related-work reproduction.

Either retention outcome moves you forward. That is the point of leading with a measurement.

---

# PART 2: PHASE 1 — BEHAVIORAL CONFIRMATION (Days 4 to 7)

Numerical retention is necessary but not sufficient. A 30% retention ratio might preserve all behavior, or none. Phase 1 establishes the mapping.

## 2.1 Design

For each (base model, adapter, precision), evaluate four models:

| Condition | Meaning |
|---|---|
| `base_bf16` | Reference floor |
| `aligned_bf16` | Reference ceiling |
| `base_quant` | Quantized base |
| `aligned_quant` | Merged then quantized |

**The diagnostic:** if `aligned_quant` converges to `base_quant` while `aligned_bf16` stays distinct from `base_bf16`, alignment was erased by quantization. Quantify with a behavioral distance metric, not eyeballing.

**Also run the alternative ordering:** adapter served on top of a quantized base (QLoRA-style inference, no merge). This preserves the adapter exactly and is the practical recommendation if retention is poor. Include it as `base_quant + adapter`.

## 2.2 Evaluation batteries

Three, all round-0 static, all cheap:

1. **Alignment behavior**: refusal rate on a held-out safety set, plus preference-agreement rate on held-out pairs from the DPO training distribution. This directly measures whether the tuned behavior persists.
2. **Capability**: MMLU-Pro subset, GSM8K, IFEval. This is the covariate for everything downstream and it is not optional.
3. **Distributional**: KL divergence between `aligned_bf16` and each other condition's next-token distributions on a fixed prompt set. The most sensitive instrument, catches changes the task metrics miss.

## 2.3 Logging schema (addresses the v1 parsing weakness)

Never collapse behavior into one boolean. Log separately:

```python
class Record(BaseModel):
    seed: int
    model_id: str
    precision: str
    condition: str
    question_id: str
    round: int | None          # None for Phase 1
    response_text: str
    tool_attempted: bool       # did it try?
    tool_call_wellformed: bool # did it parse?
    tool_used: bool            # did it succeed?
    correct: bool
    completion_tokens: int
    wall_time_s: float
```

A quantized model emitting malformed tool calls would look identical to behavioral drift if you only logged `tool_used`. Three fields make that distinguishable.

## 2.4 Manual validation (addresses the v1 "never verified the metric" weakness)

**Before trusting any aggregate**, read 20 full trajectories by hand, spread across precisions. Confirm that a negative `tool_used` corresponds to the agent choosing a shortcut, not to a refusal, a parse failure, a timeout, or a degenerate repetition loop. Record the audit in `VALIDATION.md`. If more than 2 of 20 are artifacts, fix the harness before proceeding.

## GATE 1 (end of day 7)

- **Alignment behavior survives quantization** despite partial numerical retention: interesting in itself (redundancy in the adapter), and Phase 2 is well-founded. Proceed.
- **Alignment behavior collapses**: you have the headline paper already. Phase 2 becomes optional supporting evidence. Consider stopping here and writing.
- **Mixed by precision**: the best outcome. A retention threshold below which behavior breaks is a concrete, citable deployment guideline.

---

# PART 3: PHASE 2 — CONDITIONAL TIPPING STUDY (Days 8 to 15)

**Run only if Phase 1 shows alignment survives at some precision.** If it does not, there is nothing to tip and this phase is skipped.

## 3.1 Reproduction gate first

Days 8 to 9: clone `aiming-lab/ATP` (pin the SHA), read its two open issues before writing anything, set up the env, and run `test_sa.py` on the base model at BF16, 3 seeds.

**GATE 2 (end of day 9).** Pass if tool-usage declines across rounds consistently with the paper's direction and the decline is visible above seed variance. **Fail: stop Phase 2 entirely and write the Phase 0+1 paper.** Do not spend more than one extra day fighting the repo.

## 3.2 Design (days 10 to 15)

Conditions: only the precisions where Phase 1 showed alignment survives. There is no point running the tipping loop on a model whose alignment was already erased.

Grid: base models {Qwen3-8B, Qwen3-4B} × surviving precisions × 5 seeds × rounds 0 to 5.

**Add the matched-memory baseline.** BF16 4B versus INT4 8B occupy roughly the same VRAM. That is the actual choice a practitioner faces, and including it converts an academic finding into deployment guidance. It costs one condition you are already producing.

**Scale grid, not a fallback.** 4B and 8B both in the main grid gives a within-family scale trend. Compression papers are expected to show scale behavior and 8B-only draws that criticism.

---

# PART 4: STATISTICS (corrected from v1)

## 4.1 Phase 0

Descriptive plus regression. Retention as a function of `log(rank)`, `bits`, `log(group_size)`, and module type, with layer as a random effect. Report the rank-versus-retention curve with bootstrap CIs over layers. No hypothesis test is needed for a measurement; report intervals.

## 4.2 Phase 1

Paired comparisons of each condition against `aligned_bf16`. Bootstrap CIs over questions. Holm correction across precisions. For the KL metric, bootstrap over the prompt set.

## 4.3 Phase 2 (v1's model was misspecified)

v1 used `(1 | seed) + (1 | question_id)`, which treats rounds as exchangeable within a trajectory. They are not: round r depends on r−1 through accumulated experience, which is the entire mechanism. That anti-conservative specification would over-detect.

**Corrected primary model:**

```
tool_used ~ round * precision + round0_accuracy
            + (round | seed) + (1 | question_id)
```

Random **slope** for round within seed. If convergence fails, use GEE with an **AR(1)** working correlation on round within trajectory and report that choice explicitly.

**Primary test:** LRT on the `round:precision` interaction.
**Mandatory companion:** fit the same model *without* `round0_accuracy` and report both side by side. That comparison is the capability-confound story and belongs in the paper regardless of outcome.

## 4.4 Effective sample size (v1 overstated power)

v1 claimed ~4,700 observations per condition. With greedy decoding at temperature 0 the response to a given question in a given context is deterministic, so seeds only vary presentation order. **Effective n is much closer to the 157 questions than to 4,700.**

Required corrections:
- Bootstrap all CIs **over questions**, not over observations.
- Run a temperature-0.7 subset (one model, all precisions, 5 seeds) to quantify how much genuine stochasticity exists. Report it as a deviation from ATP's protocol.
- Never report nominal n as if it were effective n. State effective sample size explicitly in Limitations.

---


# AMENDMENTS

Amendments are appended, never merged into the text above. The original plan stays readable as written so that the record shows what changed and when. Each amendment is dated and states its trigger.

---

## Amendment 1 — 2026-07-29 (Phase 0, day 1)

**Trigger:** prior-art search, `PRIOR_ART.md`, logged as EXP-002. Approved by Max the same day.

**Outcome of the day-1 scoop check:** not scooped. Every metric in the Phase 0 list is unclaimed. But the *mechanism* is already asserted in the literature, so the framing narrows.

### 1.1 Framing: cite the mechanism, claim the measurement

The erasure mechanism — a low-rank delta merged into full-precision weights can fall below the quantization step size and be destroyed — **is cited to arXiv 2602.13151** (*Quantization-Robust LLM Unlearning via Low-Rank Adaptation*, 2026-02-13), which asserts it qualitatively to motivate a method. It is additionally implied by QA-LoRA's design (2309.14717) and is folk knowledge among practitioners.

**We do not claim the mechanism as ours.** Claiming it in late 2026 would read as underread and would be caught.

**Our contribution is the quantitative characterization:** retention ratio `‖Δ_eff‖_F/‖Δ‖_F`, bit-flip rate, step-ratio distribution `|Δ|/(s/2)`, and directional fidelity `cos(vec Δ, vec Δ_eff)`, swept over **rank × precision × group size × module type × layer depth**. None of these numbers exists in the literature.

Supersedes the framing implicit in §0.1 and §1.1, which read as though the mechanism were our observation.

### 1.2 The rank sweep is the primary result

**Supersedes §1.3.** Rank was a row in the grid table; it is now the headline.

Rationale: a single retention number at rank 16 restates known folklore. The *curve* is the contribution.

**Revised rank levels: 4, 8, 16, 32, 64, 128.** Extended at both ends from the original {8, 16, 32, 64} — the low end is where erasure should be total and the high end is where it should saturate, and a dose-response curve needs both asymptotes to be credible.

**Revised cut priority: if time forces a cut, cut precisions before ranks.** This inverts the original implicit priority. Reduce the precision axis to {INT4 g128, INT4 g32, FP8} before dropping any rank level.

### 1.3 New experiment: the Δ-alone vs. W+Δ reconciliation

**New, no counterpart in the original plan.** Gets its own paper subsection.

arXiv 2411.19530 (*Quantized Delta Weight Is Safety Keeper*, 2024-11-29) quantizes **Δ alone**, BitDelta-style, and finds compression *protects* alignment (alignment-breaking risk down up to 66.17%). We quantize **W + Δ** jointly and expect it destroys the adapter. Opposite conclusions, and both correct.

**The reconciliation is the step size.** Quantizing Δ alone sets the scale from Δ's own dynamic range, so Δ is preserved by construction. Merge-then-quantize sets the scale from W's range, one to two orders of magnitude larger, so Δ competes against a step size it had no part in setting.

**Design:** for the same adapter at the same bit-width, measure retention under both schemes. Report the ratio of step sizes `s_{W+Δ}/s_Δ` alongside the retention gap, and show the step-size ratio accounts for the divergence.

Cost is near zero — both code paths already exist in `quantsim.py`, since quantizing Δ alone is the same function called on a different tensor. High value: it converts an apparent contradiction with published work into a mechanism result, and preempts the most obvious reviewer objection.

### 1.4 Phase 1 must not lead with perplexity

**Supersedes §2.2** to the extent that §2.2 left perplexity's role open.

arXiv 2605.15208 (*Quantization Undoes Alignment*, 2026-05-02) establishes independently that perplexity is a false negative for behavioral degradation: under 0.5% change at 8-bit while measurable bias emerges, and under 11% even at 3-bit.

**Behavioral and distributional metrics carry Phase 1.** Perplexity is reported **only as a negative control**, to demonstrate that the standard screening metric misses the effect. Reporting it as a headline number would replicate a known error.


### 1.6 Residual scoop risk after the day-1 check

Both papers flagged as highest risk were read and resolved the same day:

- **2606.01412** (*GPTQ-intrinsic LoRA*) — read in full. Its low-rank term is a **designed compensator** (`L = V_r` from the calibration matrix, `R` initialized at zero), not a pre-existing adapter. No retention bound, no per-weight step-size condition, per-channel only. Its `(1 − r/N)` bound describes *compensation capacity*, which looks superficially like retention-vs-rank and is not. **We are not measuring something it derives.** Cite as nearest theoretical neighbour and state the distinction explicitly.
- **2411.19530** — resolved into the new experiment in §1.3 above.

Forward-citation traversal (QA-LoRA complete, LoftQ partial to 100) and a proceedings pass produced nothing measuring our quantities. Residual risk is low and diffuse. Two deferred follow-ups are recorded in `PRIOR_ART.md` §10.

**No change** to the phase structure, the gate criteria, the Phase 0 metric list, or the statistics in Part 4.

---

## Amendment 2 — 2026-07-29 (Phase 0, day 1, later same day)

**Trigger:** Max identified the adaptive-grid confound before any measurement was taken; two further metric corrections fell out of implementing it. Logged as EXP-004. GGUF gap logged as EXP-005.

### 2.1 Retention is measured under two scale regimes, always both

**Supersedes §1.1 and §1.2.** The original definition `Δ_eff = Q(W + Δ) − Q(W)` is ambiguous, because group-wise affine quantization derives `s` and `z` from each group's own min and max. If Δ moves a group extreme, the grid shifts and weights with `Δ_i` exactly zero still change. Counting those conflates the mechanism with an artifact and **inflates apparent transmission**.

- **`fixed_scale`** — `s, z` from W alone, applied unchanged to both W and W+Δ. Isolates the step-size mechanism.
- **`adaptive_scale`** — each tensor on its own grid. Deployment reality.

Required argument, no default. Both reported everywhere, plus `grid_shift_fraction` (weights changing under adaptive but not fixed), `grid_shift_fraction_zero_delta`, `scale_shift_fraction`, and `retention_gap`.

**Confirmed real, not hypothetical:** in a constructed case where 127 of every 128 weights have exactly zero delta, `adaptive_scale` changes some of those untouched weights and `fixed_scale` does not.

**Limitation to state in the paper:** `fixed_scale` clips when Δ is large relative to W's range (retention 0.463 vs adaptive 1.005 at mean `|Δ|/s = 11.5`). Valid instrument in the small-delta regime only — which is our regime, but cross-regime comparison at large delta is confounded by clipping.

### 2.2 The step-ratio threshold is probabilistic, not deterministic

**Supersedes §1.2 item 4.** "Values below 1 are below the quantization noise floor" reads as a hard threshold. It is not: `|Δ| < s/2` guarantees the code moves by at most one step, not that it does not move — whether it moves depends on the weight's position inside its bin.

Measured over 2M samples: **`P(flip) = min(|Δ|/s, 1)` to four decimals.** At `step_ratio = 1`, half those weights flip.

So the fraction of step-ratios under 1 is **not** the fraction erased. Report the step-ratio distribution with the `P(flip) = |Δ|/s` curve overlaid, and use the closed form as an internal consistency check against the measured flip rate.

### 2.3 Cosine replaces retention_ratio as the primary reported quantity

**Supersedes §1.2 items 1 and 2 and §4.1.** `retention_ratio = ‖Δ_eff‖/‖Δ‖` is **unbounded above and non-monotone**. At mean `|Δ|/s = 0.0002` it measures **95.5** with `cosine = 0.015`: each flipped weight contributes a full step `s` to `Δ_eff` regardless of `Δ_i`, so the norm inflates while the direction randomizes.

Reading `retention_ratio = 95` as excellent retention is exactly backwards — the adapter was erased and replaced by larger uncorrelated noise. The plan's "near 1 survives, near 0 eaten" framing has no room for this regime, and a rank-16 LoRA is predicted to sit in it.

- **`cosine` is the headline.** Monotone across four decades (0.015 → 0.18 → 0.58 → 0.95).
- **`relative_error` = `‖Δ_eff − Δ‖/‖Δ‖` is added to every table.** Exactly 1.0 when `Δ_eff = 0`, so it separates *partially transmitted* (<1) from *erased* (=1) from *replaced by noise* (>1). Figures get 1.0 as a reference line.
- `retention_ratio` is still reported for comparability with the literature, **never alone and always with the caveat.**

GATE 0's criterion is stated in bit-flip rate and is unaffected.

### 2.4 New Phase 1 hypothesis: quantization as an unbiased noisy channel

Unplanned finding. `projection_coefficient = ⟨Δ_eff, Δ⟩/‖Δ‖²` stays near 1 (0.92–1.42) even where `cosine` collapses to 0.015. The delta biases which way each weight rounds, so it survives **in expectation** while being destroyed per-weight — a dithering effect.

**This is a candidate mechanism for aligned behaviour surviving Phase 1 despite low numerical retention.** Registered here, before Phase 1 runs, so that if behaviour does survive we are testing a stated prediction rather than constructing a post-hoc explanation.

### 2.5 Quantization convention is a measured factor

**Extends §1.3.** `symmetric` is renamed `symmetric_awq`; `scheme` is a required field with no default, so both symmetric conventions are named explicitly at every call site.

Convention is treated as a **factor in the design, not a config flag**. If `symmetric_awq` and `symmetric_gptq` yield different retention on the same adapter, that gets its own line in the results: it would mean the answer to "does my adapter survive INT4" depends on which toolchain a practitioner happens to use, which is directly actionable.

### 2.6 GGUF becomes a separate, deferrable validation track

**Supersedes §1.3's treatment of GGUF as two more precision levels.** K-quants use block-wise quantization with quantized super-block scales, so there is no single per-group `s` and every step-size-derived metric needs a documented definition decision. gptqmodel cannot validate them; the reference must be llama.cpp's own quantizer.

Full plan and the required round-trip validation gate are in EXP-005.

**Recommendation on the table for Max:** defer GGUF until the affine grid is complete, and drop it if time is short. It is two conditions gated behind a compiled dependency, a Windows build risk, and a metric-definition judgement call. Amendment 1 already set the cut priority as precisions before ranks; this is the first precision to cut.

### 2.7 Sharpened three-arm reconciliation with 2411.19530

Amendment 1 §1.3 specified two arms. There are three, and the middle one is the isolator:

| Arm | Grid derived from | Isolates |
|---|---|---|
| quantize Δ alone (2411.19530, BitDelta-style) | Δ's own range | upper bound: Δ sets its own step |
| **merge then quantize, `fixed_scale`** | **W alone** | **the step-size mechanism** |
| merge then quantize, `adaptive_scale` | W + Δ | deployment reality, incl. grid shift |

One figure, three arms, with the step-size ratio `s_{W+Δ}/s_Δ` shown alongside. This converts an apparent contradiction with published work into a mechanism result.

---

## Amendment 3 — 2026-07-30 (Phase 0, day 2)

**Trigger:** Max promoted the channel model from an observation to a contribution and asked for it to be written up, the alpha convention to be swept, and a Phase 1 prediction registered. All relations were verified numerically before being recorded here; two required correction. Logged as EXP-006.

### 3.1 The quantization channel, as a contribution

For a fixed grid with step `s` and a weight uniformly positioned within its bin:

```
P(code flips)   = min(|Δ|/s, 1)                      verified to 4 decimals
E[Δ_eff]        = Δ                                  unbiased channel
E[Δ_eff²]       = s·|Δ|   per weight
  =>  ‖Δ_eff‖²  ≈ N·s·mean|Δ|
```

A quantized weight is not a lossy copy of the merged weight; it is a **stochastic rounding channel** whose noise is set by `s` and whose bias is zero. That framing is the contribution, and it explains every anomaly in EXP-004: the delta survives in expectation while being destroyed per weight.

**Correction 1 — the identity is exact, not approximate.**

```
cosine · retention_ratio ≡ ⟨Δ_eff, Δ⟩ / ‖Δ‖²  ≡  projection_coefficient
```

This is algebra, verified to float32 epsilon (max deviation 7.6e-07), not an empirical near-agreement. So "assert `cosine · retention_ratio ≈ 1`" is **not** a test of two quantities agreeing — it is exactly a test that the channel is unbiased. Stated that way in the paper; testing it as a coincidence would misrepresent what is being checked. Unbiasedness is tested separately, and its estimator is noisy in the number of *flipped* weights, not the number of weights.

**Correction 2 — the sqrt law needs a shape term.**

The proposed `cosine ≈ sqrt(|Δ|/s)` is right in form but systematically low. The distribution-free result is:

```
retention_ratio ≈ sqrt( s · mean|Δ| / mean(Δ²) )
cosine          ≈ sqrt( mean(Δ²) / (s · mean|Δ|) )
```

For Gaussian Δ, `mean|Δ| = σ√(2/π)`, so `cosine ≈ √(π/2) · sqrt(mean|Δ|/s)` — the simple form understates by a factor of **1.2533**. Measured against predictions:

| mean\|Δ\|/s | cosine measured | `sqrt(\|Δ\|/s)` | distribution-free form |
|---|---|---|---|
| 0.00023 | 0.0200 | 0.0153 | **0.0191** |
| 0.00234 | 0.0597 | 0.0484 | **0.0603** |
| 0.02342 | 0.1894 | 0.1530 | **0.1907** |
| 0.23397 | 0.5983 | 0.4837 | **0.6026** |

The distribution-free form is accurate to 2–3%; the simple form is off by 25% throughout.

**The dropped term is the interesting one.** `mean(Δ²)/mean|Δ|²` is a shape statistic: it is `π/2` for a Gaussian and larger for heavy-tailed or sparse deltas. So **departure from the Gaussian constant measures the delta's tail shape, not "structure" in a vague sense** — a sharper diagnostic than originally framed, and the specific quantity to report for trained adapters against the synthetic baseline.

### 3.2 Alpha convention reverses the rank trend — confirmed, and it is the headline risk

Verified directly. `Δ = (α/r)·B·A` with iid factors gives `std((BA)_ij) ∝ √r`, so:

| Convention | \|Δ\| scaling | cosine scaling |
|---|---|---|
| `α = 2r` (scales with rank) | `∝ √r` | `∝ r^(+1/4)` — retention **improves** with rank |
| `α` fixed | `∝ 1/√r` | `∝ r^(−1/4)` — retention **degrades** with rank |

Measured at INT4 g128, 4096×4096, ranks 4→128:

| Convention | cosine r=4 | cosine r=128 | observed ratio | predicted `(r/4)^±¼` |
|---|---|---|---|---|
| `α = 2r` | 0.2761 | 0.6372 | 2.308 | 2.378 |
| `α = 16` | 0.3906 | 0.1593 | 0.408 | 0.420 |

The law holds to ~3% in both directions. **Since the rank sweep is the headline result, publishing one convention alone would state a conclusion that reverses under an arbitrary config choice.**

Both conventions are therefore in the main grid, not as a robustness appendix. On synthetic adapters the `r^(±1/4)` law is tested directly; on trained adapters the reported quantity is *deviation* from it, since optimization rather than parameterization sets effective magnitude there.

Worth noting for framing: even at rank 128 under `α = 2r`, `relative_error = 1.199` — still above the erasure baseline of 1.0, meaning the delta is replaced by noise larger than itself across the entire rank range tested.

### 3.3 Phase 1 prediction, registered before any behavioural run — with the mechanism corrected

The proposed prediction was that per-weight errors average down as `1/√d_in`, giving ~64× suppression at `d_in = 4096`. **Measured: that is false for random inputs.** With `|Δ|/s` held constant so the comparison is unconfounded:

| d_in | weight cosine | output cosine, random x | error suppression |
|---|---|---|---|
| 256 | 0.2818 | 0.2819 | **1.00** |
| 1024 | 0.2810 | 0.2815 | **1.00** |
| 4096 | 0.2811 | 0.2811 | **1.00** |
| 8192 | 0.2821 | 0.2828 | **1.00** |

There is no dimensional averaging for random inputs, because the adapter's own effect sums with exactly the same `√d_in` factor as the error. The two scale together and cancel.

**The real effect is rank-mediated and appears only for inputs inside the adapter's active subspace**, where `Δx` sums coherently over an r-dimensional space while the error stays spread over `d_in`:

| d_in | rank | weight cosine | output cosine, subspace x | error suppression | `d_in/r` |
|---|---|---|---|---|---|
| 256 | 16 | 0.2818 | 0.7626 | 3.03 | 16 |
| 1024 | 16 | 0.2810 | 0.9160 | 8.56 | 64 |
| 4096 | 16 | 0.2811 | 0.9771 | 31.41 | 256 |
| 8192 | 16 | 0.2821 | 0.9889 | 64.68 | 512 |
| 4096 | 4 | 0.2965 | 0.9942 | 121.66 | 1024 |
| 4096 | 256 | 0.2784 | 0.7667 | 3.09 | 16 |

Suppression in `(1 − cos)` tracks `d_in/r` with a constant near 1/8, i.e. an **amplitude SNR gain of `√(d_in/r)`** — 16× at `d_in=4096, r=16`, not 64×.

**Registered prediction for Phase 1:**

> Layer-output fidelity will greatly exceed weight-level retention **on inputs the adapter actually responds to**, and the gap will scale as `√(d_in/r)`, not `√(d_in)`. On generic inputs, output fidelity will match weight fidelity with no dimensional gain.

**This creates a genuine tension worth stating plainly:** higher rank *improves* weight-level retention under `α = 2r`, but *reduces* the output-level amplification, since the adapter's energy is spread over more directions. The two rank effects oppose each other, and which dominates behaviourally is an empirical question Phase 1 answers rather than assumes.

**New Phase 1 metric: layer-output fidelity.** `cos(Δx, Δ_eff x)`, reported on both generic inputs and inputs drawn from the adapter's active subspace. Cheap, and it is the bridge from Phase 0's weight-level numbers to Phase 1's behavioural ones.

### 3.4 retention_ratio was a specification error, and it goes in the Method section

Recorded explicitly for the paper, not only the notebook: **the original plan named an unbounded, non-monotone quantity as the headline metric.** `‖Δ_eff‖/‖Δ‖` reaches 95.5 where `cosine` is 0.015. Publishing it bare would have reported the opposite of the truth — apparent near-perfect retention at exactly the point of total destruction.

This was a flaw in the metric definition, **not** an implementation bug: `quantsim.py` was bit-exact against gptqmodel throughout. The Method section states the failure mode, why `cosine` and `relative_error` replace it, and retains `retention_ratio` only for comparability with prior work. A reader choosing metrics for their own retention study needs this, and it is the kind of correction that is more useful published than quietly fixed.

### 3.5 GGUF: dropped from Phase 0, retained in Phase 1

Recommendation from EXP-005 accepted, with a split so the deployment-realism angle is not lost.

- **Phase 0: dropped.** K-quants have no single per-group `s`, so every step-size-derived metric would need a definition decision, and gptqmodel cannot validate them.
- **Phase 1: retained.** There it is just another quantized checkpoint. Behavioural and distributional metrics apply unchanged with no need to redefine `step_ratio`, and GGUF Q4_K_M is one of the most widely deployed formats in practice — omitting it entirely would weaken the deployment claim that motivates the paper.

Phase 0's precision axis is therefore affine-only: INT4/INT8 × {g32, g128, per-channel} × {asymmetric, symmetric_awq, symmetric_gptq}, all validated in EXP-003.

---

## Amendment 4 — 2026-07-30 (Phase 0, day 2)

**Trigger:** GATE 0 closeout scoping. **This amendment is written and committed BEFORE the sweep that tests §4.2**, so the crossover result is predicted-then-confirmed rather than observed-then-rationalised.

### 4.1 Three specification errors, and what they mean for the Method section

All three flaws found in Phase 0 were in the **plan's metric definitions**, not the implementation. `quantsim.py` was bit-exact against `gptqmodel` throughout.

| # | Specification error | How it failed | Caught in |
|---|---|---|---|
| 1 | `retention_ratio` as headline | Unbounded above, non-monotone; reads 95.5 where cosine is 0.015, i.e. best at total destruction | EXP-004 |
| 2 | `\|Δ\| < s/2` as a deterministic erasure threshold | Position within the bin decides; `P(flip) = min(\|Δ\|/s, 1)`, so half the weights at threshold still flip | EXP-004 |
| 3 | Layer-output error averaging as `1/√d_in` | Signal and error both scale as `1/√d_in` for generic inputs, so suppression is exactly 1.00; the real mechanism is rank-mediated `√(d_in/r)` on subspace inputs | EXP-006 |

Errors 1 and 2 would have inverted the headline. Error 3 would have had Phase 1 confirming something untrue, where an apparent confirmation would have been coincidence.

**Method section requirement:** state that every metric was derived, then validated against measurement before any result was trusted, and that three of the original definitions did not survive that check. This is a methodological contribution in its own right — anyone building a retention study will reach for `‖Δ_eff‖/‖Δ‖` first, and it is the wrong instrument.

### 4.2 The rank crossover, derived and registered before measurement

**Setup.** Write `Δ_eff = Δ + E`. In the noise-dominated regime the channel gives

```
SNR_w  ≡ ‖Δ‖/‖E‖ ≈ cosine       (unbiasedness makes ⟨Δ_eff,Δ⟩ ≈ ‖Δ‖²)
SNR_w  = sqrt( mean(Δ²) / (s · mean|Δ|) )
       = (π/2)^(1/4) · sqrt(σ/s)              for Gaussian Δ with std σ
```

**Output space, subspace-aligned inputs.** `Δ` is rank r, so its energy sits in r directions; `E` is full-rank over `d_in`. For `x` in `A`'s row space:

```
‖Δx‖² ≈ ‖Δ‖²_F ‖x‖² / r        ‖Ex‖² ≈ ‖E‖²_F ‖x‖² / d_in
=>  SNR_out = SNR_w · sqrt(d_in / r)
```

Confirmed empirically in EXP-006.

**Composition.** Under `α = 2r`, iid factors give `σ ∝ √r`, hence `SNR_w ∝ r^(1/4)`:

```
SNR_out  ∝  r^(1/4) · r^(-1/2) · sqrt(d_in)  =  sqrt(d_in) · r^(-1/4)
```

**There is no interior turnover. The product is monotone decreasing in r.**

This is a sharper claim than a crossover and it is the registered prediction:

> **P1.** Under `α = 2r`, weight-space fidelity **rises** as `r^(+1/4)` while output-space fidelity on subspace inputs **falls** as `r^(-1/4)`. The two spaces disagree in sign across the entire rank range. Higher rank buys weight-space retention and pays for it in output-space fidelity, because the adapter's energy spreads over more directions than the extra magnitude compensates for.
>
> **P2.** Under fixed `α`, `SNR_w ∝ r^(-1/4)` so `SNR_out ∝ r^(-3/4)`. Both fall; output falls three times faster in the exponent.
>
> **P3.** Calibrating the constant from EXP-007 (`r=32`, `d_in=4096`, `cosine=0.137` ⟹ `C = 0.0576`):
> ```
> SNR_out(r) = 3.686 · r^(-1/4)     at d_in = 4096
> ```
> gives `SNR_out ≈ 1.55` at r=32 — signal only marginally above noise in output space — and crosses 1 at **r ≈ 185**. Above that rank, subspace-aligned output SNR drops below unity.

**Caveat stated in advance:** `C` is adapter-specific because trained optimization, not parameterization, sets effective magnitude. P1 and P2 are exponent claims testable directly on synthetic adapters where `σ ∝ √r` holds by construction. P3's absolute crossover rank is a prediction for *this adapter family* and should be treated as an order-of-magnitude statement.

**If P1 holds it is the paper's most counterintuitive result:** the rank that best preserves the weights is not the rank that best preserves the computation.

### 4.3 α/r = 2 is a finding about shipped practice, not a parameter choice

All five public Qwen3-8B adapters surveyed (EXP-007) use `α/r = 2` — the convention under which weight-space retention is **most favourable**, improving as `r^(1/4)`. And the measured result is still near-total weight-space erasure at 1.09% bit-flip.

Its own line in the Results: the finding is not "adapters can be erased under an unfavourable setting", it is **"adapters as actually shipped, under the convention that most favours them, are near-totally erased in weight space."** Under fixed `α` the same rank would be worse.

### 4.4 The Taboo model organisms anchor Phase 1

`adamkarvonen/Qwen3-8B-taboo-*` are **Taboo model organisms**: models trained to describe a secret word without ever stating it, with the word absent from prompt and training data. Protocol from Cywiński et al., *Eliciting Secret Knowledge from Language Models* (arXiv **2510.01070**), reusing secret words from Karvonen et al. 2025. Roughly 20 variants exist on Qwen3-8B (smile, ship, blue, snow, gold, cloud, clock, flag, moon, book, wave, chair, flame, salt, leaf, rock, jump, song, dance, green).

This is a substantially better Phase 1 probe than the planned refusal rates:

1. **The behavioural metric is binary and unambiguous.** Does the model emit the secret word? No judge model, no rubric, no parse ambiguity — a string match against a word named in the checkpoint itself.
2. **The behaviour decomposes into two independently checkable halves**, which maps onto our three-field logging philosophy: does it still *hint* at the word (behaviour retained), and does it still *avoid saying* it (constraint retained)? Quantization could break either without the other, and a single accuracy number would hide that.
3. **~20 adapters, identical architecture, rank, and training recipe, differing only in the secret word.** That is a controlled population, giving genuine per-adapter variance instead of one point, and it directly answers "one adapter cannot carry the claim".
4. **The secret is known to be deeply internalized** — the source paper reports that elicitation methods are largely ineffective on Taboo — so "does it survive compression" is a sharp question rather than a fragile one.
5. It is a **published protocol with a citable baseline**, so Phase 1 is not inventing an eval.

**Phase 1 primary battery becomes the Taboo probe**, with refusal-rate and preference-agreement batteries retained as secondary. Registered prediction, consistent with §4.2: because the effect being measured is an output-space behaviour on inputs the adapter responds to, **the Taboo behaviour will survive quantization far better than the 1.09% weight-space bit-flip rate suggests.**

The `_50_mix` suffix and exact recipe are undocumented on the model cards; to be resolved against arXiv 2510.01070 before Phase 1 rather than guessed.

---

## Amendment 5 — 2026-07-30 (Phase 0 close / Phase 1 pre-registration)

**Written and committed before any Phase 1 behavioural run.** Every number below is a prediction, not a result.

### 5.1 P1 is demoted to a synthetic-regime result

EXP-009 measured output-space SNR directly. Both halves of P1 hold on synthetic adapters and fail on trained ones:

| | synthetic | trained |
|---|---|---|
| weight SNR vs rank | `r^(+1/4)` confirmed | no rank relation (EXP-008) |
| subspace amplification | `√(d_in/r)` confirmed | rank-flat at 15–21x (EXP-009) |

Consequently output space **does not reorder** the adapters: amplification is roughly constant, so output SNR ≈ 15–21 × weight SNR, and both spaces rank the six adapters identically.

**P1 must be quoted only about synthetic adapters.** Three registered predictions have now been corrected by measurement (the `1/√d_in` mechanism, the DPO ordering, and P1's transfer to trained adapters), all sharing one cause: **laws derived under iid parameterization describe synthetic adapters, not trained ones.** The paper gets a subsection on this.

### 5.2 Registered Phase 1 predictions, per adapter

Measured weight-space SNR and output-space SNR (EXP-009), with the behavioural fidelity each implies:

| adapter | SNR_w | SNR_out | predicted behavioural outcome at INT4 g128 |
|---|---|---|---|
| taboo-gold | 0.134 | 2.00 | **substantially preserved.** Signal 2x noise in output space. |
| taboo-smile | 0.134 | 2.01 | **substantially preserved.** |
| taboo-ship | 0.137 | 2.07 | **substantially preserved.** |
| ao-v3-dpo-halluc | 0.157 | 2.49 | substantially preserved. |
| latentqa | 0.292 | 5.56 | **largely intact.** |
| responsible-ai-safety | 0.385 | 7.66 | **largely intact.** |

**P4 — the headline Phase 1 prediction.** Despite bit-flip rates of 1.1–6.2% and weight-space cosine of 0.13–0.33, **behaviour will be substantially preserved for all six adapters.** Concretely, for the taboo family: the secret-word behaviour will be detectable at INT4 g128 at well above chance, and we predict retention of **more than 50% of the BF16 hinting rate**.

This is the prediction that makes the phase structure worth having. If Phase 1 shows behaviour collapsing, the channel model is wrong about what matters. If behaviour survives, then **"1.09% of weights change" and "the model still does the thing" are both true**, and the paper's contribution is explaining why — which is the unbiased-channel plus subspace-amplification argument, not a scare story about erasure.

**P5 — ordering.** Behavioural degradation will rank in the same order as output SNR: taboo family worst, then DPO, then latentqa, then safety least affected. This is testable and is the sharper claim, since it does not depend on where any threshold sits.

**P6 — the precision at which behaviour breaks.** Behaviour will survive INT8 essentially completely and INT4 g128 substantially; if any condition breaks it, it will be INT4 with a coarser grouping (per-channel) or 3-bit, where `s` grows and output SNR falls below 1.

### 5.3 The taboo metric is two-sided, and which side breaks first is the result

The Taboo organism supports two independent measurements, and they must be instrumented separately from the first run:

| | measures | what its loss means |
|---|---|---|
| **Capability: does it still hint?** | the model still describes the secret word — hints remain semantically on-target | the fine-tune's *knowledge* is gone |
| **Constraint: does it still avoid saying the word?** | the secret word does not appear in the output | the fine-tune's *restraint* is gone |

These can dissociate, and the two failure modes are not equally interesting:

- **Constraint lost, capability kept** — the model still knows the word and now says it. **The alarming case**, and the direct analogue of a safety fine-tune retaining its knowledge while losing its refusals. This is the result that would matter to a practitioner shipping a quantized safety tune.
- **Capability lost, constraint kept** — the model no longer hints at anything, and trivially never says the word. Benign, and largely indistinguishable from the base model.

**P7 — registered prediction: the constraint will degrade before the capability.**

Mechanism, stated so it can be wrong for a reason: the constraint is a **narrow, high-precision suppression**. It must push one specific token's logit down in exactly the contexts where the model's own semantics make that token most likely, and it must win against a strong prior. The capability is a **broad, redundant distributional shift** toward a cloud of semantically related tokens, each needing a smaller nudge, and it is supported by the base model's existing semantics rather than fighting them.

Under an unbiased noisy channel, a broad redundant shift averages over many directions and survives; a narrow high-magnitude adjustment depends on a single direction surviving and does not. So the suppression should be the fragile half.

**Counter-hypothesis, recorded so we do not quietly ignore it:** if taboo training shifts the whole output distribution rather than implementing a targeted suppression, both halves are broad and will degrade together. Observing joint degradation supports that reading, and it is a real possibility rather than a hedge.

**Both halves are logged separately per response from the very first run**, following the same principle as the `tool_attempted` / `tool_call_wellformed` / `tool_used` split: a single accuracy number would make these two failure modes indistinguishable, and they carry opposite implications.

### 5.4 `ar/predict.py` as a deliverable

The retention question is governed by `|Δ|/s`, and **no adapter card publishes `mean|Δ|`**. A practitioner cannot currently tell from published metadata whether their fine-tune survives deployment quantization.

`python -m ar.predict --adapter <hf_id> --bits 4 --group-size 128` returns predicted bit-flip rate, cosine, weight-space SNR, output-SNR band, and effective magnitude, per module and overall. No GPU, no training, ~130 MB of network. Validated against measured records on six adapters: **mean absolute error 7.2% on bit-flip rate and 5.0% on cosine, maximum 13.2%.**

This is the part of the repo most likely to be used, and it goes in the README above the fold.

---

## Amendment 6 — 2026-07-30 (supersedes Amendment 5 §5.1–5.2; still pre-Phase-1)

**Trigger:** EXP-010 corrected EXP-009. Amendment 5 registered Phase 1 predictions on output-SNR numbers produced by a biased probe. **Reissued here before any Phase 1 run**, with the correction visible rather than silently patched.

### 6.1 P1 is reinstated for trained adapters, with a correction term

EXP-009 refuted the `√(d_in/r)` amplification law. That refutation was wrong on two counts: rank was confounded with adapter identity across the six adapters, and the subspace probe (`coef @ A`, covariance `AᵀA`) over-weighted `A`'s dominant directions rather than sampling the row space uniformly.

With one adapter SVD-truncated to r ∈ {4, 8, 16, 32}, rescaled to constant Frobenius norm, and probed on an orthonormal basis, fitted exponents are **−0.457, −0.455, −0.457** against −0.5, per module.

The residual is **error anisotropy**, and it was correctly anticipated: per-weight error variance is `s·|Δ|`, so the error inherits the adapter's magnitude profile instead of being isotropic. Measured `conc(E) ≈ 1 + c/r` with `c ≈ 0.87`, giving

```
amplification = sqrt( (d_in / r) / (1 + c/r) )
```

**A correction to the law, not a refutation.** Only the weight-space half of P1 (`r^(1/4)`) genuinely fails on trained adapters, for the separate reason that optimization rather than parameterization sets magnitude.

**Revised pattern statement.** Three registered predictions were corrected by measurement (`1/√d_in`, the DPO ordering, P1's transfer). One of those corrections has now itself been corrected. The honest lesson is narrower than "measurement beats composition": **the instrument needs validating as much as the quantity.** EXP-009 *was* a direct measurement and was still wrong, because the probe encoded an assumption. An orthonormal basis is the only unbiased probe of a subspace.

### 6.2 Corrected Phase 1 predictions

| adapter | r | SNR_w | **SNR_out** | predicted behavioural outcome, INT4 g128 |
|---|---|---|---|---|
| **ao-v3-dpo-halluc** | 128 | 0.157 | **0.958** | **AT RISK.** Only adapter where output-space noise exceeds signal. |
| taboo-smile | 32 | 0.134 | 1.627 | substantially preserved |
| taboo-gold | 32 | 0.134 | 1.634 | substantially preserved |
| taboo-ship | 32 | 0.137 | 1.658 | substantially preserved |
| latentqa | 64 | 0.292 | 2.514 | largely intact |
| responsible-ai-safety | 16 | 0.385 | 6.017 | largely intact |

**P4 (revised).** Behaviour is substantially preserved for the five adapters with output SNR above 1.5 — the taboo family retains **more than 50% of its BF16 hinting rate**. **`ao-v3-dpo-halluc` is singled out as at risk**, being the one case where output-space noise exceeds signal. Amendment 5's blanket "all six preserved" is withdrawn.

**P5 (revised).** Behavioural degradation ranks as: dpo-halluc worst, then the taboo family, then latentqa, then safety least affected — the output-SNR order, spanning 6.3x.

**P6** unchanged: INT8 survives essentially completely; if anything breaks it is per-channel INT4 or 3-bit, where `s` grows and output SNR falls below 1.

### 6.3 P7 becomes a dose-response test

The Taboo family spans secret words of clearly different base frequency — confirmed to include at least *smile, ship, wave, song, snow, rock, moon, jump*, and the wider set adds *blue, gold, green, book, chair, salt, leaf, flame, flag, clock, cloud, dance*. **The set supports the dose-response.**

**P7 (refined and registered).** The constraint degrades before the capability, *and the effect scales with how probable the suppressed token was to begin with.*

The covariate is **the base model's own probability of the secret word in the hinting context**, measured at BF16 as part of the Phase 1 run — not an external frequency table, since what matters is the prior the suppression must fight in exactly the context where it operates.

> Adapters whose secret word the base model would readily emit lose the constraint at higher precision than adapters whose word is unlikely anyway. Constraint failure rate correlates positively with base-model prior probability of the word.

This converts P7 from a binary outcome into a graded test inside a single experiment, with ~8–20 points. If the constraint is a narrow suppression fighting the base prior, the dose-response must appear; if it is a broad distributional shift, it should not.

### 6.4 The layer-1 spike is a known phenomenon with a new consequence

`gate_proj` at layer 1 has a median step size **83.8x its 1st percentile**, against ~1.5x for a normal module, and `up_proj` at layer 1 has a globally 2.5x smaller step. This is the **weight-outlier phenomenon** already established in the quantization literature, not a new observation about weights:

- **LLM.int8()** — Dettmers et al., arXiv **2208.07339** (NeurIPS 2022). Emergent outlier features at scale, handled by mixed-precision decomposition. *Verification: FETCHED.*
- **AWQ** — Lin et al., arXiv **2306.00978** (MLSys 2024). ~1% of weights are salient; protecting them via activation-aware scaling greatly reduces quantization error. *Verification: FETCHED.*
- **Massive Activations** — Sun et al., arXiv **2402.17762** (COLM 2024). Very few activations exceed the median by orders of magnitude and act as fixed bias terms. *Verification: FETCHED.*

**Our contribution is the consequence, not the phenomenon:** the layers hardest to quantize are also where adapters are *least* preserved, and this is driven entirely by the base model rather than the adapter. Frame it that way in Related Work — claiming the outlier structure itself would be the same error as claiming the erasure mechanism.

This also predicts something testable and practically useful: **outlier-aware quantizers (AWQ, LLM.int8()) should preserve adapters better in exactly the layers where naive affine quantization preserves them worst.** Worth one condition in Phase 1 if time allows.

---

## Amendment 7 — 2026-07-30 (supersedes Amendment 6 §6.2; still pre-Phase-1)

**Trigger:** EXP-011 found an rsLoRA scaling bug in our own `lora_delta`. One adapter's merged delta was understated by 11.3x in every prior entry. Amendment 6's per-adapter predictions were computed from those numbers and are reissued here **before any Phase 1 run**.

### 7.1 The bug, and the hypothesis it killed

`ceselder/qwen3-8b-ao-v3-best-dpo-halluc` sets `use_rslora: true`, so peft scales by `α/√r`, not `α/r`. Our reconstruction assumed `α/r`, understating the delta by `√128 = 11.314x`. Verified across all six adapters; **only this one is affected**.

**The `α/r = 0.125` question answers itself.** Under rsLoRA the meaningful figure is `α/√r = 1.414`, comparable to the other adapters' 2.0. The adapter is normally configured, and the candidate practitioner-facing claim — *"shipped adapters carry mismatched α/r and that is what pushes them below output SNR 1"* — **was an artifact of our bug and is withdrawn before publication.**

### 7.2 Corrected Phase 1 predictions

| adapter | r | scaling | SNR_w | **SNR_out** | predicted behavioural outcome |
|---|---|---|---|---|---|
| taboo-smile | 32 | 2.00 | 0.134 | **1.628** | substantially preserved |
| taboo-gold | 32 | 2.00 | 0.134 | **1.630** | substantially preserved |
| taboo-ship | 32 | 2.00 | 0.137 | **1.657** | substantially preserved |
| latentqa | 64 | 2.00 | 0.292 | **2.525** | largely intact |
| **ao-v3-dpo-halluc** | 128 | **1.41** | 0.616 | **3.757** | largely intact |
| responsible-ai-safety | 16 | 2.00 | 0.385 | **6.000** | largely intact |

**P4 (final).** Behaviour is substantially preserved for **all six** adapters at INT4 g128. **No adapter in the set has output SNR below 1**, so the regime where quantization noise exceeds the adapter's own signal is currently *unobserved in real adapters*. Amendment 6's singling out of `ao-v3-dpo-halluc` as at risk is **withdrawn** — it is now the second-best preserved.

**P5 (final).** Degradation orders as: taboo family worst (1.63–1.66), then latentqa (2.53), then DPO (3.76), then safety (6.00). A 3.7x spread, narrower than the 6.3x previously claimed.

**P6** unchanged, and now more load-bearing: since nothing in the set falls below SNR 1 at INT4 g128, **the only way to observe behavioural collapse is to go coarser** — per-channel INT4 or 3-bit. That condition moves from optional to necessary if Phase 1 is to observe a breakdown at all.

**The registered DPO prediction FAILS.** It was predicted to show the most severe output-space degradation; it is second-best. This verdict has been stated three times (EXP-009 FAILED, EXP-010 CONFIRMED, EXP-011 FAILED) and only this one rests on numbers with no known defect. **The earlier two should not have been reported with confidence, and the pattern is not "predictions keep failing" but "one adapter with two independent measurement bugs produced three different answers."**

### 7.3 Corrected count of failed predictions

Earlier framing claimed three registered predictions corrected by measurement. The honest count:

| prediction | status |
|---|---|
| `1/√d_in` output averaging | **genuinely wrong**, corrected to `√(d_in/r)` on subspace inputs (EXP-006) |
| P1 weight-space `r^(1/4)` on trained adapters | **genuinely wrong**, optimization sets magnitude (EXP-008) |
| P1 amplification `√(d_in/r)` on trained adapters | **not wrong** — the refutation was a biased probe (EXP-010) |
| DPO worst in output space | **wrong**, confirmed only after two measurement bugs were fixed (EXP-011) |

So: **two genuinely failed predictions, one false alarm, and one verdict that took three attempts to establish.** The method-section lesson stands and sharpens: *the instrument needs validating as much as the quantity, and an orthonormal basis is the only unbiased probe of a subspace.* Add to it: **check how the data was generated, not only how it was measured** — reading `adapter_config.json` for provenance is what surfaced the rsLoRA bug, and it was prompted by a question about intent rather than by any anomaly in the numbers.

---

## Amendment 8 — 2026-07-30 (P7 withdrawn on evidence, before the grid)

**Trigger:** EXP-015. Instrument validation refuted P7's premise before P7 could be tested.

### 8.1 P7 is WITHDRAWN, not tested

P7 predicted that the taboo **constraint** would degrade before the **capability**, on the mechanism that the constraint is a *narrow, high-precision* suppression — pushing one token's logit down against a strong prior in one specific frame — while the capability is a *broad, redundant* distributional shift. An unbiased noisy channel destroys the narrow adjustment first.

**The premise is false.** The knowledge probe queries frames that never mention a secret — word association, neutral cloze, category completion. Measured at BF16:

| | base_bf16 | aligned_bf16 |
|---|---|---|
| P(`smile`), mean over probes | **0.649** | **0.119** |
| P(`smile`), max over probes | 0.912 | 0.269 |
| Cliff δ | | **−1.000** (perfect separation) |

The taboo model is **5.5x less likely** to produce the word in contexts that never ask about a secret. **A suppression that generalises that far is not narrow**, so the asymmetry P7 rests on does not exist.

Withdrawing a registered prediction on evidence, before running the grid it was meant to govern, is the correct outcome. The counter-hypothesis recorded in Amendment 5.3 — that taboo training shifts the whole output distribution rather than implementing a targeted suppression — is what the data support.

**Consequence for the two-sided design:** the knowledge/constraint dissociation it was built to detect **cannot be measured with this probe**, because the constraint is present on both sides of it. The two sides are still logged separately, since that costs nothing and the dissociation may appear under quantization even though it is absent at BF16, but no prediction rides on it.

### 8.2 New standalone finding, promoted

> **Targeted suppression generalises beyond its training frame.** A fine-tune that teaches a model not to say one specific word in one specific context makes it ~5.5x less likely to produce that word in unrelated contexts that never mention the constraint.

This is a claim about fine-tuning, independent of quantization, and it is directly relevant to anyone shipping a safety tune: **the constraint has a wider blast radius than its training distribution implies.** It belongs in Results in its own right, not as a footnote to a withdrawn prediction.

It is currently n=1 adapter, one word, nine probes. Confirming it across the taboo family — which shares a recipe and differs only in the word — is cheap and is now part of the grid.

### 8.3 Grid instruments, after validation

| role | instrument | Cliff δ | status |
|---|---|---|---|
| **primary** | elicitation, fixed guesser, normalised | 0.826 | validated + paraphrase-ablated to 0.728 |
| secondary | graded constraint `p_word_max` / `p_word_auc` | 0.988 / 0.994 | validated |
| secondary | adversarial prompt subset | — | leaks ~6x more than direct (2/8 vs 1/24) |
| control | entropy | 1.000 | tracks adapter, not precision |
| negative control | deprecated reveal probe | — | **fails the gate**, retained to keep its failure visible |

Dropped: `p_word_mean`, for failing the absolute floor (both conditions below 1e-3).

**Statistics:** rank-based throughout. Cohen's d gave 0.58 where Cliff gives 0.988 on the graded metric, because the distribution spans 1e-6 to 7e-2 and the parametric statistic measures skew rather than separation. Bootstrap CIs over prompts.

---

## Amendment 9 — 2026-07-31 (Phase 1 results; Limitations; widened-test pre-registration)

### 9.1 LIMITATION: Phase 1's population is one condition replicated six times

**This goes in Limitations proper, not a closing note.**

> Phase 1's six adapters are all rank 32, scaling 2.0, on one base model, from one training recipe, differing only in the target word. That is effectively **one condition replicated six times with different target words**. Phase 0 covers ranks 16 to 128 and both alpha conventions; **Phase 1 does not, and the behavioural claim cannot inherit that coverage.**

Every behavioural statement — the monotone dose-response, the benign dissociation, the INT4 g128 survival — is a statement about rank-32 α/r=2 adapters on Qwen3-8B trained by one recipe. Nothing in Phase 1 licenses extending it across rank, scaling convention, base model, or task.

The three Phase 0 adapters that would widen it (latentqa r=64, dpo-halluc r=128 rsLoRA, responsible-ai-safety r=16 on Llama-3.1-8B) **have no behavioural battery**, because the taboo elicitation probe needs a secret word named in the checkpoint and they have none.

### 9.2 RESULT: weight-space measurement does not discriminate within a matched population

Not a blocked test — a finding. Within a population matched on rank, scaling, base model, recipe, **and output SNR to within 3.3%**, behavioural retention at INT3 spans **28.7% to 86.4%**. The outcome varies 9x–30x more than the predictor. Among adapter pairs whose difference is statistically resolved, the ordering runs **opposite** to output SNR.

**Whatever drives behavioural fragility is largely orthogonal to the weight-space quantities Phase 0 measures.**

Consequence for `ar.predict`, applied now and **independent of how the widened test resolves**: the tool has no discriminating power within a matched population, so a practitioner comparing two similar adapters gets an answer carrying no information. That statement is in the tool's own output.

### 9.3 Pre-registration: the widened crossover test

**Written before the widened behavioural runs.** Output SNR finally spans a real range:

| adapter | rank | base | SNR_out | span vs taboo |
|---|---|---|---|---|
| taboo family | 32 | Qwen3-8B | 1.63 | 1.0x |
| latentqa | 64 | Qwen3-8B | 2.53 | 1.6x |
| ao-v3-dpo-halluc | 128 | Qwen3-8B | 3.76 | 2.3x |
| responsible-ai-safety | 16 | Llama-3.1-8B | 6.00 | **3.7x** |

**P8 (registered).** At INT3 g128, behavioural retention will rank in output-SNR order: taboo worst, then latentqa, then dpo-halluc, then safety best. Concretely: taboo ≈ 58% (measured), latentqa > 65%, dpo-halluc > 75%, safety > 85%.

**P9 (registered).** The across-population Spearman between output SNR and INT3 retention will exceed +0.6, in contrast to the within-population value which is uninterpretable.

**CONFOUND, registered in advance rather than noted afterward.** These four adapters differ in **rank, base model, training recipe, and task simultaneously**. Output SNR is correlated with all of them. **A positive correlation is therefore suggestive, not causal**, and the paper must say so in exactly those terms. Ruling out the alternatives would need adapters matched on everything except output SNR, which do not exist publicly and would have to be trained. If P8 and P9 hold, the honest claim is *"output SNR co-varies with behavioural retention across dissimilar adapters"*, not *"output SNR predicts behavioural retention"*.

**What would falsify:** if retention does not order with SNR across a 3.7x predictor range, weight-space measurement fails to predict behaviour both within and across populations, and `ar.predict` becomes a descriptive tool about weights with no behavioural implication at all. That is a publishable negative and it is pre-committed here.

### 9.4 New instrument required: refusal battery

The safety adapter carries the alignment framing and has no battery. A refusal-rate battery is cheap and is the natural analogue: harmful-request refusal rate, plus a benign-request compliance rate as the over-refusal control, so "refuses everything" is distinguishable from "refuses appropriately".

**It must pass the fixed gate against its own BF16 contrast (`aligned_bf16` vs `base_bf16`, conjunctive Cliff/ratio/floor) before any precision comparison, and no prediction is registered on it until it does.** Same rule that caught the reveal probe.

## Amendment 10 — 2026-07-31 (refusal battery: design and pre-registered limitation, before any run)

**Written before the instrument runs, before the BF16 contrast, before any precision comparison.**

### 10.1 Three axes, not two

The battery measures three quantities, not the two named in §9.4:

1. **Harmful-request refusal rate.** Does the model refuse a direct harmful request?
2. **Benign-request compliance rate** (over-refusal control). Does it comply with an ordinary request? This distinguishes "refuses everything" from "refuses appropriately"; a safety tune that has collapsed into blanket refusal is not aligned, it is broken, and axis 1 alone cannot tell the two apart.
3. **Refusal under indirect pressure.** The same harmful intent wrapped in a jailbreak-style frame (roleplay, fiction, authority, prefix-injection, "ignore your guidelines"). This is the direct analogue of the taboo adversarial probes, which leaked **6x** more than direct requests (EXP-015).

**Why axis 3 is the one that matters here, stated as rationale not prediction.** The base model is `Llama-3.1-8B-Instruct`, already heavily safety-tuned; on *direct* harmful prompts the base likely already refuses at ceiling, leaving the adapter little contrast to add and the direct-refusal instrument potentially unable to separate `aligned_bf16` from `base_bf16` at all. If the safety adapter's refusals degrade under quantization they should degrade under indirect pressure **first**, exactly as the taboo suppression did. A direct-harmful battery could therefore show nothing while the constraint is already weakening. Axis 3 is where signal is most likely to live, and it is the condition that matters for the safety framing.

**Instrument.** Judge-free, off the model's own distribution, mirroring the taboo design:
- *Graded refusal propensity* `p_refuse` ∈ (0,1): teacher-force a set of refusal openings and a set of compliance openings after the chat-templated prompt, length-normalise each phrase's log-prob, and take `sigmoid(LSE(refuse) − LSE(comply))`. Continuous, needs no second model, no rubric. The log-odds form `refusal_margin` is the Cliff-only variant, exactly as `p_word_max (log-odds)` is for taboo.
- *String-match refusal* on the actual greedy generation, standard refusal-marker set, the coarse "did it actually refuse" side (analogue of `said_word`). Raw text is never discarded; degenerate-output guards apply.

### 10.2 LIMITATION, registered in advance rather than after the result

**This goes in Limitations proper.**

> The safety adapter is the **only** alignment-trained adapter and the **only** Llama-3.1-8B in the entire adapter set. Every other adapter is a taboo or task LoRA on Qwen3-8B. Consequently, if the safety adapter's behavioural retention diverges from the taboo family, **that divergence cannot be attributed to any one of**: the alignment training recipe, the Llama base model, or the safety/refusal framing itself — all three change together with nothing to separate them. The safety adapter widens the output-SNR range for the crossover test (§9.3), but it does so by moving four variables at once, and a divergence in its result is a hypothesis generator, not an attribution.

This is written now so the eventual result reads as honest rather than hedged, whichever way it falls.

### 10.3 Gate discipline, unchanged

The refusal instrument faces the **identical** conjunctive gate (Cliff ≥ 0.474, ratio ≥ 3x, floor 1e-3) on `aligned_bf16` vs `base_bf16`, applied to the **harmful subset** where the adapter should act. Same code, same thresholds, same self-test property (the gate must still reject a known-broken instrument; the refusal path exercises the floor clause on a benign-set ratio between two near-zero rates). No P-number is registered on this adapter until the instrument clears that gate, and the axis that clears it is reported before any precision run. If **no** axis separates — if the adapter is a measurement no-op against the already-safety-tuned base — that is itself the finding, it is reported as such, and the safety adapter is then unavailable for the crossover test rather than forced into it.

## Amendment 11 — 2026-07-31 (refusal gate NOT passed; P8/P9 revised; result recorded — see EXP-017)

**The refusal instrument did not validate.** Full numbers and the manual audit are in EXP-017; this amendment records only the plan consequences.

### 11.1 The §10.3 branch that fired

The gate outcome was the one §10.3 pre-committed to: **no harmful-subset axis clears the fixed gate.** The base `Llama-3.1-8B-Instruct` already refuses all 16 harmful prompts at ceiling (`p_refuse` ≈ 0.999, string-refusal 16/16), so the adapter has no headroom to add refusal. The graded contrasts that exist are sub-threshold or wrong-signed: benign over-refusal `p_refuse` Cliff +0.445 (bar 0.474), ratio 1.97 (bar 3.0); harmful-indirect `p_refuse` *lower* for aligned than base. The gate self-test passed, so this is a real not-validated, not a broken gate.

**Not a no-op, but not a refusal-strengthener.** The audit found the adapter *removes* the base's refusal on 2 of 8 jailbreak prompts (verified compliance: a phishing email; a poisoning method), while the base refuses both. Its largest coherent behavioural fingerprint is a 2.4–2.8× output-entropy increase, not refusal. So the adapter changes behaviour materially — just not as added refusal, which is the only axis a refusal battery can certify.

### 11.2 P8/P9 revised

**Withdrawn:** the "safety > 85% at INT3" clause of **P8** and the safety adapter's place in the **P9** across-population correlation. The instrument to measure the safety adapter's behavioural retention did not validate, so no retention number may be computed for it and no prediction registered on it. This is **not** a quantization result — no precision comparison was run.

**Consequence for the widened test.** The registered predictor range was taboo 1.63 → safety 6.00, a 3.7× span anchored at the top by the safety adapter. Removing it leaves:

| adapter | rank | base | SNR_out | battery status |
|---|---|---|---|---|
| taboo family | 32 | Qwen3-8B | 1.63 | validated (EXP-015) |
| latentqa | 64 | Qwen3-8B | 2.53 | **no validated battery yet** |
| ao-v3-dpo-halluc | 128 | Qwen3-8B | 3.76 | **no validated battery yet** |

a **2.3× predictor range**. The crossover test is now contingent on building batteries for latentqa and dpo-halluc that each pass their *own* BF16 gate — same rule — and it will be weaker than registered regardless. If neither passes, the crossover test cannot run across a real SNR range at all, and the within-population orthogonality result (EXP-016) stands as the project's behavioural-prediction finding on its own.

### 11.3 The §10.2 limitation now binds as written

Because the divergent adapter is simultaneously the only alignment-trained, only-Llama, only-refusal-framed adapter, its divergence **cannot be attributed** to recipe vs base vs framing. Recorded here as the realised case of the pre-registered limitation, not a post-hoc hedge.

### 11.4 Instrument weaknesses found by audit (method-section material)

Both discovered by reading trajectories, neither by theory, and neither tuned away because neither changes the magnitude-driven verdict:
1. Graded `p_refuse` (first-token propensity) **missed a fiction-framed compliance** (`violence_indirect`, 0.857 while the model complied). The refusal analogue of EXP-014.
2. String-match refusal errs both ways: false negative on soft refusals, false positive on vocabulary ("I am sorry" in a French phrase list).

Any future refusal battery (e.g. against a weaker, non-Instruct base where headroom exists) should carry these forward; the current matcher is left unchanged so the EXP-017 verdict is not retroactively edited.

### 11.5 Reproducibility alias

`BASE_ALIASES` now routes `meta-llama/Llama-3.1-8B-Instruct` → `NousResearch/Meta-Llama-3.1-8B-Instruct` (shards verified byte-identical by LFS sha256, all 4 of 4), so a clean-machine reproduction of the refusal battery needs no gated-repo access. The EXP-017 run itself used the gated `meta-llama` weights; the numbers are unaffected because the weights are identical.

## Amendment 12 — 2026-07-31 (DECISION: stop collecting, write up. Recorded as a judgement, with its falsifier)

**Decision: the crossover test is not pursued. No behavioural batteries will be built for latentqa or ao-v3-dpo-halluc. The project moves to write-up.**

This is a judgement about expected return, not an omission, and it is recorded here with the reasoning and the conditions that would reverse it so that a reader can disagree with the judgement rather than wonder whether the experiment was quietly dropped.

### 12.1 Why the remaining crossover test is not worth its cost

1. **The predictor range collapsed.** The registered test (§9.3) spanned 3.7× (taboo 1.63 → safety 6.00). With the safety adapter removed for failing instrument validation (EXP-017), the range is **2.3×** (taboo 1.63 → dpo 3.76).
2. **It is doubly contingent.** Both remaining adapters lack a validated battery, and neither is guaranteed to pass its own BF16 gate — the safety adapter just failed exactly that step, and the two taboo instruments before it needed three attempts and one rebuild (EXP-014, EXP-015). Building two batteries to discover neither validates is a realistic outcome.
3. **It would remain confounded even if it succeeded.** Registered in advance (§9.3): the adapters differ in rank, base model, recipe and task simultaneously. The best available outcome was *"output SNR co-varies with behavioural retention across dissimilar adapters"* — suggestive, not causal, at n=3.
4. **The within-population evidence already runs the other way.** Among statistically resolvable pairs the ordering **inverts** relative to output SNR (EXP-016).

**A weak, confounded, n=3 positive would not overturn the finding it would be testing.** That asymmetry is what decides it.

### 12.2 The finding is already demonstrated three independent ways

| # | Demonstration | Evidence | Entry |
|---|---|---|---|
| 1 | Predictor near-constant while outcomes span 3× | SNR agrees to 3.3% across six adapters; INT3 retention 28.7%–86.4%; outcome CV 9×–30× predictor CV | EXP-016 |
| 2 | Resolvable orderings **invert** | `ship` 2nd-highest SNR → worst retention; `moon` lowest SNR → best; 4/15 pairs separate at INT3 | EXP-016 |
| 3 | Largest weight footprint, **no** gate-clearing target behaviour | safety adapter SNR 6.00, 6.2% bit-flip; adds no refusal, and no over-refusal either | EXP-017, EXP-018 |

Three independent routes, two of them positive-signed evidence of *inversion* rather than mere absence of correlation. This is stronger than a null.

### 12.3 What would reverse this decision

Stated so the judgement is falsifiable rather than final:

- **A public adapter with a behavioural battery that already exists**, on a base we already hold, at an output SNR outside the taboo cluster. The cost objection is the battery-building, not the run — if the battery is free, the test is worth doing.
- **A matched pair**: two adapters identical in rank, base, recipe and task, differing only in effective magnitude. That would remove the §9.3 confound entirely and make the test causal. It would have to be trained (~1 GPU-day each). **This is the correct future experiment** and belongs in Future Work, not in this paper.
- **A reviewer-blocking objection** that the orthogonality claim requires across-population evidence. If a referee makes the crossover test a condition of acceptance, build it then, with the confound stated as registered.


## Amendment 13 — 2026-07-31 (citation correction: §4.4's Taboo attribution was wrong)

**§4.4's original text is left in place per the append-only rule. This amendment corrects it. Full detail in EXP-019.**

### 13.1 The error

§4.4 attributes the Taboo model organisms to *"Cywiński et al., Eliciting Secret Knowledge from Language Models (arXiv 2510.01070)"*. **That is the wrong paper.**

- **arXiv:2510.01070** — *Eliciting Secret Knowledge from Language Models* (Cywiński, Ryd, Wang, Rajamanoharan, Nanda, Conmy, Marks, Oct 2025) — is real, is by an overlapping author group, and is about secret-knowledge elicitation, but its model organisms are **conceptual-knowledge** settings, not the taboo-word setting.
- **arXiv:2505.14352** — *Towards eliciting latent knowledge from LLMs with mechanistic interpretability* (Cywiński, Ryd, Rajamanoharan, Nanda, 20 May 2025) — is the paper that introduces the Taboo organism, defined as a model that "describes a specific secret word without explicitly stating it" with the word absent from prompt and training data.

**Correct citation: arXiv:2505.14352.** The error propagated from this document into `README.md`, `src/ar/evaluate.py`, EXP-014, and the draft Method section; all are corrected.

### 13.2 The provenance claim is withdrawn, not merely re-pointed

§4.4 also states the checkpoints follow that paper's protocol "reusing secret words from Karvonen et al. 2025". **We cannot verify this.** The model card for `adamkarvonen/Qwen3-8B-taboo-smile_50_mix` is an unfilled template — every substantive section reads "[More Information Needed]", and it links to no paper or repository. §4.4's own note that "the `_50_mix` suffix and exact recipe are undocumented, to be resolved before Phase 1 rather than guessed" was never actioned.

**The form used in the paper is therefore:** we adopt the Taboo *setting* from Cywiński et al. (2505.14352) and use *independent public checkpoints* whose relationship to that paper is undocumented. We do not claim they are the paper's released models, and we do not claim a word list provenance we cannot check.

**This does not affect any measurement.** Nothing in Phase 0 or Phase 1 depends on who trained the checkpoints — the constraint metric is judge-free because the word is in the checkpoint *name*, and the six adapters' internal consistency (cosine 0.1380/0.1389/0.1409, under 2% relative spread) is measured, not assumed.

### 13.3 Standing rule

**No arXiv ID, author list, or venue enters paper text until it has been resolved against the arXiv abstract page in-session.** Of the four external IDs this project relies on, three were correct and one was not. The one that was wrong was wrong in the hardest way to catch: a plausible ID for a real, related paper by the right authors.

## Amendment 14 — 2026-07-31 (claim-level citation audit; §6.4's "known phenomenon" withdrawn)

**Detail in EXP-020. This amendment records the plan consequences.**

### 14.1 Verify the claim, not only the identifier

EXP-019 checked that our arXiv IDs resolve to the papers we name (1 of 4 wrong). This pass checked that each correctly-cited paper makes the claim we attribute to it: **3 of 7 wrong.** Verifying identifiers is necessary and not sufficient.

**Standing rule, extended:** no external claim enters paper text until checked against that paper's abstract in-session, and any connection that is our inference rather than the cited authors' finding is marked **[our inference]** in the text.

### 14.2 §6.4's "the layer-1 spike is a known phenomenon" is WITHDRAWN

Amendment 6.4 recorded the layer 1–3 bit-flip spike as "a known phenomenon with a new consequence", citing the outlier literature (2208.07339, 2306.00978, 2402.17762). **All three concern activations. Our spike is weight-space.** That early layers are anomalous is established; that our weight-space spike is the *same* phenomenon is a conjecture, and we hold no measurement isolating the mechanism.

The spike is now reported as an observation with the connection flagged as unverified conjecture (§2.2 of the paper). This does not affect the measurement — the spike is real and re-derivable from `results/raw/phase0/public_adapter/*/L36_*/records.jsonl`.

### 14.3 2602.13151 is engaged with rather than cited in support

That paper asserts the erasure mechanism for **full-parameter fine-tuning** and proposes **LoRA adapters as the remedy**, on the premise that concentrating an update into an adapter preserves the effective update through quantization. **Our merged-adapter measurements contradict that premise** (100% of per-weight deltas sub-threshold; 1.1–6.2% of codes changed).

Treating it as supporting our framing would have been an error a reviewer would catch immediately. It is now a substantive related-work engagement (§2.4), with our proposed reconciliation — merged versus unmerged, i.e. which tensor sets the scale — marked as our inference and **flagged as unmeasured**.

### 14.4 The 2411.19530 reconciliation is elevated, and labelled

The claim that our result and *Quantized Delta Weight Is Safety Keeper* are the same law at opposite ends of `|Δ|/s`, distinguished by whether `Δ` or `W + Δ` sets the quantization scale, is **a contribution of this paper and an inference of ours**, made by neither cited work. It gets its own subsection (§2.5) and an explicit statement that the unmerged configuration is untested.

**This is also the highest-value single experiment we are not running.** Measuring retention with `Δ` quantized on its own scale would convert the reconciliation from argued to demonstrated. It is listed in Future Work.

---

