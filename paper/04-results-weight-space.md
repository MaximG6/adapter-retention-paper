# 4. Results: weight space

*Draft. All values re-derived from `results/raw/phase0/**/*.jsonl`.*

---

## 4.1 What survives quantization is predicted by `|Δ|/s`, and by nothing else

**The central weight-space result of this paper is a closed form with no fitted
parameters.** For a merged LoRA under group-wise affine quantization, the fraction of
stored integer codes that change is

```
flip rate = mean( min(|δ|/s, 1) )
```

where `δ` is the per-weight adapter delta and `s` the step size of that weight's
quantization group. Nothing in this expression refers to rank, to architecture, to base
model, or to how the adapter was trained.

Measured against nine public adapters spanning **two base models, four ranks (16–128),
both scaling conventions, and four training regimes** (behavioural SFT, DPO,
interpretability probe, safety):

| adapter | flip measured | flip predicted | abs. error | projection identity |
|---|---|---|---|---|
| taboo-moon | 0.01092 | 0.01094 | 0.1% | 0.9934 |
| taboo-smile | 0.01093 | 0.01095 | 0.1% | 0.9924 |
| taboo-snow | 0.01102 | 0.01103 | 0.1% | 0.9931 |
| taboo-gold | 0.01114 | 0.01116 | 0.1% | 0.9924 |
| taboo-rock | 0.01137 | 0.01140 | 0.2% | 0.9919 |
| taboo-ship | 0.01139 | 0.01142 | 0.2% | 0.9926 |
| latentqa | 0.03882 | 0.03912 | 0.8% | 0.9912 |
| responsible-ai-safety (Llama-3.1-8B) | 0.06191 | 0.06335 | 2.3% | 0.9743 |
| ao-v3-dpo-halluc (rsLoRA) | 0.14813 | 0.14949 | 0.9% | 0.9900 |

**Maximum error 2.3%** (Figure 2). Those nine span **1.1 decades** of `|Δ|/s` — 0.0109 to
0.148, a factor of 13.6. The three-decade claim is the *synthetic* sweep below, on a real
Qwen3-8B `q_proj` base at rank 32 with `Δ` constructed rather than trained:

| mean \|Δ\|/s | flip measured | flip predicted | cosine measured | cosine predicted |
|---|---|---|---|---|
| 0.00109 | 0.0015 | 0.0011 | 0.0382 | 0.0402 |
| 0.01087 | 0.0108 | 0.0109 | 0.1263 | 0.1271 |
| 0.10866 | 0.1076 | 0.1086 | 0.4000 | 0.4019 |
| 0.32598 | 0.3153 | 0.3187 | 0.6847 | 0.6961 |
| 1.08659 | 0.6919 | 0.7000 | 0.9400 | 1.0000 |

The tail-shape statistic `τ = mean(Δ²)/mean|Δ|²` is a flat 1.5962 across all magnitudes,
against the Gaussian reference `π/2 = 1.5708` — products of Gaussians are marginally
heavier-tailed, as expected. **This is a property of the synthetic generator and not of
trained adapters, and the paper uses it as if it were both.** Backed out per module from
a real adapter (A.2) `τ` runs **1.82–2.20**, 16% to 38% higher. Since `cosine ∝ √τ`, a
predictor carrying the synthetic constant under-predicts a trained adapter's cosine by 7%
to 17% — the right size to be most of the 10.4% error A.3 reports for cosine, against
2.3% for the flip rate, which carries no such constant. Wherever this paper writes
`τ = 1.5962` it means the synthetic value.

**What this predicts, precisely, is the adapter's own contribution.** The model answers:
given a fixed grid, what fraction of codes does this delta move across a boundary? That is
a parameter-free prediction accurate to 2.3%, and it is the quantity the paper's mechanism
is about — the one that separates what the adapter did from what the quantizer would have
done anyway.

**Merging also moves the grid, and that is a second effect the model does not describe.**
Under `adaptive_scale` both act at once, and measured code-flip rates run 1.5–1.9× the
single-effect prediction; read as a prediction of the combined outcome, the same
per-adapter error is 32–47% (B.1). This is a statement about **scope, not accuracy**: the
model is exact about one mechanism and silent about the other, which is why §3.3 defines
two regimes rather than one. We do not refit — a version tuned to absorb grid movement
would have a free parameter, and having none is the entire claim. What sets the size of
the grid-movement population is not established here.

**Why the model is licensed, which is the substantive claim.** A closed form of this
shape requires that trained deltas carry no information about where a weight sits
within its quantization bin. They do not: the correlation between delta magnitude and
bin position is **below 0.0011** across the **six** adapters this control covers, and a
permutation control that destroys any delta–position association changes the measured flip
rate by **under 1.5%** on the same six (EXP-009). An earlier draft said "all nine"; the
released tool's banner said six, correctly, and the paper drifted away from it. Gradient descent, optimising a loss that knows nothing about the deployment
quantizer, produces updates statistically independent of the quantization grid. **That
independence — not the numerical agreement — is what makes the formula more than a
curve fit.**

**There are three such assumptions and all three are measured.** Independence of bin
position and delta magnitude is the one above. Equation 4 also needs `u` to be
**uniform**, which is a different property and has a structural reason to fail near the
boundary; B.11 measures it at 1.8% of uniform over the whole range our adapters occupy.
And because the flip is two-sided, the argument that the lower tail's excess cancels the
upper tail's deficit needs `δ` to be **sign-balanced** and its sign independent of `u` —
a sign–position association would leave the magnitude correlation above untouched and
break the cancellation exactly. That third assumption came into existence when the
cancellation argument was written, one revision round after Equation 4 was published, and
we did not notice until a reader counted. It was registered as P11 and then measured:
`P(δ<0)` departs from one half by at most 0.000237 and `|r(sign δ, u)|` by at most
0.001060, both at their sampling floor (B.11, EXP-046, EXP-047).

**The first of the three was measured globally where the derivation needs it locally,
and that has now been closed too.** A Pearson correlation over the whole bin is dominated
by the bulk; what §3.5 rests on is the density of `u` in the lowest 1%, conditional on
the delta that has to cross it. Registered as P12 and measured over the same 42
module-instances: binned by decile of `|Δ|/s`, the flip probability at a common
`t = 0.011` runs 0.01085–0.01092, a worst departure from the pooled value of **0.34%**
(B.11, EXP-052). The drift across deciles is monotone — its rank correlation, +0.87,
breached the bound we registered — and it is 0.6% wide end to end; the bound was a
Spearman with no magnitude qualifier, which is scale-free by construction and was the
wrong falsifier to have written.

**The same measurement settles the error budget, which two appendices disagreed about.**
B.11 read the residual sub-uniformity as flat at 0.985 and inferred a near-constant
1.3–1.5% over-prediction for every adapter; B.2 measures 0.1% for each of the taboo six
and 2.3% for `responsible-ai-safety`. **The departure is a function of `|Δ|/s`, not a
constant**: true-flip over `min(t,1)` reads 1.12 at `t = 0.0024`, 0.97–0.98 through the
middle and 0.95 at `t = 0.124`, so an adapter's error is set by where its own
distribution sits. Measured through a separate code path, the two adapters reproduce
B.2's split — 0.973 against 0.977 and 0.999 against 0.999. **That is agreement to two
decimals, not three**: on the (layer, module) instances the two runs share, the two code
paths agree to *five* decimals, but they do not share a layer set — the safety adapter's
four sampled layers are 0/10/21/31 and this run used 0/12/24 — so the pooled ratios are
not the same quantity, and the safety adapter reads a 2.7% over-prediction here against
2.3% in B.2. **The 2.3% is a property of the sampled layers rather than a bound**, the
same layer sensitivity §3.1 reports for the flip rate itself. **The budget is under 0.5%
at the `t` the taboo adapters occupy and about 2.5% at four times that `t`.**

The practical consequence is §4.3: since `|Δ|/s` is the governing quantity and no
adapter card publishes effective magnitude, retention cannot currently be predicted
from published metadata. Appendix A describes a tool that computes it without a GPU.

## 4.2 Applied: published adapters are almost entirely erased at INT4 g128

Nine public LoRA adapters, merged into their base models and quantized at INT4 with
group size 128 (asymmetric, `fixed_scale`). Confidence intervals bootstrapped over
layers; **Figure 3** shows the same values as a forest plot. The table below shows six of
the nine plus the 36-layer run, so the distinct configurations are visible without
repeating three near-identical taboo rows; **B.1 has all nine, under both scale regimes**.

| adapter | base | r | scaling γ | layers | cosine | 95% CI | code-flip | rel. error |
|---|---|---|---|---|---|---|---|---|
| taboo-smile | Qwen3-8B | 32 | 2.00 | 36 | 0.1380 | [0.1355, 0.1404] | 1.22% | 7.29 |
| taboo-smile | Qwen3-8B | 32 | 2.00 | 4 | 0.1374 | [0.1244, 0.1504] | 1.09% | 7.41 |
| taboo-gold | Qwen3-8B | 32 | 2.00 | 4 | 0.1389 | [0.1248, 0.1531] | 1.11% | 7.35 |
| taboo-ship | Qwen3-8B | 32 | 2.00 | 4 | 0.1409 | [0.1279, 0.1539] | 1.14% | 7.23 |
| latentqa | Qwen3-8B | 64 | 2.00 | 4 | 0.2760 | [0.2548, 0.2996] | 3.88% | 3.58 |
| responsible-ai-safety | Llama-3.1-8B | 16 | 2.00 | 4 | 0.3298 | [0.3069, 0.3664] | 6.19% | 2.90 |
| ao-v3-dpo-halluc | Qwen3-8B | 128 | 1.41 (rsLoRA) | 4 | 0.5050 | [0.4755, 0.5386] | 14.81% | 1.74 |

Three facts, in decreasing obviousness.

**Only 1.1%–14.8% of stored integer codes change at all**, and for eight of the nine
adapters the figure is under 6.2%. The overwhelming majority of merged weights quantize to
exactly the value the *base* weight would have quantized to.

**The effective update is not a shrunken version of the intended one — it is
uncorrelated noise several times its size.** Cosine similarity between `Δ` and `Δ_eff`
runs 0.14 to 0.51. Relative error runs **1.74 to 7.42 against an erasure baseline of
1.0**: even the best-retained adapter receives a delta roughly 1.7× its own magnitude,
pointing somewhere it did not ask for. The few weights that move jump a full
quantization step.

**The rank-128 rsLoRA adapter is the best-retained of the nine, and reading its scaling
convention correctly is what makes that visible.** Under `α/r` it would appear to have
γ = 0.125 and a delta 11.3× smaller than it has; we read `use_rslora` from each
adapter's config and verify the resulting delta against peft's own merge (§3.8).

**For the rank-32 adapters, essentially every weight is far below the step size.** The
step-ratio distribution `|Δ|/(s/2)` for `taboo-smile`, pooled: p1 0.0003, p50 0.0156,
p95 0.0638, p99 0.1020, worst-case p99 over records 0.2066. **99.998% of the rank-32
weights are sub-threshold** (`|Δ| < s/2`), the median delta is about 1/128 of a step size,
and the highest 99th percentile of `|Δ|/(s/2)` over any of their records is 0.239 — so
fewer than 1 weight in 100 reaches a quarter of the half-step. This is why the flip rate
is ~1%.

**The tail is in `s`, not in `Δ`.** Those figures put the 99th percentile of `|Δ|/s` at
about 11× its mean, where a Gaussian would give 3.2× and the measured `τ = 1.60` says `Δ`
itself is barely heavier-tailed than Gaussian. The ratio inherits a second distribution:
`s` is a per-group quantity set by that group's range, and step sizes vary by orders of
magnitude across the groups of a layer (B.10's `step med/p1` column). A weight with a
typical delta in an unusually narrow group produces a large `|Δ|/s` without `Δ` having a
heavy tail at all.

*An earlier draft said "100% of weights are sub-threshold; not one reaches a quarter of
the half-step." The first clause rounds 0.99998 and the second is a statement about the
maximum over ~10⁹ weights that we never computed — the quantiles we do have put the
99th percentile at 0.24, which says nothing about the tail beyond it. Both are now stated
at the level they were measured.*

The rsLoRA adapter sits at the other end of the same scale — `mean|Δ|/s = 0.149`
(per-module range 0.083–0.267), hence its 14.8% flip rate and 0.505 cosine. **Both are the same law at different points
of one ratio** (§4.1), which is precisely the structure that reconciles our result with
prior work reporting that compressing delta weights preserves alignment (§2.5).

Had we reported `retention_ratio` as originally specified, this adapter would have
scored **7.48** and read as excellent retention (§3.4).

## 4.3 Rank does not predict weight-space retention in trained adapters — magnitude does

Ordering the `α/r = 2` adapters by rank gives cosine 0.330 (r=16), 0.138/0.139/0.141
(r=32), 0.276 (r=64): **non-monotone, and the lowest-rank adapter retains best.**

On *synthetic* adapters, where magnitude is set by parameterization, the clean scaling
law holds and our registered predictions P1 and P2 were confirmed:

| convention | quantity | fitted exponent | predicted |
|---|---|---|---|
| α = 2r | weight SNR | +0.286 | +0.25 |
| α = 2r | output SNR, subspace input | **−0.182** | **−0.25** |
| α = 2r | output SNR, generic input | +0.286 | tracks weight |
| α = 16 | weight SNR | −0.275 | −0.25 |
| α = 16 | output SNR, subspace input | **−0.744** | **−0.75** |

**Weight-space and output-space fidelity move in opposite directions with rank under
`α = 2r`** — weight SNR climbs 0.255 → 0.688 from r=4 to r=128 while subspace output
SNR *falls* 7.34 → 3.94. This was registered before measurement.

The synthetic law does not transfer to trained adapters because **optimization, not
parameterization, sets effective magnitude** — and effective magnitude is not a
quantity any adapter card publishes. This is the practical gap our tool addresses
(Appendix A).

**The scaling convention is not a free choice.** Five of six adapters use `α/r = 2`,
the convention under which retention is *most* favourable, and it does not rescue them.

## 4.4 Layer-output fidelity greatly exceeds weight-space fidelity

Isolating rank as the only variable (SVD-truncating one adapter's delta and rescaling
to fixed Frobenius norm), the amplification law of §3.6 holds (**Figure 4**):

| module | d_in | r | conc(Δ) | conc(E) | amplification | √(d_in/r) | ratio | generic-input |
|---|---|---|---|---|---|---|---|---|
| q_proj | 4096 | 4 | 1024.0 | 1.201 | 29.20 | 32.00 | 0.913 | 0.997 |
| q_proj | 4096 | 8 | 513.2 | 1.088 | 21.71 | 22.63 | 0.960 | 0.995 |
| q_proj | 4096 | 16 | 260.2 | 1.040 | 15.82 | 16.00 | 0.989 | 0.997 |
| q_proj | 4096 | 32 | 129.8 | 1.012 | 11.33 | 11.31 | **1.001** | 0.998 |
| down_proj | 12288 | 4 | 3079.8 | 1.228 | 50.09 | 55.43 | 0.904 | 0.993 |
| down_proj | 12288 | 32 | 382.9 | 1.020 | 19.38 | 19.60 | 0.989 | 0.991 |

Fitted exponents −0.4569 (`q_proj`), −0.4554 (`gate_proj`), −0.4574 (`down_proj`)
against a predicted −0.5. The `√3` prediction *between* module families is confirmed:
at r=4, amplification 29.20 (`q_proj`) vs 50.09 (`down_proj`), ratio **1.715** against
`√3 = 1.732`.

**Generic-input amplification is 0.991–1.005 at every rank and module.** There is no
dimensional averaging: the effect exists only on inputs inside the adapter's active
subspace. An earlier version of our own design predicted `1/√d_in` suppression for
generic inputs; the measured value is exactly 1.00 (`METHODOLOGY.md`).

The residual deviation from `√(d_in/r)` is fully explained by error anisotropy:
`conc(E)` falls from 1.20–1.25 at r=4 to 1.01–1.03 at r=32, matching `1 + c/r` with
`c ≈ 0.87`, which is what `Var(E) ∝ s|δ|` predicts.

Applied to the nine published adapters with an orthonormal probe:

| adapter | r | weight SNR | **output SNR** | amplification | √(d_in/r) | ratio | conc(E) |
|---|---|---|---|---|---|---|---|
| taboo-moon | 32 | 0.1334 | 1.6200 | 12.39 | 12.50 | 0.992 | 1.017 |
| taboo-snow | 32 | 0.1342 | 1.6254 | 12.37 | 12.50 | 0.990 | 1.017 |
| taboo-smile | 32 | 0.1341 | 1.6286 | 12.38 | 12.50 | 0.991 | 1.019 |
| taboo-gold | 32 | 0.1342 | 1.6299 | 12.40 | 12.50 | 0.992 | 1.016 |
| taboo-ship | 32 | 0.1366 | 1.6566 | 12.38 | 12.50 | 0.990 | 1.019 |
| taboo-rock | 32 | 0.1375 | 1.6728 | 12.38 | 12.50 | 0.990 | 1.018 |
| latentqa | 64 | 0.2920 | 2.5250 | 8.80 | 8.84 | 0.996 | 1.012 |
| ao-v3-dpo-halluc | 128 | 0.6164 | 3.7571 | 6.22 | 6.25 | 0.995 | 1.007 |
| responsible-ai-safety | 16 | 0.3854 | 5.9995 | 16.54 | 17.99 | 0.919 | 1.414 |

**A weight-space SNR of 0.13 corresponds to an output SNR of 1.63 — signal above
noise.** **Every adapter measured has output-space signal exceeding noise**, with the
weakest at 1.62 and the strongest at 6.00, while their weight-space SNRs run 0.13–0.62.
The amplification law accounts for the gap to within 1% for eight of the nine; the
lowest-rank adapter deviates most (ratio 0.919) and has by far the largest `conc(E)`
at 1.414, consistent with `1 + c/r` at r = 16.

**A warning about how to read this table, because its arrangement invites an inference we
spend §5.4 refuting.** The six adapters at the bottom of the output-SNR range are also
the six we measure behaviourally, and they are the ones that degrade at coarser
precision. It is natural to read 1.62 as "close to the noise floor, hence fragile", and
to treat the column as a fragility ranking. **We do not support that reading, and our own
data contradicts it.**

Those six adapters agree on output SNR to within **3.3%** — 1.6200 to 1.6728 — and their
behavioural retention at INT3 spans **28.7% to 86.4%** on the pre-registered instrument,
28.4%–84.4% floor-corrected (B.7). Within that near-constant band,
`ship` at 1.6566 is the *second-highest* predictor value and has the *worst* retention,
while `moon` at 1.6200 is the *lowest* and has the *best*. A quantity that is constant to
3% cannot explain an outcome that varies threefold, and where the differences resolve
statistically the ordering runs backwards (§5.4, PG-1 and PG-2).

**What this table does establish** is that the amplification law is quantitatively
correct, and therefore that weight-space cosine systematically understates layer-output
fidelity — the mechanism by which behaviour can survive weights that look destroyed.
**What it does not establish** is a usable ranking of which adapter will survive.

## 4.5 Secondary structure

**Depth.** Retention rises weakly and non-monotonically with depth: first-quartile
cosine 0.1322 → last-quartile 0.1446, **+9.4%** over 36 layers. A 4-layer sample of the
same adapter gave 0.119 → 0.154 (+29%), monotone — a trend three times too large with
the wrong shape (`METHODOLOGY.md`). The full profile also shows a **bit-flip spike at layers 1–3
(2.5–2.7% against ~1.0% elsewhere)** that is invisible at 4-layer resolution.

### 4.5.1 The spike is a low-activation weight structure, not an activation outlier

The full 36-layer profile is **Figure 11**. The spike is driven by a heavy small-step
tail — `gate_proj` step median/p1 of 83.5 (layer 1), 44.6 (layer 2), 145.1 (layer 3),
against 1.4 and 1.6 in layers 0 and 18 — and small `s` raises `|Δ|/s`, hence the flips.

Since quantization groups run along the input dimension, each group maps to a block of
128 input channels, and we can ask directly whether the narrow-range groups coincide
with high-activation channels, as the activation-outlier literature would suggest. **They
coincide with the opposite.** Relative to each module's mean, the narrowest 1% of groups
sit at channels with activation **0.17 / 0.19 / 0.15** in layers 1/2/3, while the widest
1% sit at 1.12 / 1.58 / 1.05. Spearman correlation between log step size and block
activation is **+0.244 / +0.275 / +0.156** in the spike layers and **+0.033 / +0.012** in
the controls; the activation profile is stable across a split-half of the calibration
text at r ≥ 0.99.

The association is therefore real, confined to the layers with the spike, and **inverted
relative to the massive-activation phenomenon**. §2.2 discusses why this makes the spike
a distinct phenomenon rather than an instance of the known one.

**The mechanism is not established, and we are not offering one.** We have shown what the
spike is *associated* with — narrow-range weight groups at quiet input channels — and
what it is *not*: the activation-outlier phenomenon, from which it is inverted. Neither
is an account of *why* early-layer `gate_proj` and `up_proj` groups have that range
structure, and nothing here isolates a cause. An earlier draft of this section recorded
the spike as "a known phenomenon"; that was withdrawn on measurement, and we have not
replaced it with a second explanation.

What we offer instead is a falsifiable prediction, **FW-1**: outlier-aware quantizers
select what to protect by *high* activation (AWQ, SmoothQuant) or by large per-weight
quantization error (SpQR). The groups driving this spike qualify under neither rule, so
those methods should not protect them, and adapter retention in layers 1–3 should not
improve under them. That is testable with existing tools and we did not test it.

**Module type.** Pooled over six adapters, `gate_proj` retains most (cosine 0.2840) and
`down_proj` least (0.2097); full ordering in Appendix B.5. The ordering broadly follows
median `|Δ|/s`, so module differences are **mostly** a magnitude effect rather than an
architectural one — but not entirely, and B.5's own table says so: `k_proj` has the
second-highest flip rate and the fourth-lowest cosine while `up_proj` has the
fourth-highest flip and the second-highest cosine, which under `cosine = √(τ · |Δ|/s)`
requires `k_proj`'s tail-shape statistic to sit 17.5% below `up_proj`'s. Magnitude sets
the scale and tail shape reorders within it, which is the same τ that A.2 measures at
1.82–2.20 across modules.

**Quantization convention.** Paired on 252 identical (adapter, layer, module) cells
(B.3): asymmetric 0.2161, `symmetric_gptq` 0.2065, `symmetric_awq` 0.1980 — a maximum
deviation of 8.4%. Whether an adapter survives depends slightly on which toolchain
produced the checkpoint. Pooling unpaired records inverts this ordering (`METHODOLOGY.md` M.4).

**Scale regime.** Pooled over the nine adapters (B.4): `fixed_scale` cosine 0.2161 /
flips 0.0351; `adaptive_scale` cosine 0.2151 / code flips 0.0572 / **value changes
0.8552**, with `scale_shift_fraction` 1.0000 pooled and never below 0.9998 in any single
module. Merging moves the grid for essentially every group in the model, and almost all of
the deployment-realistic "change" is the grid moving rather than the adapter arriving:
the grid-shift fraction — weights whose value changes under `adaptive_scale` but not under
`fixed_scale` — is 0.8202. Cosine is nearly identical across regimes, so what the adapter
actually transmits does not depend on this choice; what depends on it is how much of the
checkpoint differs.

## 4.6 Summary of §4, stated as weight-space claims

Retention under merge-then-quantize is governed by a single ratio, `|Δ|/s`, and a
parameter-free channel model converts it into a code-flip rate within 2.3% across every
adapter we tested — a result licensed by the measured independence of trained deltas
from the quantization grid. Applied to published adapters at INT4 g128, that model
describes **near-total weight-space erasure**: 1.1%–14.8% of stored codes change, under
6.2% for eight of the nine, and the effective update is uncorrelated noise 1.7–7.4× the
intended update's size.

**None of this is a statement about behaviour.** The same measurements predict
layer-output SNR of 1.6–6.0 on inputs inside the adapter's subspace — 6.2–16.5× the
weight-space fidelity, the factor rising as rank falls. Whether the behaviour survives
is §5.

---

# 5. Results: behaviour

*All values re-derived from `results/raw/phase1/**/records.jsonl`: 6 adapters x 4
precisions x 32 prompts x 2 conditions (aligned and base) = 1536 records.*

**Design.** Six Taboo adapters (`smile`, `ship`, `gold`, `snow`, `moon`, `rock`), all
rank 32, `α/r = 2`, on Qwen3-8B, differing only in the secret word. Four precisions:
BF16, INT4 g128, INT4 per-channel, INT3 g128, applied as weight-space
quantize–dequantize over 252 projections. 32 prompts per condition (24 hint
paraphrases + 8 adversarial), greedy decoding. Primary instrument is elicitation
(§3.7), validated through the gate before any prediction was registered on it.

## 5.1 No detectable loss at INT4 g128; a mean that degrades and adapters that do not

Retention is each adapter's elicitation score as a fraction of its own BF16 score.
**Figure 5** plots the full dose-response, per adapter and pooled.

<!-- GENERATED: table2 -- analysis/appendix_tables.py --write. Do not edit by hand.
     This table and Appendix B.6 are the same call; they disagreed in the last digit of
     three intervals when they were two calls. -->
| precision | mean retention | 95% CI over adapters | 24 hint only | adapters below 50% |
|---|---|---|---|---|
| INT4 g128 | **99.2%** | [90.7%, 107.6%] | 104.9% | 0/6 |
| INT4 per-channel | **77.2%** | [69.0%, 86.0%] | 80.2% | 0/6 |
| INT3 g128 | **57.8%** | [41.7%, 74.3%] | 62.1% | 2/6 |
<!-- END GENERATED: table2 -->

Guesser argmax accuracy, pooled: 159/192 (BF16) → 157/192 → 128/192 → 98/192.

**The elicitation metric has a floor and it is reported rather than assumed away.** The
guesser discriminates over 20 candidates and each adapter's score is normalised against
its own BF16 value, so neither end of the scale is anchored at chance. The same instrument
scores the *base* model at 0.0039–0.1642 — small, but 40x different between `ship` and
`snow`, because the guesser has a prior. Floor-correcting against the base model at the
same precision, `(aligned − base) / (aligned_BF16 − base_BF16)`, gives **99.0%, 76.5% and
56.0%** against the 99.2%, 77.2% and 57.8% below: under 2 points at every precision.

**That is a statement about means, and this paper's claims are not.** An earlier draft
followed it with "no claim in this paper turns on the difference", which was checked at
the mean and is false per adapter. It moves two things at INT3. The split below half goes
from two adapters to three, because `smile` sits at 51.3% uncorrected and 49.3%
corrected; and the count above 80% goes from two to one, because `snow` falls from 81.5%
to 77.7%. The span moves from 28.7%–86.4% to 28.4%–84.4%. **B.7** gives all six adapters
under all three metric variants, which is the level at which those claims can be checked,
and every site quoting the split or the span names the variant it is quoted under. What
does *not* move: PG-1's ratio of outcome to predictor variation, which stays between 7.3×
and 30.5× across all nine variant × precision cells, and PG-2, which is identical under
floor correction — same pairs, same counts, same directions (B.12).

These conditions are quantized the way a toolchain would — the merged model on its own
recomputed grid, i.e. `adaptive_scale` (§3.3) — so the weight-space number to pair with a
behavioural one is the deployment-regime number for these same six adapters.
**At INT4 g128, 85.5% of their stored values differ from the base model's and 2.1% of the
integer codes do (98.9% of codes unchanged under `fixed_scale`, which isolates the adapter
but is not what was run here), and no loss of elicitation capability is
detectable.** Retention is 99.2% with an enumerated 95% interval of [90.7%, 107.6%]. That
interval spans parity, so the honest statement is a **non-detection with a bound**: the
instrument cannot separate the quantized model from the unquantized one, and it excludes
losses greater than about 9%. It is not a measurement of equality, and the point estimate
should not be read as one — retention above 100% (rock 116.2%, ship 103.2%) is noise,
since a quantized model cannot exceed its own BF16 baseline.

**The constraint side of the same battery does move, and it tightens — consistently in
direction, and not resolvably in size.** On the 8 adversarial prompts built to make the
model say the word, the leak rate falls from **16.7% at BF16 to 8.3%** at INT4 g128:
`gold` 12.5%→0, `moon` 37.5→25.0, `rock` 12.5→0, `ship` 0→0, `smile` 25.0→12.5,
`snow` 12.5→12.5 — **four leak one fewer prompt, two tie, none leaks more**. The paired
difference is **+8.3 points**, enumerated 95% interval **[+4.2, +12.5]**. Every cell is a
count out of 8, so the resolution is 12.5 points and a per-adapter difference is one
prompt.

| vs. BF16 | mean | 95% CI | sign-flip *p* |
|---|---|---|---|
| INT4 g128 | +8.3 | [+4.2, +12.5] | 0.125 |
| INT4 per-channel | +8.3 | [+2.1, +16.7] | 0.250 |
| INT3 g128 | +10.4 | [+0.0, +20.8] | 0.250 |

**The exact test this design supports is a sign-flip permutation** over the 2⁶ = 64
assignments of sign to the six paired differences. At INT4 g128 the observed sum is the
most extreme of the 64: 4/64 one-sided, **p = 0.125** two-sided. Wilcoxon on the four
non-tied pairs gives the same 0.125, as does a sign test. **Nothing survives a Holm
correction across the three precisions**, and we make no significance claim.

**An earlier draft did, and the number it used was not a test.** It reported an
enumerated-*bootstrap* p of 0.003 at INT4 g128. Because every per-adapter difference there
is non-negative, a resample mean can only reach zero if all six draws are the two tied
adapters, so that p is exactly `2 × (2/6)⁶` — **it counts how many pairs tie, not how
large the effect is**. The same arithmetic sets the interval's lower edge, since the 2.5th
percentile falls at two non-zero draws of six; *the interval excluding zero and that p are
one fact about ties, reported twice*. The interval above is a resampling summary and
nothing is inferred from its lower edge (§3.11).

**This contrast was run in response to review and was not pre-registered, and neither was
the family of three.** The adversarial set was built as a constraint probe and its design
is registered (§3.7); running *this* comparison, at these three precisions, was a choice
made after seeing that the abstract's two-sided claim rested on one side's evidence. We
report it because omitting it was the worse error — the paper ran the contrast only at
INT3 for four drafts, where `snow` moves the other way, and read that omission as evidence
of no change — but the selection of precisions is ours and the p above is uncorrected for
it.

This is the paper's headline: near-total weight-space erasure with **no detectable loss of
elicitation capability and a measurably tightened constraint**, measured end to end on one
matched population rather than paired across two. Weight-space values
are the 4-layer runs for all six (Appendix B.1); behavioural values are the full Phase 1
grid.

**The mean at INT3 conceals a split.** On the pre-registered instrument two of six
adapters fall below half their BF16 score (ship 28.7%, gold 41.3%) while two remain above
80% (moon 86.4%, snow 81.5%); floor-corrected it is three and one, as above. The
57.8% mean describes no adapter in the population, and a reader who takes it as a
description of what happens at INT3 will be wrong in both directions. Whatever governs
the split is not rank, scaling, base model or recipe, which are matched across all six;
§5.3 is what we can say about it, which is that our weight-space predictor does not
identify it.

**Degradation is a trend, and every step of it resolves.** Because the same six adapters
are measured at every precision, the contrasts are paired, and paired differences are the
correct test — overlapping marginal intervals do not imply an unresolvable difference:

| contrast | mean paired difference | exact 95% CI | separates |
|---|---|---|---|
| INT4 g128 − INT4 per-channel | 22.0% | [10.8%, 31.6%] | yes |
| INT4 g128 − INT3 g128 | 41.4% | [23.3%, 59.3%] | yes |
| INT4 per-channel − INT3 g128 | 19.4% | [5.4%, 34.5%] | yes |

All three separate. The **trend** is what the population supports, not a claim that any
individual adapter degrades monotonically: only **4 of 6** do, `snow` rising from 93.5%
at INT4 g128 to 96.8% at per-channel and `moon` from 78.1% to 86.4% at INT3.

Per adapter:

| word | BF16 (raw) | INT4 g128 | INT4 per-ch. | INT3 g128 |
|---|---|---|---|---|
| gold | 0.8280 | 81.3% | 62.4% | 41.3% |
| moon | 0.8250 | 100.2% | 78.1% | 86.4% |
| rock | 0.8603 | 116.2% | 77.5% | 57.7% |
| ship | 1.0327 | 103.2% | 79.8% | 28.7% |
| smile | 0.7178 | 100.8% | 68.5% | 51.3% |
| snow | 1.0030 | 93.5% | 96.8% | 81.5% |

## 5.2 The degradation is not distribution flattening

**Figure 7** plots entropy against retention. Mean token entropy across the aligned
conditions is flat — 1.4069 (BF16), 1.3984
(INT4 g128), 1.4998 (INT4 per-channel), 1.3480 (INT3 g128) — while elicitation falls
by nearly half. A quantized model whose output distribution merely flattened would
show entropy rising with degradation. It does not, so the effect is a change in what
the model produces, not in how sharply it samples.

## 5.3 The dissociation is benign, and the reference class decides its sign

Which side of the taboo behaviour breaks first is the question the two-sided
instrumentation exists to answer. **The constraint holds while the capability
degrades** (**Figure 6**).

| precision | knowledge, base | knowledge, aligned | ratio | Cliff's d |
|---|---|---|---|---|
| BF16 | 0.3634 | 0.0757 | 0.208 | −0.778 |
| INT4 g128 | 0.3583 | 0.0634 | 0.177 | −0.778 |
| INT4 per-channel | 0.3272 | 0.0730 | 0.223 | −0.833 |
| INT3 g128 | 0.2803 | 0.0756 | 0.270 | −0.556 |

The suppression ratio stays in **0.18–0.27** while elicitation halves. Capability
degrades; the constraint does not collapse.

**The effect size is not uniform across precisions, and the structure is worth reporting
rather than averaging away.** Cliff's *d* is −0.778, −0.778, −0.833 and **−0.556** at
BF16, INT4 g128, INT4 per-channel and INT3 g128 respectively. Three of the four sit near
−0.8; the INT3 value is **28.5%** weaker, and the suppression ratio is correspondingly at
its loosest there (0.270 against 0.208 at BF16, a rise of **29.8%**).

**Both of those numbers are driven by the denominator, and saying so is the point.**
The aligned model's own knowledge score shows **no trend across precision** — 0.0757,
0.0634, 0.0730, 0.0756, with all four enumerated intervals overlapping (B.9). The base
model's falls monotonically, **0.3634 → 0.2803**, a 22.9% drop. So the ratio rises because
the base degrades, not because the trained constraint gives way; and the ratio's own
non-monotonicity (0.208, 0.177, 0.223, 0.270) comes from the numerator, since the
denominator is monotone. **The honest statement is that we cannot detect a change in the
constraint's absolute strength, not that there is none** — an earlier draft of this
paragraph called the numerator "flat at 0.0757 and 0.0756", which is the first and last
elements of a four-element series whose largest step is 16.2%. Cliff's *d* is a rank
statistic on the same two distributions and inherits the same shift, which is why it moves
by a near-identical 28.5%.

**"The constraint weakens by about 30% at INT3" would therefore be the wrong reading, and
an earlier draft of this section made it.** What weakens by ~30% is the *separation
between aligned and base*, and it weakens because base capability fell toward the aligned
model's floor. We keep the within-precision ratio as the headline because the alternative
reference class is worse (below), but the ratio's movement is a fact about the base model
under quantization. The dissociation claim that survives is comparative: capability falls
far faster than the constraint does, and at a precision coarser than we tested the two
could cross — nothing here excludes it.

**So the dissociation holds at INT3 — ratio 0.270 against a capability retention of
57.8% — but it is measurably less sharp there than at finer precisions.** Both sides move
at the coarsest setting; capability moves far more.

**That attenuation is, weakly, in the direction the withdrawn P7 predicted.** P7 held
that the constraint would fail before the capability; the measured result is emphatically
the opposite in magnitude, and we do not retract the withdrawal. But the *only* precision
where the constraint measurably loosens is the coarsest one, which is where P7 would have
expected it to give way first. **We flag this as an observation, not a finding.** With six
adapters at one precision we have no power to distinguish a real attenuation from noise
in a rank statistic, and reading it as vindication of a prediction we withdrew on evidence
would be exactly the post-hoc move the withdrawal was meant to avoid. It is recorded
because burying an inconvenient direction is worse than reporting an underpowered one.

**The overall failure mode remains benign** — the model becomes less able to hint, not
more likely to leak. It is the opposite of the alarming case (knowledge retained,
refusals lost), and the opposite of what we predicted before withdrawing that prediction
on evidence (`METHODOLOGY.md` M.2).

**Dividing out the quantizer's effect on the base is necessary, and the result inverts
without it.** The base model's own knowledge score falls 0.3634 → 0.2803 under
quantization. Comparing aligned-quantized against base-**BF16** would show the
suppression apparently weakening; comparing against base-**quantized at the same
precision** shows it flat. Quantization moves the reference class, and a within-
precision comparison is the only valid one.

## 5.4 The predictive gap: weight-space measurement does not predict behavioural outcomes

We call this result the **predictive gap**, and refer to it by that name throughout.
It is supported by three demonstrations, labelled **PG-1**, **PG-2** and **PG-3**, which
differ in kind — a variance argument, a sign argument, and an existence argument.

| | argument | claim | evidence |
|---|---|---|---|
| **PG-1** | variance | predictor cannot explain outcome spread | predictor CV 0.0128 vs outcome CV up to 0.390 |
| **PG-2** | sign | where differences resolve, ordering is usually **backwards** | 6 of 7 resolvable pairs invert |
| **PG-3** | existence | largest predictor value, no measurable behaviour | SNR 6.00, no gate-clearing refusal |

**PG-2 and PG-3 are positive-signed: they report an inversion and an
absence-in-the-presence-of-signal, rather than a failure to detect a correlation.** This
matters for how the finding should be read. PG-1 alone is a variance argument and is
open to the objection that with a near-constant predictor at n=6 no correlation *could*
have been detected — that objection is correct, which is why we do not report a
correlation coefficient.

PG-2 does not answer that objection by being better powered; it answers it by being a
different kind of claim. It says that where differences are large enough to resolve, the
ordering usually runs the wrong way.

**We do not attach a p-value to this, and §5.1 does attach one to the leak contrast; the
difference is that there a null exists.** A paired difference over six adapters has an
exact randomization distribution under sign flipping. A count of which pairs separate does
not, for the reasons below. **An earlier draft attached one anyway and was wrong to.** That
draft
observed that 4-of-4 and 6-or-more-of-7 both give 0.0625 under a binomial null and
concluded the evidential strength was preserved. Too quick, in two ways. The two are not
estimates of the same thing: the first was *4 of 4 at INT3*, the second *6 of 7 pooled
across three precisions*. And `2⁻⁷` is the wrong null either way, because pairwise
orderings among six adapters are constrained by transitivity, and the same six adapters
recur at all three precisions, so the seven outcomes are neither independent nor
exchangeable.

**PG-2 is a descriptive claim and it does not need a p-value to do its work.** Pairs
become resolvable precisely because their intervals separate, which is a selection
effect. It is **evidence of inversion rather than of absence of correlation** — not a
significance claim, in either the old form or the new one.

PG-3 is independent of both, and unlike PG-1 and PG-2 it **is partly a power question**:
the base model already refuses at ceiling, so there is no headroom for the adapter to add
refusal in, and a ceiling effect is a power problem. What survives it is the existence
claim — a single adapter where the predictor is maximal and the target behaviour is
unmeasurable — which no amount of additional sampling within the taboo family would alter.

### PG-1 (variance) — the predictor is constant while outcomes span 3×

**Figure 8** shows the predictor against the outcome, with the resolvable pairs marked.

The six taboo adapters are matched on rank, scaling, base model, training recipe —
**and on output SNR to within 3.3%** (1.6200 to 1.6728). Their INT3 behavioural
retention spans **28.7% to 86.4%** on the pre-registered instrument, 28.4%–84.4%
floor-corrected.

| quantity | coefficient of variation |
|---|---|
| predictor (output SNR) | 0.0128 |
| outcome, INT4 g128 | 0.116 (**9×** predictor) |
| outcome, INT4 per-channel | 0.152 (**12×**) |
| outcome, INT3 g128 | 0.390 (**30×**) |

**The outcome varies 9× to 30× more than the predictor**, and **6.5× to 27.5×** once
the outcome's own measurement error is netted out of its spread (B.7). The raw comparison
puts a noisy outcome against a deterministic predictor and overstates the gap; the
corrected figures are the ones to quote, and PG-1's conclusion is unchanged by the
correction. The predictor is a Phase 0
quantity and does not depend on the choice of behavioural instrument at all; the outcome
does, and across all three metric variants of B.7 the raw ratio runs **7.3× to 30.5×** —
the table above is the pre-registered column. We do not report a
correlation coefficient here: correlating against a near-constant at n=6 is
meaningless, and the Spearman value flips sign across precisions (+0.600, −0.257,
−0.657). We decline to plot it (§ Figures).

### PG-2 (sign) — among resolvable pairs, the ordering mostly inverts

Greedy decoding makes seeds inert, so the nuisance axis is which prompts were drawn. The
sampling unit is the **intent**, not the prompt: E.1's hint battery is 8 intents × 3
paraphrases, so a prompt-level bootstrap counts clustered draws as independent ones. The
measured ICC is 0.175–0.303, giving **23–26 effective units of 32** — not the "roughly
16" this paper asserted before measuring it (B.12) — and a prompt-level standard error
too small by 11–18%. Intervals below resample
intents, stratified by prompt kind and paired across precisions (§3.11); the per-adapter
intervals are **Figure 9**:

| precision | between-word spread | mean within-adapter CI width | ratio | non-overlapping pairs |
|---|---|---|---|---|
| INT4 g128 | 34.9% | 29.7% | 1.17 | **1 of 15** |
| INT4 per-channel | 34.4% | 37.1% | 0.93 | 2 of 15 |
| INT3 g128 | 57.8% | 37.7% | **1.53** | **4 of 15** |

At INT3 four pairs separate cleanly (`gold`–`moon`, `gold`–`snow`, `moon`–`ship`,
`ship`–`snow`), and the same four separate under the published prompt-level estimator.

**Six of the seven separating pairs run against output SNR.** `ship` has the
second-highest output SNR in the family (1.6566) and the **worst** INT3 retention
(28.7%); `moon` has the **lowest** SNR (1.6200) and the **best** retention (86.4%).

**The seventh runs with it,** and it is the sole INT4 g128 pair, `gold`–`rock`. `rock`'s
point estimate there is 116.2% — above its own BF16 baseline, which a quantized model
cannot exceed, so this is a separation between one real value and one the instrument
cannot deliver. We count it against ourselves rather than excluding it, which is why this
section no longer says *every*.

**Which correction moved which count.** Two things change at once between the published
estimator and this one, and they push in opposite directions, so reporting only the net
change would attribute it to whichever was named:

| estimator | INT4 g128 | INT4 per-ch. | INT3 |
|---|---|---|---|
| A: prompts, unpaired (as published) | 0 | 1 | 4 |
| B: prompts, paired | 1 | 2 | 6 |
| C: intent clusters, paired (used here) | **1** | **2** | **4** |

Pairing narrows, because both conditions run the identical prompts and the shared
prompt-difficulty variance cancels. **Clustering was described here as widening, and
measured it does not, reliably** — at the measured ICC it is close to a wash, and the
two pairs it costs are both at INT3 and both involve `smile` (B.12). Pairing does the
work. At INT3 the net returns to the published 4, on the same four pairs; at INT4
pairing dominates and one pair appears that the published estimator called noise.

### PG-3 (existence) — the largest measured output SNR has no measurable target behaviour

Among the adapters we attempted to measure behaviourally, `responsible-ai-safety` has by
far the largest measured output SNR (5.9995, against 1.62–1.67 for the taboo six; B.13).
It also
fails to produce any gate-clearing target behaviour at all (§6). The weight-space
prediction is maximal exactly where the behavioural measurement is absent.

**Two qualifications, because this is the weakest of the three legs.** It is *not* the
highest adapter on every weight-space axis: `ao-v3-dpo-halluc` has a higher cosine (0.5050
against 0.3298) and a higher code-flip rate (14.81% against 6.19%), and we have no
behavioural battery for it, so the superlative holds only for output SNR and only within
the adapters we tried to measure. And the reason there is no gate-clearing behaviour is in
part a ceiling: the base `Llama-3.1-8B-Instruct` already refuses 16/16 harmful prompts, so
there is no headroom for the adapter to add refusal in (§6). **A ceiling effect is a power
problem**, and we do not claim otherwise. What survives is the existence claim: a
weight-space measurement can be maximal for a behaviour no instrument can find, and a
practitioner reading only the weight numbers would not know which case they were in.

### Consequence

Whatever drives behavioural fragility is largely orthogonal to the weight-space
quantities we measure — including the ones our own tool reports (Appendix A). A
practitioner comparing two adapters of comparable rank and magnitude receives an answer
that carries no information about which will survive quantization, and we say so in the
tool's own output rather than only in this paper.

**The predictive gap does not weaken §4.** The channel model predicts what it claims to
predict — stored weights — to within 2.3%. PG-1 to PG-3 establish that weight-space
retention, however precisely measured, is not a proxy for behavioural retention. Both
statements are true, and holding them together is the point of measuring at two levels.

**Nor does it undo §4.4, and the distinction is range restriction.** The amplification law
and the predictive gap are about different questions, which is easy to miss because both
are denominated in output SNR. PG-1 to PG-3 show that output SNR does not *discriminate*
within a population whose predictor spans 3.3%. They say nothing about whether it explains
the *absolute level*, and the level is what §4.4 measures: why every adapter's subspace
signal exceeds its noise at all when the weight-space SNR is 0.13. A quantity can set
the level of an outcome and still be useless for ranking cases that differ in it by 3.3% —
that is range restriction, not a contradiction. Whether output SNR discriminates across
*dissimilar* adapters is untested (§8.2), and the released tool says so in its own output.

## 5.5 Prior puzzles resolved

Two anomalies from a single-adapter pilot dissolved when the population grew to six.
The graded constraint's INT3-vs-BF16 ratio is **1.05× on average with only 1/6
adapters increasing**; the pilot's 4.23× (`smile`) was an outlier against 0.18–0.69×
for the rest. The knowledge-probe puzzle is §5.3.
