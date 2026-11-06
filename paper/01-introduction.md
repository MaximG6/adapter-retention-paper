# 1. Introduction

*Draft. Written last, against the findings as they stand rather than the hypothesis the
project began with — see `METHODOLOGY.md` M.4. The version of this section we would have written first
would have argued that quantized alignment fine-tunes are behaviourally the base model.
The measurements do not support that, and this section says what they do support.*

---

## The contrast

Take a published LoRA fine-tune, merge it into its base model, and quantize the result
to 4 bits with group size 128 — the ordinary deployment path. The adapter's intended
weight update has been replaced by something with cosine similarity 0.14 to it, and
7.5 times its magnitude, pointing in a direction the adapter never requested. Not one of
that adapter's weight deltas reaches even a quarter of a quantization half-step; the
median falls below the step size by a factor of 128.

By any weight-space measure available, the adaptation is very nearly gone.

**How much of the model looks changed depends on which tensor sets the quantization grid
and on which count you take** (§3.3), and the readings run from 1.1% to 85.5%. On the
deployment path a single adapter shows **85.5% of stored *values*** changed against
**2.1% of the integer *codes***, a factor of 41 for that adapter and 15.0× pooled over
the nine; holding the grid fixed puts its value changes at 1.1%. Those are two different
contrasts — one between regimes, one between metrics — and an earlier draft quoted one
adapter's pair with the pooled ratio, which is how 41 came to be printed as 15.

**This is why the headline of this paper is a cosine and not a count.** The cosine between
the intended and delivered update moves from 0.1390 to 0.1379 between the regimes, a 0.8%
difference. The question the paper asks — did the trained update survive? — has a
regime-independent answer; the question the counts answer — how much of this checkpoint
differs from the base model? — does not.

**On the same models, quantized the same way, elicitation capability is
indistinguishable.** Retention is 99.2%, and its enumerated 95% interval — [90.7%,
107.6%] — spans parity. The instrument cannot separate the quantized model from the
unquantized one. What it does establish is a bound: **losses beyond about 9% are
excluded.**

**The other half of behaviour does move.** §3.7 defines the battery as two-sided —
capability and constraint — and the constraint **tightens**: the same six adapters leak
the suppressed word on 16.7% of adversarial prompts at BF16 and 8.3% after quantization,
a paired **+8.3 points** with an enumerated 95% interval of **[+4.2, +12.5]** that
excludes zero (§5.1). So "no behavioural change" is not what was measured and is not what
this paper claims: one side is unchanged and the other moves, in the direction of
disclosing less.

**Figure 1** puts the weight-space and capability panels side by side. This paper is
about that contrast: how both statements can be true at once, what governs each of them,
and what follows for anyone shipping a quantized fine-tune.

## Why this matters, and why the obvious worry is the wrong one

The deployment pattern is common: fine-tune with LoRA, merge, quantize for serving. If a
rank-16 adapter produces a weight delta smaller than the 4-bit step size, the merged
model is numerically indistinguishable from the base model — and an "aligned quantized
model" could in principle be behaviourally the base model with the alignment silently
absent. That concern is what motivated this work, and it is asserted in adjacent
literature: work on quantization-robust unlearning states that ordinary fine-tuning
updates are "too small to survive 4-bit quantization" and that quantization can cause
models to revert to pre-unlearning behaviour (§2.4).

**We find the weight-space half of that concern to be correct and the behavioural half
not to follow.** The stored weights really are almost unchanged. The behaviour is not.
The reason is not that the weight measurement is wrong — we validate our quantizer
bit-exact against `gptqmodel` on 36 of 36 configurations — but that weight-space
fidelity is the wrong quantity for predicting behaviour, in a way that is derivable
rather than accidental.

## The governing quantity

A single ratio explains both halves: `|Δ|/s`, the adapter's per-weight magnitude against
the step size of its quantization group.

**At the level of stored weights**, `|Δ|/s` determines everything. Treating quantization
as a stochastic rounding channel gives a code-flip rate of `mean(min(|Δ|/s, 1))`, and
this closed form — with **no fitted parameters** — predicts the measured flip rate of
every adapter we test to within **2.3%**, across two base models, ranks from 16 to 128,
both scaling conventions, and four different training regimes. What licenses the form is
a measured property rather than goodness of fit: trained deltas are statistically
independent of where a weight sits within its quantization bin (correlation < 0.0011),
because gradient descent optimises a loss with no knowledge of the deployment quantizer.

**At the level of layer outputs**, the same ratio behaves differently, because a rank-`r`
adapter concentrates its effect on an `r`-dimensional subspace while quantization error
spreads across all `d_in` input directions. On inputs inside that subspace, signal is
amplified relative to noise by `√(d_in/r)` — a factor of 6.2–16.5 across the nine
adapters measured, at ranks 16 to 128.
A layer whose weight-space cosine is 0.14 can carry an output signal-to-noise ratio near
1.6. **Weight-space erasure and layer-output survival are not in tension; they are one
measurement read at two levels of the weight-to-output map.** Behaviour is a third level
and follows from neither.

## What we find

1. **Erasure with survival.** At INT4 g128 on the deployment path, stored values change in
   85.5% of positions and stored codes in 2.1%, with 98.9% of codes unchanged when the
   grid is held fixed, and **no loss of elicitation capability is detectable** — while
   the constraint side tightens by 8.3 points — **measured on the same six
   adapters under the same regime**. The mean
   degrades as the grid coarsens — 99.2% → 77.2% → 57.8% across INT4 g128, INT4
   per-channel and INT3 g128, all three paired contrasts excluding zero (B.8) — but
   **individual adapters do not**: four of six fall monotonically across the three
   quantized grids and only `gold` falls at every step including BF16, and at INT3 the mean
   sits between two adapters below half and two above 80% on the pre-registered
   instrument (three and one floor-corrected, B.7).

2. **A parameter-free channel model** predicting weight-space retention within 2.3%
   across every adapter tested, together with a derived subspace-amplification law
   accounting for why behaviour outlives weights.

3. **The dissociation is benign.** Where behaviour does degrade, the trained *capability*
   weakens while the trained *constraint* holds (suppression ratio 0.18–0.27, Cliff's
   *d* between −0.56 and −0.83 at every precision). This is the opposite of the alarming
   failure mode — retained knowledge with lost restraint — and it is the opposite of what
   we predicted before measuring it.

4. **The predictive gap.** Weight-space retention does not predict behavioural retention.
   Within six adapters matched on rank, scaling, base model, recipe *and* measured
   output SNR to 3.3%, behavioural retention spans 28.7%–86.4% (28.4–84.4%
   floor-corrected); six of the seven pairs
   the data can resolve invert the ordering; and the adapter with the largest weight-space footprint has no
   measurable target behaviour at all. This limits what our own released tool can claim,
   and we say so in the tool's output.

## Contributions

- **A parameter-free model of adapter retention under merge-then-quantize**, validated
  within 2.3% on nine published adapters spanning 1.1 decades of magnitude, and on a
  synthetic sweep across three decades, with all three of its licensing assumptions
  measured rather than assumed (§4.1), the first of them conditionally. The
  third was created by an argument we added a revision round after the model, and counted
  only when a reader counted for us.
- **A derived subspace-amplification law**, `√((d_in/r)/(1+c/r))`, reconciling
  weight-space erasure with *layer-output* survival. Its single empirical input `c ≈ 0.87`
  is a correction term predicted by the channel model, not a fitted scale (§3.6, §4.4).
  It accounts for the **level** — why a weight-space SNR of 0.13 still leaves subspace
  signal above noise — and, by §5.4, not for which adapter beats which.
- **End-to-end behavioural measurement** on the same models as the weight measurement,
  establishing no detectable loss at INT4 g128, a dose-response monotone in the mean but
  not in every adapter, and the benign direction of the dissociation (§5.1–5.3).
- **The predictive gap** (§5.4): three demonstrations, differing in kind, that
  weight-space measurement does not predict behavioural outcomes.
- **A weight-space account of why two opposing results can both hold.** Prior work reports that
  compressing *delta weights* protects alignment; we report that merged adapters are
  weight-space-erased. We argue these are the same law evaluated at opposite ends of
  `|Δ|/s`, distinguished by **which tensor sets the quantization scale** (§2.5), and we
  measure the unmerged case rather than arguing it (`METHODOLOGY.md`). Their claim is behavioural and
  ours is not, so this is an account of the mechanism, not a reconciliation of findings.
- **`ar.predict`**, a tool computing effective adapter magnitude from published
  checkpoints without a GPU — together with an explicit statement, in its own output, of
  what it cannot predict (Appendix A).
- **A record of what measurement corrected** (`METHODOLOGY.md`): eight methodological practices, each
  evidenced by a specification error of our own that measurement caught before
  publication.

## Scope

Every weight-space number in this paper is a statement about stored weights, and we hold
to that distinction in the prose throughout: an adapter whose weights are erased is not
thereby an adapter whose behaviour is erased, and our own results are the reason for the
care. Behavioural claims are correspondingly narrow — they cover rank-32, `α/r = 2`
adapters on one base model from one training recipe, and do not inherit the rank and
convention coverage of the weight-space measurements (§8.1).
