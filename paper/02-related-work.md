# 2. Background and related work

*Draft. **Every claim attributed to another paper in this section was verified against
that paper's abstract in-session** (see `METHODOLOGY.md` M.1). Where a connection is our own inference
rather than the cited authors' finding, it is marked* **[our inference]** *explicitly.*

---

## 2.1 LoRA, merging, and scaling conventions

Low-rank adaptation represents a weight update as `Δ = γ·BA` with `A ∈ R^{r×d_in}`,
`B ∈ R^{d_out×r}`, and `γ = α/r` conventionally or `γ = α/√r` under rsLoRA. Deployment
typically *merges* the adapter — folding `Δ` into `W` — so that inference costs nothing
extra, and then quantizes the merged matrix.

The scaling convention is not cosmetic for our purposes: misreading the `use_rslora`
flag changes a rank-128 adapter's delta magnitude by `√128 ≈ 11.3×`, and magnitude is
the quantity that governs *weight-space* retention (§4.1). It does not govern
behavioural retention — §5.4 is three ways of saying so. Eight of the nine adapters we measure use
`α/r = 2`; one uses `α/r = 0.125` with rsLoRA.

## 2.2 Post-training quantization and where the difficulty lies

Group-wise affine PTQ stores weights as low-bit integers with a per-group scale and zero
point. The quantity that matters throughout this paper is the **step size** `s`, since a
weight perturbation smaller than `s` may leave the stored integer unchanged.

A substantial body of work establishes that the difficulty of LLM quantization is
concentrated in a small number of extreme values. **LLM.int8()** (Dettmers, Lewis,
Belkada & Zettlemoyer, arXiv:2208.07339) identifies systematic *emergent outlier
features* that dominate transformer performance and isolates those dimensions in 16-bit
while multiplying more than 99.9% of values in 8-bit. **AWQ** (Lin et al.,
arXiv:2306.00978) shows that "not all weights in an LLM are equally important" and that
protecting roughly 1% of salient weights greatly reduces quantization error —
emphasising that **to identify salient weight *channels* one should refer to the
activation distribution, not the weights**. **Massive Activations** (Sun, Chen, Kolter &
Liu, arXiv:2402.17762) documents a small number of activations up to ~10⁵× larger than
others, largely input-independent, functioning as bias-like terms.

**These results concern activations; our layer 1–3 bit-flip spike (§4.5) is a
weight-space observation.** We initially assumed the two were the same phenomenon. They
are not, and we tested it rather than leaving the assumption in place.

The spike is driven by a heavy small-step tail: `gate_proj` at layer 1 has a median step
size **83.5× its 1st percentile**, against 1.3–1.6× in control layers. Small `s` means
large `|Δ|/s` means more code flips. Because quantization groups run along the **input**
dimension, each group covers a contiguous block of 128 input channels — the same axis
activations live on — so the two framings make opposite, checkable predictions about
where the narrow-range groups sit.

Capturing per-input-channel activation magnitude on fixed calibration text (§4.5.1):

| module | step median/p1 | activation at **narrowest** 1% of groups | at widest 1% | Spearman(log `s`, activation) |
|---|---|---|---|---|
| layer 0 `gate_proj` (control) | 1.4 | 0.97 | 1.03 | +0.033 |
| **layer 1 `gate_proj`** | **83.5** | **0.17** | 1.12 | **+0.244** |
| **layer 2 `gate_proj`** | **44.6** | **0.19** | 1.58 | **+0.275** |
| **layer 3 `gate_proj`** | **145.1** | **0.15** | 1.05 | **+0.156** |
| layer 18 `gate_proj` (control) | 1.6 | 0.94 | 1.03 | +0.012 |

*(activation columns are relative to each module's own mean; split-half stability of the
activation profile is r ≥ 0.99 in the spike layers.)*

**Two findings, and the second is why the conjecture was wrong.** First, the association
is real and confined to the spike layers: step size and input-channel activation
magnitude are correlated at ρ = +0.16 to +0.28 in layers 1–3 and at ρ ≈ 0 in both
controls. Second, **the direction is the inverse of the massive-activation pattern**.
In `gate_proj`, where the spike lives, the narrow-range weight groups sit at input
channels whose activation is **0.15–0.19× the module average** — the *quietest* channels,
not the outlier-loud ones. The widest-range groups sit at the higher-activation channels.
`up_proj` in the same layers is milder on both axes (0.61–0.89× at its narrowest, ρ =
+0.04 to +0.11, two of the three at control level), and the 0.15–0.19 range is
`gate_proj` alone; §4.5 names both module families and this is where the quoted figure
comes from.

So the weight-space spike is **a distinct phenomenon from the activation outliers of
LLM.int8(), AWQ and Massive Activations**, not an instance of them. It is a
low-activation, narrow-weight-range structure specific to early layers. **[our
inference]** We have established the coincidence and its direction, not its mechanism;
whether these channels are near-inert and their weights consequently under-dispersed, or
whether some other structure produces both, is not something this measurement decides.

**This yields a concrete, falsifiable prediction about existing quantization methods.**
Outlier-aware schemes select what to protect by *high* activation: AWQ scales up the
salient channels identified from the activation distribution, and LLM.int8() routes the
outlier dimensions through 16-bit. The weight groups driving adapter erosion in early
layers sit at the **opposite** end of that distribution — activation 0.15–0.19× the
module mean. **They would not be selected for protection by any of these methods**,
because on the criterion those methods use they are the least salient channels in the
layer.

The prediction follows directly: applying AWQ- or LLM.int8()-style outlier protection to
a merged adapter should leave the layer 1–3 flip-rate spike substantially intact,
whereas protecting the **narrowest-step groups** — a criterion no current method uses —
should remove it. **[our inference]** We have not run this; it is stated as a named
experiment in Future Work (§9) rather than as a result. If it holds, it identifies a
class of weight that matters for adaptation retention and is invisible to the saliency
criterion the field currently uses.

## 2.3 Quantization-aware low-rank adaptation

**QLoRA** (Dettmers et al., arXiv:2305.14314) fine-tunes adapters through a frozen 4-bit
base and established the deployment pattern this paper examines the tail of. **LoftQ**
(Li et al., arXiv:2310.08659) and **QA-LoRA** (Xu et al., arXiv:2309.14717) are motivated
by the same interaction we study — that quantization and low-rank adaptation interfere —
but address it by *changing the training procedure*: LoftQ by choosing the low-rank
initialisation jointly with the quantized weights, QA-LoRA by balancing the degrees of
freedom of quantization and adaptation so that adapter and base integrate into a quantized
model afterwards. All three **control** the ratio `|Δ|/s` at training time. Our question
is what that ratio turns out to be for the adapters that already exist, trained without
any of them. **GPTQ-intrinsic LoRA** (Zhang & Saab,
arXiv:2606.01412) is the most closely related theoretical work: it establishes
information-theoretic **lower bounds for the layer-wise reconstruction problem** under
finite-alphabet and bounded low-rank compensation constraints, and proves **upper bounds
on reconstruction error** for its algorithm in which "the usual GPTQ dependence on
`‖X‖²_F` is replaced by the rank-`r` residual `‖X − X_r‖²_F`".

**How this differs from our question, stated precisely.** Zhang & Saab bound
*reconstruction error* — how well a quantized-plus-low-rank-compensated layer can
reproduce the original layer's output. That is a statement about the **capacity of a
low-rank term to compensate for quantization**. Our question is the reverse direction:
given an adapter that already exists and was trained without any knowledge of a
quantizer, **how much of it survives** merging and quantization. A bound on how much a
low-rank term *could* compensate does not tell you how much a particular trained adapter
*does* retain. Framing their bound as a "retention" result would misstate it, and we
avoid that framing.

More broadly, this entire line of work proposes *methods*. **We measure published
artefacts.** None of these papers reports the retention of existing, already-trained,
publicly released adapters under a standard deployment quantizer, which is the gap this
paper addresses.

## 2.4 Quantization, unlearning, and the erasure mechanism

The mechanism we measure — that a fine-tuning update can be numerically lost because it
falls below the quantization step — is asserted in the unlearning literature.
**Quantization-Robust LLM Unlearning via Low-Rank Adaptation** (Abitante et al.,
arXiv:2602.13151) states that standard full-parameter fine-tuning produces updates "too
small to survive 4-bit quantization", and that aggressive low-bit quantization can mask
unlearning updates, "causing quantized models to revert to pre-unlearning behavior".

**Their proposed remedy is the object we measure, and our result complicates it.** That
paper's method freezes the base model and concentrates unlearning into trainable
adapters *precisely so that* "the effective update is preserved after quantization" —
i.e. it treats LoRA as the mechanism by which an update becomes large enough to survive
the quantization step.

**We confirm their premise and find their remedy conditional on deployment.** That small
updates do not survive 4-bit quantization is exactly what §4.1 predicts and §4.2 measures.
What our measurements do not support is that being low-rank is itself sufficient: nine
published LoRA adapters, **merged** and quantized at INT4 g128, change only 1.1%–14.8% of
stored integer codes — under 6.2% for eight of the nine — with individual weight deltas
falling below the step size (§4.2). Concentrating an update into a low-rank adapter does
not, by itself, make the per-weight delta large relative to `s`.

**[our inference]** The distinction that matters is not full-FT versus LoRA but **which
tensor sets the quantization scale**. A merged adapter is subject to the base weights'
step size and is erased on the same terms; an adapter quantized on its own scale is not,
and `METHODOLOGY.md` measures that case directly (cosine 0.9948–0.9952 against 0.14–0.51 merged). Their
method freezes the base and keeps the update in adapters, which is the second
configuration, so their claim and ours are consistent (§2.5). What would be wrong is to
read "use LoRA" as sufficient without saying whether the adapter is merged before
quantization. This remains an
explanation to be tested, not as a finding.

Separately, our behavioural results (§5.1) show that near-total *weight-space* erasure
is compatible with behaviour surviving essentially intact — so "the update did not
survive quantization" and "the behaviour did not survive quantization" are not
interchangeable statements, and papers in this area (ours included) need to say which
one they mean.

## 2.5 The apparent contradiction with "compression protects alignment", and a weight-space account of it

**Quantized Delta Weight Is Safety Keeper** (Liu, Sun, He & Huang, arXiv:2411.19530)
reports that partial compression "can enhance model security against fine-tuning-based
attacks with bearable utility loss", mitigating alignment-breaking risks **by up to
66.17%** — a result the authors describe as a "free lunch". That paper quantizes the
**delta weights** between the fine-tuned and base model, in the manner of BitDelta,
rather than the merged matrix.

Taken at face value, that finding and ours point in opposite directions: they report
compression *preserving* alignment properties, while we report the merged adapter's
weight-space representation being almost entirely erased.

**They are not directly comparable, and saying so is part of the point.** Their endpoint
is behavioural — resistance to fine-tuning attacks, backdoors and targeted manipulation —
and every number in §4 of this paper is a statement about stored weights. A behavioural
finding and a weight-space one cannot contradict each other without a bridge, and §5.4
is our own evidence that the bridge does not hold in general. **What we offer below is a
weight-space account of why both can be true**, not a reconciliation of the two findings.

**[our inference — this reconciliation is ours, not a claim made by either cited
paper.]** The two setups differ in **which tensor determines the quantization scale**:

| | tensor quantized | scale set by | consequence for `Δ` |
|---|---|---|---|
| Liu et al. (2411.19530) | `Δ` alone | `max(Δ) − min(Δ)` | step size is proportional to the delta's *own* range, so `\|Δ\|/s` is O(1) and the delta is well represented |
| this paper | `W + Δ` | `max(W+Δ) − min(W+Δ) ≈ range(W)` | step size is set by the **base weights**, which are orders of magnitude larger, so `\|Δ\|/s ≈ 0.01` and the delta is sub-threshold |

Under our channel model (§4.1), retention is governed entirely by `|Δ|/s`. When `Δ` sets
its own scale, that ratio is large and retention is high; when `W` sets the scale, it is
small and retention is low. **Both results are then the same law evaluated at opposite
ends of one ratio**, and the practical rule that follows is concrete: *an adapter kept
separate and quantized on its own scale is numerically preserved; the same adapter
merged before quantization is not.*

**We tested it** (§7). Quantizing `Δ` alone on its own grid, same simulator, same bit
widths, all nine adapters, 756 records: at INT4 g128 `|Δ|/s` rises from 0.011–0.149 merged
to **2.31–2.38**, cosine from 0.14–0.51 to **0.9948–0.9952**. The prediction is **P10**,
registered before the run; three of its four clauses held, and the fourth put `|Δ|/s` in
0.1–1.0 and measured 2.3 — right in direction, wrong in range. This remains a statement
about stored weights on one arithmetic operation, and says nothing about behaviour or
about an end-to-end serving path.

## 2.6 Evaluating alignment under compression, and why we do not lead with perplexity

Work on compressed-model behaviour repeatedly finds that aggregate quality metrics miss
behavioural change. **Quantization Undoes Alignment: Bias Emergence in Compressed LLMs
Across Models and Precision Levels** (Rath & Maliakkal, arXiv:2605.15208) reports that
"perplexity increases by less than 0.5% at 8-bit and under 3% at 4-bit across all three
models, yet 2.5–5.6% of items already develop new biases at 4-bit", concluding that
"aggregate evaluation metrics systematically miss fairness-critical degradation".

This directly motivates a design choice in §3.7. We do not use perplexity as a
behavioural endpoint. Our behavioural measurements are targeted at the specific trained
behaviour — whether the model still produces hints that identify its secret word, and
whether it still avoids stating it — with a decoding-entropy control to separate genuine
behavioural change from distribution flattening. A perplexity-based evaluation of our
INT4 g128 condition would have shown very little and concluded very little.

## 2.7 Behavioural testbeds

The **Taboo** setting (Cywiński, Ryd, Rajamanoharan & Nanda, arXiv:2505.14352)
fine-tunes a model to describe a specific secret word without ever stating it, with the
word absent from both training data and prompt. It is well suited to our purpose because
it yields a **judge-free** ground truth and decomposes into two separately measurable
sides (constraint and capability, §3.7).

We use public Qwen3-8B instantiations of this setting. As noted in §8.8, those
checkpoints carry no documentation of their training recipe or their relationship to
that paper, so we attribute the **setting** to Cywiński et al. and treat the checkpoints
as independent public artefacts.

**XSTest** (Röttger, Kirk, Vidgen, Attanasio, Bianchi & Hovy, arXiv:2308.01263; NAACL
2024) provides the design principle behind our over-refusal control: safe prompts phrased
in language resembling unsafe prompts are the ones over-safe models refuse. Our
surface-harmful/actually-benign prompts are authored on that principle and are **not**
XSTest's items; we claim no correspondence to its ten prompt types (§3.10).

The **Alignment Tipping Process** (Han et al., arXiv:2510.04860) studies alignment
erosion in self-evolving agents over repeated rounds. We do not use it in this paper.
Whether quantization changes the *rate* of that erosion is the natural extension of our
results and is left to future work.
