# 8. Limitations

*Draft. Stated as constraints on what the measurements license, with the scope of each
claim named explicitly.*

---

## 8.1 The behavioural population is one condition replicated six times

Phase 1's six adapters are all **rank 32, `α/r = 2`, on one base model, from one
training recipe**, differing only in the target word. That is effectively *one condition
replicated six times with different secret words*, not a survey of adapters.

Our weight-space measurements span ranks 16–128, both scaling conventions, two base
models and four training regimes. **The behavioural claims do not inherit that
coverage.** Every statement in §5 — the dose-response, the benign
dissociation, the survival of INT4 g128 — is a statement about rank-32, `α/r = 2`
adapters on Qwen3-8B trained by one recipe. Nothing in this paper licenses extending it
across rank, scaling convention, base model, or task.

This is the single largest gap between what we measured and what a reader might want to
conclude, and it is the reason the paper's headline is stated at the precision it is.

## 8.2 We did not run the across-population predictive test

The predictive gap (§5.4) is established **within** a matched population. Testing
whether output SNR predicts behavioural retention **across** dissimilar adapters would
require batteries for adapters we did not build them for, and we made a deliberate
decision not to pursue it.

The reasoning, recorded before the decision rather than after: the available predictor
range collapsed from 3.7× to 2.3× when the safety adapter failed instrument validation
(§6.4); both remaining candidates lacked validated batteries and might not have passed
(the safety adapter had just failed exactly that step); and the result would have
remained confounded regardless, since the candidate adapters differ in rank, base model,
recipe and task simultaneously. The best available outcome was a suggestive,
non-causal, n=3 correlation, which would not have overturned PG-1 to PG-3.

**What would change this.** A matched pair — two adapters identical in rank, base
model, recipe and task, differing only in effective magnitude — would remove the
confound and make the test causal. No such pair exists publicly; it would have to be
trained. We regard this as the correct next experiment and describe it in Future Work
rather than claiming the present study substitutes for it.

## 8.3 Weight-space quantize–dequantize, not a deployment kernel

We simulate quantization as quantize–dequantize on the target projections, validated
bit-exact against `gptqmodel`. This is the arithmetic whose retention we characterise,
**not** an end-to-end deployment stack: we do not exercise fused INT4 kernels, and our
results do not capture kernel-level effects such as accumulation order or activation
quantization.

We also do not cover **GGUF K-quants**, which use block-wise super-block scales rather
than plain affine quantization and would need llama.cpp's own quantizer rather than a
reimplementation. Their retention behaviour is unmeasured here.

## 8.4 Effective sample size is far below nominal

Greedy decoding at temperature 0 makes generation deterministic, so seeds are not a
replicate axis: re-running reproduces outputs exactly. All intervals are therefore
bootstrapped over **intent clusters** (per-adapter) or **adapters** (pooled), never over
observations. The 32 prompts are not 32 independent draws: they are 8 intents × 3
paraphrases plus 8 adversarial prompts. Measured (B.12) the battery carries **23–26
effective units**, not 32 and not the 16 an earlier draft assumed.

The resulting precision is limited. Per-adapter 95% intervals are **13–47% wide** under
the clustered estimator, which is why PG-2 resolves only 4 of 15 pairs at INT3 and 1 of
15 at INT4 g128. That single INT4 pair involves an adapter whose point estimate is 116.2%
of its own BF16 baseline — impossible, and therefore a statement about the instrument
rather than about the adapter. An earlier reading of the whole INT4 spread as a real
effect was our own error, corrected by the bootstrap.

## 8.5 The safety adapter is confounded four ways, and we registered this in advance

`responsible-ai-safety` is simultaneously the **only** alignment-trained adapter, the
**only** Llama-3.1-8B, the **only** refusal-framed task, and the **only** rank-16
adapter with a behavioural battery attempt. Its divergence from the taboo family
therefore **cannot be attributed** to the training recipe, the base model, or the safety
framing — all change together with nothing to separate them.

This limitation was written into the pre-registration before the adapter was run, not
added after its result was known.

## 8.6 §6 is a case study at n=2

The finding that the safety adapter removes its base model's refusal under indirect
pressure rests on **2 of 8 jailbreak-framed prompts, on one adapter, at BF16**, with
each instance verified by reading the full trajectory. It is a case study motivating
the practice of checking that an adapter implements its advertised behaviour before
deploying it. It is **not** a population estimate about alignment LoRAs, and it says
nothing about that adapter's behaviour under quantization, which we did not measure.

## 8.7 Instrument limitations found by audit

Two instruments in this paper have documented failure modes that we report rather than
tune away, because in both cases the fix would not have changed the verdict and
retuning after seeing the data is the error the gate exists to prevent.

- **`p_refuse` responds to prompt appearance as well as model behaviour** (§6.5). Within
  a fixed prompt set it remains a valid across-condition comparison; its absolute level
  is not a refusal probability.
- **String-match refusal errs in both directions** — a false negative on soft refusals
  ("I'm not sure I can help with that"), and a false positive on vocabulary content
  (the marker "i am sorry" matching the English gloss in the entry
  *Je suis désolé(e)* *(I am sorry)* in a French vocabulary list). We report raw and audit-corrected counts separately (§6.2).

## 8.8 Checkpoint provenance

The Taboo checkpoints we use carry no model card documenting their training recipe or
their relationship to the paper that introduced the Taboo setting. We adopt the setting
from Cywiński et al. (arXiv:2505.14352) and treat the checkpoints as independent public
artefacts. Their internal consistency is measured — three same-recipe adapters give
weight-space cosines of 0.1380, 0.1389 and 0.1409, under 2% relative spread — but their
training details are not documented and we do not assume them.

---

# 9. Conclusion

Merging a LoRA adapter into a base model and quantizing to INT4 changes only
**1.1%–14.8%** of the model's stored integer codes — under 6.2% for eight of the nine
adapters measured — and replaces the intended weight update with something 1.7–7.4× its
size pointing in a largely uncorrelated direction. By any
weight-space measure, the adaptation is very nearly gone.

The capability is not. At INT4 with group size 128 — the standard deployment
configuration — **85.5% of the six taboo adapters' stored values differ from the base
model's, and no loss of elicitation capability is detectable**: retention 99.2%,
enumerated interval [90.7%, 107.6%], which spans parity and excludes losses beyond about
9%. The trained constraint, behaviour's other side, does move and **tightens**: the
adversarial leak rate falls 8.3 points, [+4.2, +12.5] (§5.1). Capability degrades only at
coarser grids, reaching 77.2% at
INT4 per-channel and 57.8% at INT3, and in this population it degrades in the benign
direction: the model becomes less able to express the trained behaviour while the
trained constraint holds, rather than retaining the capability and losing the
restraint.

Both halves follow from one quantity. `|Δ|/s` — the adapter's effective magnitude
against the quantization step — predicts the stored-weight outcome through a
parameter-free channel model accurate to 2.3% across two base models, four ranks, both
scaling conventions and four training regimes. The same quantity, applied to inputs
inside the adapter's rank-`r` active subspace rather than to individual weights, is
amplified by `√(d_in/r)`, and layer-output fidelity is *measured* 6.2–16.5× higher than
weight-space fidelity, rising as rank falls. **Weight-space erasure and layer-output
survival are not in tension; they are one measurement read at two levels of the
weight-to-output map.** Behaviour is a third level and follows from neither — PG-3 is
this paper's own counterexample.

The practical guidance is uncomfortable in both directions. INT4 g128 is **safer** than
the weight-space numbers suggest, and practitioners who would have been alarmed by a
cosine of 0.14 should not be. But weight-space diagnostics — including the tool we ship
with this paper — **cannot tell you which adapter will survive**. Within a population
matched on rank, scaling, base model, recipe and output SNR to 3.3%, behavioural
retention spans 28.7% to 86.4% (28.4–84.4% floor-corrected); among the pairs whose
difference we can resolve, the
ordering runs opposite to the predictor in six of the seven resolvable pairs; and the
adapter with the largest weight-space
footprint in our study has no measurable target behaviour at all. A practitioner
choosing between two comparable adapters gets no information from these measurements,
and we say so in the tool's own output.

The honest summary of what we set out to test is that the alarming version did not
occur. We began by asking whether a quantized alignment fine-tune is behaviourally the
base model. At standard settings it is not, and the mechanism by which it survives is
the same mechanism that makes the weights look destroyed. What replaces that concern is
narrower and more tractable: **effective adapter magnitude is the quantity that governs
retention, no adapter card publishes it, and no weight-space measurement of it predicts
behaviour.**

## Future work

Two of these are named experiments with pre-stated predictions, and we flag them as
untested rather than implying our data covers them.

**FW-1. Does outlier-aware quantization protect the wrong weights for adapter
retention?** Our layer 1–3 flip-rate spike is driven by weight groups sitting at input
channels whose activation is **0.15–0.19× the module mean** — the quietest channels,
the inverse of the massive-activation pattern (§2.2, §4.5.1). Existing outlier-aware
methods select what to protect by *high* activation: AWQ scales up the salient channels
identified from the activation distribution; LLM.int8() routes outlier dimensions
through 16-bit. **On that criterion, the groups driving adapter erosion are the least
salient channels in the layer and would not be protected.**

*Prediction:* applying AWQ- or LLM.int8()-style protection to a merged adapter leaves the
early-layer spike substantially intact, while protecting the **narrowest-step groups** —
a criterion no current method uses — removes it. This is a one-afternoon experiment on
our existing harness. If it holds, adaptation retention depends on a class of weight
that current saliency criteria are constructed to ignore, and a retention-aware
quantizer would need a different selection rule from an accuracy-aware one.

**FW-2. A matched pair.** Two adapters identical in rank, base model, recipe and task,
differing only in effective magnitude, would convert the predictive gap from a
within-population observation into a causal test (§8.2). No such pair exists publicly;
both would have to be trained.

**FW-3. Behavioural coverage across rank and convention**, which §8.1 identifies as the
largest gap between our weight-space and behavioural claims.

**FW-4. GGUF K-quants and deployment kernels** (§8.3), using llama.cpp's own quantizer
rather than a reimplementation.

**FW-5. Whether the benign dissociation generalises** beyond a suppression-style
behaviour to capabilities with no constraint component.

*Quantizing `Δ` on its own scale was FW-2 in an earlier draft. It is now measured (`METHODOLOGY.md`),
so it is a result rather than future work, and the remaining items are renumbered.*
