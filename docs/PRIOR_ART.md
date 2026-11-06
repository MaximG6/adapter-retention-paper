# Prior Art

**Date of search:** 2026-07-29. **Phase:** 0, day 1, before any code.
**Purpose:** determine whether the Phase 0 numerical result is already published, and if so, how much of it.

## How to read this file

Every entry carries a **verification level**, because the difference between "I read the abstract" and "it appeared in a search listing" matters when the output is a citation:

- **FETCHED** — abstract retrieved directly from arXiv and read. Title, authors, ID, and date are as reported by arXiv.
- **LISTED** — appeared in search results with a title and ID, abstract not independently retrieved. Treated as a lead, not a citation. **Must be upgraded to FETCHED before it enters the paper's bibliography.**

No entry in this file is a citation until it is FETCHED. Nothing here was reconstructed from memory.

---

## 1. The closest work: mechanism asserted, never measured

### Quantization-Robust LLM Unlearning via Low-Rank Adaptation
**arXiv:** 2602.13151 · **Date:** 2026-02-13 · **Verification:** FETCHED
**Authors:** Abitante, Pasquali, Garcia, de Oliveira, da Silva Paula, Barros, Kupssinskü

**Summary:** Argues that full-parameter unlearning produces weight updates small enough to be "masked" by aggressive low-bit PTQ, and that concentrating the update into a LoRA subspace with a higher learning rate produces a displacement large enough to cross the quantization boundary and survive. Evaluated on Llama-2-7B with MUSE.

**Coverage verdict: THIS IS THE ONE TO WATCH — and it does not scoop us.**

This paper states our mechanism almost in our own words: updates below the step size are erased; the fix is to make the displacement larger. That is the intuition our Phase 0 is built on, and **we can no longer present the mechanism as our observation.** Anyone reviewing our paper who knows this one will say so.

But it asserts the mechanism to motivate a method. It does not measure it. Confirmed absent: no retention ratio, no bit-flip rate, no rank sweep against retention, no quantification of what fraction of the delta is destroyed. The direction is also opposite to ours — it uses LoRA as the *fix* that survives quantization, where we ask whether a LoRA that someone already trained *does* survive. And the domain is unlearning, not alignment.

**Net:** converts our contribution from "we noticed this" to "we measured this." That is a smaller claim and a more defensible one. It also helps us: an independent group asserting the mechanism is evidence the question is worth quantifying.

---

## 2. Quantization-aware LoRA methods: adjacent, solve the inverse problem

These are the papers the plan flagged as the scoop risk. They are not.

### LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models
**arXiv:** 2310.08659 · **Venue:** ICLR 2024 · **Verification:** FETCHED

**Summary:** Jointly quantizes an LLM and finds a LoRA initialization that narrows the gap between the quantized and full-precision model, improving downstream fine-tuning over QLoRA at every precision.

**Coverage verdict: NO OVERLAP WITH OUR MEASUREMENT.** LoftQ's problem is *initialization before fine-tuning on an already-quantized base*. Ours is *fate of an already-trained adapter merged into a base that is quantized afterwards*. Different order of operations, different quantity. LoftQ is motivated by the same underlying tension (quantization and low-rank adaptation interact badly) and belongs in Related Work, but it publishes none of our metrics.

### QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models
**arXiv:** 2309.14717 · **Date:** 2023-10 · **Verification:** FETCHED

**Summary:** Balances the degrees of freedom of quantization and adaptation with group-wise operators, so that after fine-tuning the model and adapter merge into a quantized model "without loss of accuracy."

**Coverage verdict: ADJACENT, AND THE CLOSEST THING TO A MOTIVATING ADMISSION.** That phrase — merging "without loss of accuracy" as an advertised feature — is only a selling point if the naive path *does* lose accuracy. QA-LoRA is therefore implicit evidence that merge-time loss is real and known. But it is a training-time method that avoids the problem by construction; it never quantifies the loss on the naive path, which is exactly our measurement. Cite as motivation, not as a competing result.

### GPTQ-intrinsic LoRA: A Near-optimal Algorithm for Low-precision Quantization with Low-rank Adaptation
**arXiv:** 2606.01412 · **Date:** 2026-05-31 · **Verification:** FETCHED
**Authors:** Shihao Zhang, Rayan Saab

**Summary:** Post-training quantization with low-rank correction, W ≈ Q + LR. Information-theoretic lower bounds plus a GPTQ variant that folds the low-rank correction into the calibration Hessian. Experiments on Qwen3 and DeiT.

**Coverage verdict: MATHEMATICALLY THE NEAREST, STILL NOT OUR MEASUREMENT.** It explicitly analyses the regime where the low-rank correction's Frobenius norm is comparable to the quantization error — the same regime we are measuring. Flagged on first pass as the highest residual scoop risk and read in full the same day. **Verdict below.**

#### Full-text read, 2026-07-29 — it does NOT derive our curve

Resolved, with the decisive detail being *what L and R are*:

- **Its low-rank term is a designed compensator, not a pre-existing adapter.** `L = V_r`, the top-r right singular vectors of the *calibration matrix X*, and `R` is initialized at **zero** and updated across GPTQ steps to absorb quantization error. The optimization is Eq. (1), `min ‖XW − X(Q+LR)‖²_F`, jointly designing `Q` and `LR` from scratch. Our Δ is fixed, given, and trained by someone else on data we never see. Different mathematical object.
- **No retention bound.** No theorem bounds `‖Q(W+Δ) − Q(W)‖/‖Δ‖` or anything algebraically equivalent for a fixed Δ.
- **No per-weight step-size condition.** Its error bounds depend on `‖X − X_r‖²_F` (calibration-data residual) and grid spacing δ — not on per-weight delta magnitude relative to `s/2`. The `|Δ| < s/2` erasure condition that drives our step-ratio distribution appears nowhere.
- **Its rank dependence runs the opposite way.** The §4.4.1 bound `‖XW − X(Q+LR)‖²_F ≲ (1 − r/N)·σ²_N/B²` says a *larger designed compensator absorbs more error*. That is compensation capacity. Ours is the survival probability of a delta that already exists. Both are monotone in r and they are not the same quantity — conflating them would be an error, and one a reviewer might attempt on our behalf.
- **Per-channel only.** No group-wise scale analysis, so our group-size axis is untouched.
- **No retention-vs-rank plots.** Tables 1–3 report perplexity and accuracy against baselines.

**Consequence: we are not measuring something this paper derives.** Of the two outcomes Max named, this is the second — its theory does not predict our curves, and we proceed as planned. Still cite it as the nearest theoretical neighbour, and state the compensation-capacity/retention distinction explicitly in Related Work, because the `(1 − r/N)` bound looks superficially like a retention-vs-rank result and is not one.

One genuine borrowing: its `W ≈ Q + LR` decomposition is a clean formalism for the `base_quant + adapter` condition in Phase 1, where the adapter is served unmerged on a quantized base. Worth adopting its notation for that condition.

### CLoQ: Calibrated LoRA Initialization for Quantized LLMs
**arXiv:** 2501.18475 · **Verification:** LISTED

Same family as LoftQ (initialization for fine-tuning on a quantized base). Lead only; upgrade before citing.

---

## 3. Quantization and safety/alignment: behavioral results, no weight-level analysis

This is the cluster the plan pre-identified. All four IDs in the plan document resolved to real papers.

### Quantization Undoes Alignment: Bias Emergence in Compressed LLMs Across Models and Precision Levels
**arXiv:** 2605.15208 · **Date:** 2026-05-02 · **Verification:** FETCHED
**Authors:** Plawan Kumar Rath, Rahul Maliakkal

**Summary:** Qwen2.5-7B, Mistral-7B, Phi-3.5-mini at five precisions (BF16 to 3-bit), 12,148 BBQ bias items across 5 seeds. Extreme quantization makes previously unbiased items exhibit stereotypical behaviour, and aggregate metrics miss it — perplexity rises under 0.5% at 8-bit while behaviour degrades.

**Coverage verdict: OVERLAPS OUR PHASE 1 FRAMING, NOT PHASE 0.** No LoRA anywhere in it — it quantizes already-aligned models. Confirmed: no weight-level retention metric of any kind. Its real significance is methodological and it is bad news we should absorb rather than dodge: it independently establishes that **perplexity is a false negative for behavioural degradation.** Our Phase 1 must therefore not lean on perplexity, and its seed-and-item design is a good model for ours. It is also the strongest existing statement of "compression undoes alignment," so our Phase 1 novelty rests specifically on the *merged-adapter* pathway, not on the general claim.

### Safety-Preserving PTQ via Contrastive Alignment Loss / Alignment-Aware Quantization for LLM Safety
**arXiv:** 2511.07842 · **Verification:** LISTED (two title variants seen across v1/v2; resolve before citing)

PTQ that adds a contrastive alignment loss so the objective optimizes behavioural alignment rather than reconstruction error alone. Premise — standard PTQ ignores alignment and models keep low perplexity while losing safety — is our Phase 1 premise, arrived at independently. Method paper, no adapter retention analysis. Lead.

### Preserving Fairness and Safety in Quantized LLMs Through Critical Weight Protection
**arXiv:** 2601.12033 · **Verification:** LISTED

Protects safety-critical weights during quantization. Directly relevant to Phase 1 and a potential mitigation baseline. Lead.

### The Joint Effect of Quantization and Sampling Temperature on LLM Safety Alignment: A Factorial Analysis
**arXiv:** 2606.29581 · **Verification:** LISTED

Factorial design over quantization and temperature. Relevant to our Part 4.4 concern about effective sample size under greedy decoding — they may have already characterised the temperature axis we planned to sample. Read before finalising the Phase 1 decoding protocol. Lead.

---

## 4. Delta-weight compression: same objects, opposite operation

### Quantized Delta Weight Is Safety Keeper
**arXiv:** 2411.19530 · **Date:** 2024-11-29 · **Verification:** FETCHED
**Authors:** Yule Liu, Zhen Sun, Xinlei He, Xinyi Huang

**Summary:** Evaluates BitDelta-style delta-weight quantization against security threats on Llama-2-7b-chat. Reports a "free lunch": under 10% utility loss, partial compression mitigates alignment-breaking by up to 66.17%, backdoors by 64.46%, targeted manipulation by up to 90.53%.

**Coverage verdict: CLOSEST ON SUBJECT MATTER, AND THE MOST INSTRUCTIVE CONTRAST.** It is the one paper touching delta weights, quantization, and safety together — so it must be cited and distinguished carefully.

The distinguishing detail is the arithmetic, and it cuts in our favour. They quantize **Δ on its own**; we quantize **W + Δ jointly**. Quantizing Δ alone sets the scale from Δ's own dynamic range, so Δ is preserved comparatively well by construction. Quantizing W + Δ sets the scale from W's range, which is one to two orders of magnitude larger, so Δ competes against a step size it had no part in setting. **These two operations have opposite retention behaviour, and that contrast is a genuine contribution of ours rather than a weakness.**

Their finding also inverts ours in sign: they find compression *protects* alignment against a deliberate attacker. We ask whether compression *destroys* alignment accidentally. Both can hold simultaneously — different threat models, different arithmetic — and saying so explicitly will preempt the obvious reviewer objection.

### Task Vector Quantization for Memory-Efficient Model Merging
**arXiv:** 2503.06921 · **Date:** 2025-03-10 (rev. 2025-08-07) · **Verification:** FETCHED

**Summary:** Quantizes task vectors instead of full checkpoints; notes task vectors have a weight range an order of magnitude narrower than fine-tuned weights, so quantizing them incurs less error. Residual decomposition for 2-bit.

**Coverage verdict: SUPPORTS OUR PREMISE FROM THE OTHER SIDE.** The observation that task vectors are an order of magnitude narrower than weights is precisely why a delta merged into full weights is vulnerable to a step size set by those weights. They exploit the narrow range deliberately; we show it is a liability when the delta is merged first. Goal is deliberate compression, not accidental erasure. No retention-versus-rank curve.

### Others in this family (all LISTED, leads only)
- **Recover-LoRA** (2606.04238) — recovers capability lost to aggressive quantization via LoRA + distillation. Post-hoc repair, not measurement.
- **D-QRELO** (2604.16940), **DeltaLLM** (2501.18596), **SVD-based delta compression** (2506.11087) — delta compression methods. Same objects, engineering goal.

---

## 5. The reverse framing: is merge-then-quantize assumed safe?

The task asked whether the assumption is widespread and unexamined. **It is, and it is documented in the tooling.**

**Hugging Face PEFT documentation** (`docs/source/developer_guides/quantization.md`) presents the standard deployment path as `merge_and_unload()` → quantize with GPTQ/AWQ. The caveat it does raise is about the *opposite* order: merging an adapter into an already-quantized model "leads to rounding errors, decreasing model performance," with the advice to keep adapters separate in that case.

So the ecosystem has noticed the hazard in one direction and treats the other direction as the safe default. Our Phase 0 measures exactly the direction the documentation implicitly blesses. **Verification:** FETCHED (search-surfaced documentation text, not an arXiv paper).

Practitioner writing (Benjamin Marie, Rohan Paul, and others — blog-level, not citable) repeats the merge-then-quantize recipe with warnings about precision mismatch but no quantitative retention analysis. This is folk knowledge exactly as the plan predicted: enough awareness that we cannot claim novelty of the *idea*, no numbers anywhere that would preempt the *measurement*.

**This section is a motivation asset.** "The reference implementation's documented happy path is the one nobody has measured" is a stronger opening than "we had an idea."

---

## 6. Deliberately excluded: LoRA as a safety *attack*

A large literature — *LoRA Fine-tuning Efficiently Undoes Safety Training in Llama 2-Chat 70B* (2310.20624, LISTED) and successors — shows an adversary can strip safety with a cheap LoRA.

**Not our question, and the distinction must be explicit in the paper.** Those describe an attacker deliberately removing alignment. We describe alignment evaporating by accident during a routine deployment step performed by someone trying to *keep* it. Same components, opposite intent. Conflating them would be a serious framing error, and a reviewer skimming our abstract could easily make it for us if we are not precise.

---

## 7. Coverage assessment

| Component of our Phase 0 | Published already? | By whom |
|---|---|---|
| Mechanism: small deltas erased below the step size | **YES, asserted** | 2602.13151, qualitatively, unlearning domain |
| Merge-then-quantize loses accuracy | **YES, folk knowledge** | QA-LoRA implicitly; PEFT docs; practitioner blogs |
| Retention ratio ‖Δ_eff‖/‖Δ‖ for a merged adapter | **No** | — |
| Bit-flip rate under merge-then-quantize | **No** | — |
| Step-ratio distribution \|Δ\|/(s/2) | **No** | — |
| Retention vs. **rank** curve | **No** | — |
| Retention vs. group size / precision / module type | **No** | — |
| Layer-depth retention profile | **No** | — |
| Link from numerical retention to alignment *behaviour* | **No** | Nearest: 2605.15208, no LoRA, no weight metrics |
| Threshold guidance: retention below which behaviour breaks | **No** | — |

**The mechanism is claimed. Every number we planned to produce is unclaimed.**

Honest caveat: absence of evidence after one day of searching is not proof of absence, particularly for negative results buried in appendices of quantization papers I did not open. The two entries most likely to contain something that surprises us are **2606.01412** (theory in our exact regime) and **2411.19530** (delta + safety + quantization). Both are FETCHED at abstract level only. Read both in full before writing Method.

---

## 8. Recommendation

# PROCEED WITH A NARROWED CLAIM.

Not "proceed as planned" — the plan's implicit framing that we noticed the erasure mechanism is no longer available, and pretending otherwise would be caught.

**Drop this claim:** that merging a low-rank adapter before quantization can erase it. Asserted in 2602.13151, implied by QA-LoRA's design, folk knowledge among practitioners. Anyone who claims it as a discovery in late 2026 looks underread.

**Claim these instead:**

1. **The first quantitative characterisation of adapter retention under merge-then-quantize**, as a function of rank × precision × group size × module type × depth. The mechanism is folklore; the dose-response surface does not exist in the literature. This is the paper.
2. **The link from numerical retention to alignment behaviour.** No paper connects a weight-level retention metric to whether tuned behaviour persists. 2605.15208 has the behaviour without the weights; 2602.13151 has the weights-level intuition without the behaviour.
3. **The contrast between quantizing Δ alone and quantizing W + Δ.** 2411.19530 does the former and finds compression protects; we do the latter and expect it destroys. Both true, opposite arithmetic, and nobody has stated them side by side.
4. **Deployment guidance:** if retention is poor, serve the adapter on a quantized base instead of merging. Already in the plan as `base_quant + adapter`. Given that the PEFT docs recommend the opposite order, this is directly actionable.

**Three consequences, to act on now rather than in week three:**

- **The rank sweep is promoted from supporting evidence to the primary result.** A single retention number at rank 16 is a restatement of known folklore. The *curve* is the contribution. If time is cut, cut anything before cutting rank coverage.
- **Read 2606.01412 and 2411.19530 in full before writing Method.** Highest residual scoop risk, and 2606.01412's bounds may predict our curves — worth knowing before we produce them, not after.
- **Phase 1 must not rely on perplexity.** 2605.15208 independently shows perplexity misses behavioural degradation (<0.5% change at 8-bit while bias emerges). Our KL and task batteries already avoid this; do not add perplexity as a headline metric.

**Not recommended:** stopping, or pivoting to behavioural-only. Phase 0 survives contact with the literature intact — every metric in section 7 is unclaimed. The one thing that changed is which sentence goes in the abstract.

---

## 9. Searches run

`LoftQ LoRA-fine-tuning-aware quantization` · `QA-LoRA quantization-aware low-rank adaptation` · `merging LoRA adapter then quantizing degrades accuracy retention` · `safety alignment degradation post-training quantization LLM` · `LoRA weight update below quantization step size erased low-rank delta magnitude` · `merge LoRA quantize INT4 lost/erased/survive empirical study rank` · `peft merge_and_unload then GPTQ AWQ quantize recommended workflow` · `retention ratio Frobenius norm effective weight delta quantization LoRA rank sweep` · `does safety fine-tuning survive quantization LoRA adapter compressed deployment jailbreak` · `quantization erases task vector fine-tuning delta model editing compressed weights`

**Gaps in this search, stated honestly:** no non-English sources; no systematic sweep of NeurIPS/ICML/ICLR 2025–2026 proceedings beyond what search surfaced; no citation-graph traversal forward from LoftQ or QA-LoRA, which is the most likely place a direct hit is hiding. If a scoop exists, forward citations of QA-LoRA are where I would look next.

ATP is **2510.04860**, *Alignment Tipping Process: How Self-Evolution Pushes LLM Agents Off the Rails*, Han et al., submitted 2025-10-06, v2 2026-02-11.

---

## 9b. Weight outliers and quantization difficulty (added 2026-07-30, after EXP-009)

Not a scoop-risk cluster — this is established work we must cite because our layer-1 spike finding is a *consequence* of it.

### LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale
**arXiv:** 2208.07339 · **Venue:** NeurIPS 2022 · **Verification:** FETCHED
Emergent outlier features appear at scale, affect all layers, and dominate quantization error; up to 150k outliers per 2048-token sequence at 13B, concentrated in at most ~7 feature dimensions. Handled by isolating them into a 16-bit mixed-precision path.

### AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration
**arXiv:** 2306.00978 · **Venue:** MLSys 2024 · **Verification:** FETCHED
Not all weights matter equally; protecting ~1% of salient channels, identified from *activation* rather than weight magnitude, greatly reduces quantization error. Implemented as an equivalent scaling transformation rather than mixed precision.

### Massive Activations in Large Language Models
**arXiv:** 2402.17762 · **Venue:** COLM 2024 · **Verification:** FETCHED
A very small number of activations exceed the median by up to four orders of magnitude, sit in fixed feature dimensions, stay roughly constant across inputs, and function as bias terms.

**Coverage verdict: WE CLAIM THE CONSEQUENCE, NOT THE PHENOMENON.**

Our measurement that `gate_proj` at layer 1 has a median step size 83.8x its 1st percentile is a restatement of this literature's central observation, and must be presented as such. What is unclaimed is the link: **layers that are hardest to quantize are also where adapters are least preserved**, and the variation is driven by the base model's weight distribution rather than by anything about the adapter.

Actionable prediction that follows and belongs in Phase 1 if time allows: outlier-aware quantizers should preserve adapters *better* in exactly the layers where naive affine quantization preserves them worst.

---

## 10. Gap-closing sweep (same day, 2026-07-29)

Section 9 named three gaps and flagged forward-citation traversal as the likeliest hiding place for a direct hit. Closed as follows, timeboxed to one pass each.

### Forward citations of QA-LoRA (2309.14717)
Semantic Scholar citation graph, 40 citing papers returned, complete (not truncated). Scanned every title. **Nothing measuring adapter retention, erasure, or survival under PTQ.** The quantization-adjacent citers (Recover-LoRA 2606.04238, *LLM Compression with Jointly Optimizing Architectural and Quantization choices* 2606.04063, *Signs Beat Floats* 2605.24058) apply quantization and LoRA sequentially without isolating what happens to the delta.

### Forward citations of LoftQ (2310.08659)
Semantic Scholar, **100 returned and the list is truncated** (API reports a further offset). This is partial coverage and I am not claiming otherwise — LoftQ is an ICLR 2024 paper with more citations than one page. Of the 100 scanned, five titles warranted a look; the two closest were fetched:

- **Q-resafe** (2506.20251, 2025-06-25) — *Assessing Safety Risks and Quantization-aware Safety Patching for Quantized LLMs*. **Verification: FETCHED.** Evaluates safety across mainstream quantization techniques and calibration datasets, then patches. No LoRA/PEFT adapter pathway, no weight-level retention metric, no merge-then-quantize. Method paper. Tangential to Phase 1 but cite it — it is direct evidence that "quantization may compromise safety capabilities" is an established concern, which supports our motivation without touching our measurement.
- **Q-realign** (2601.08089, 2026-01-13) — *Piggybacking Realignment on Quantization for Safe and Efficient LLM Deployment*. **Verification: FETCHED.** Post-hoc defense that recovers safety alignment in fine-tuned LLMs by reframing quantization to serve compression and safety jointly. Method paper, behavioral metrics only, no retention ratio or bit-flip rate, does not study merge-then-quantize of a pre-trained adapter. **Notable for our framing:** the premise that you would *piggyback realignment on quantization* concedes that quantization disturbs alignment in fine-tuned models. Another independent voice for our motivation, still not our measurement.
- Not fetched, judged off-target from title and context: *Breaking the Blocks* (2601.22716), *AutoQRA* (2602.22268), *DiaBlo* (2506.03230), *LoRA Reduces Catastrophic Forgetting in Sequential Fine-Tuning* (2603.27707 — "retention" there means knowledge retention across adaptation cycles, not numerical delta survival).

### Proceedings sweep
One pass over ICML/NeurIPS/ICLR 2026 material. Surfaced a *safety-subspace* cluster (SafeLoRA, Safe Pruning LoRA 2506.18931, *Guardrail for Safety Preservation* 2510.14301, *Decoupling Safety into Orthogonal Subspace* 2510.09004, SafeAnchor 2604.17691) and a "Residual-Safe Quantization" item at ICML 2026. All LISTED, none fetched. This cluster projects LoRA weights onto safety-aligned subspaces or prunes for safety — **relevant to Phase 1 mitigation discussion, not to Phase 0 measurement.** Worth a proper pass before writing Related Work; not a scoop risk.

### Result of the gap-closing sweep

**Nothing turned up.** Three independent traversals (QA-LoRA citers complete, LoftQ citers partial, proceedings one pass) produced no paper measuring retention ratio, bit-flip rate, or step-ratio distribution for a merged adapter under PTQ. The coverage table in §7 stands unchanged.

**Residual risk after this sweep: LOW, and no longer concentrated.** The two papers I had flagged as most dangerous (2606.01412, 2411.19530) are now both resolved — neither measures our quantities. What remains is diffuse: an unfetched appendix somewhere in the 100+ untraversed LoftQ citers, or a negative result buried in a quantization paper's supplement. That risk does not justify further delay on day 1. **Recommendation unchanged: proceed with the narrowed claim.**

Two standing follow-ups, cheap and deferrable:
- Re-run the LoftQ citation traversal past offset 100 during the Phase 1 lull, when Related Work is being written anyway.
- Fetch the safety-subspace cluster before writing the Phase 1 mitigation discussion, since that is where it bears.
