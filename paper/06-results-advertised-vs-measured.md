# 6. Advertised versus measured: an alignment adapter that does not implement refusal

*All values re-derived from
`results/raw/phase1/refusal_validation/*.jsonl` (96 records total: 64 refusal battery,
32 over-refusal control). BF16 only — **no precision comparison was run**, for the
reason given in §6.4.*

This section makes a different claim from §5. It is not a fourth instance of
weight-space measurement failing to predict behaviour; it is a claim about the gap
between what an adapter is **advertised** to do and what it **measurably** does. We
report it because it bears directly on the deployment question the paper opens with —
if you are shipping a quantized alignment fine-tune, the first thing worth checking is
whether the fine-tune implements the alignment behaviour at all.

## 6.1 Setup

One adapter in our set is marketed for alignment:
`Kurapika993/llama-3.1-8b-responsible-ai-safety-lora` (rank 16, `α/r = 2`, 224 target
projections) on `Llama-3.1-8B-Instruct`. It has the largest predicted **layer-output SNR**
in the set (6.00 against 1.62–1.67 for the taboo six), which is what makes §5.4's PG-3
work; it is *not* the largest on every weight-space axis, since `ao-v3-dpo-halluc` has a
higher cosine (0.5050 against 0.3298) and a higher code-flip rate (14.81% against 6.19%),
and it is not one of the six taboo adapters at all. The taboo instruments do not apply to
it — there is no secret word —
so we built the refusal battery of §3.10 and put it through the same gate at the same
thresholds.

## 6.2 The adapter adds no refusal, and removes it under indirect pressure

**Figure 10** plots these contrasts by prompt kind.

| axis | base `p_refuse` | aligned `p_refuse` | Cliff's d | ratio | string-refusal base → aligned (raw) |
|---|---|---|---|---|---|
| harmful, direct | 0.9994 | 0.9881 | −1.000 | 0.99 | 8/8 → 8/8 |
| harmful, indirect | 0.9988 | 0.8533 | −1.000 | 0.85 | 8/8 → **5/8** |
| harmful, pooled | 0.9991 | 0.9207 | −0.977 | 0.92 | 16/16 → 13/16 |
| benign | 0.1449 | 0.2850 | +0.445 | 1.97 | 0/16 → 1/16 |

**The string-refusal column is the raw marker heuristic; the audit below corrects it in
both directions.** Audit-corrected, harmful-indirect is 6/8 (one aligned refusal was
phrased in words the marker list does not contain) and benign is 0/16 (the single
aligned "refusal" is a false positive: the model produced a French vocabulary list,
and the marker *"i am sorry"* matched the **English gloss** in the entry
*Je suis désolé(e)* *(I am sorry)*, not the French phrase itself). Neither correction changes
any verdict, because the gate fails on magnitude, and we report both the raw metric and
its audited value rather than silently substituting one for the other.

**No axis clears the gate.** The base `Llama-3.1-8B-Instruct` already refuses **16/16**
harmful prompts at ceiling, leaving the adapter no headroom to add refusal; what
contrast exists is either sub-threshold (benign, Cliff 0.445 against a 0.474 bar) or
**wrong-signed** — the aligned model refuses *less* than the base on harmful prompts.
The gate's self-test passed on the same run, so this is a genuine non-validation
rather than a broken gate.

**Manual audit of the three aligned non-refusals under indirect pressure.** Reading
every trajectory rather than trusting the aggregate:

- **weapons** — a *soft refusal* our marker list missed ("I'm not sure I can help with
  that. It sounds like it could be used to make a bomb"). The graded instrument caught
  it (`p_refuse` = 0.936). A string-match false negative, not a behavioural failure.
- **fraud** — **genuine compliance.** Given a prefix-injection instruction, the aligned
  model wrote a complete phishing email. The base refused cleanly ("I can't assist with
  that request").
- **violence** — **genuine compliance.** Under a fiction framing, the aligned model
  named a specific poison and described administration. The base refused cleanly.

So the adapter **removes the base model's refusal on 2 of 8 jailbreak-framed prompts**,
with the base refusing both.

**This is n = 2, on one adapter, at BF16, and we state that here rather than only in
Limitations.** It is a case study, not a population estimate. We are not claiming this
adapter is broadly unsafe, that alignment LoRAs generally degrade safety, or anything
about its behaviour under quantization — which we did not measure.

## 6.3 It is not a hidden over-refuser either

A plausible reading of §6.2 is that the adapter trades harmful-refusal for exaggerated
safety. A supplementary control of **surface-harmful, actually-benign** prompts (§3.10)
did not clear the gate either: `p_refuse` 0.8118 base against 0.8554 aligned (Cliff's *d*
0.188), string-refusal 0/16 against 1/16 (*d* 0.062). **The null is informative because
the prompt set demonstrably discriminates** — the base model scores 0.1449 on plain benign
prompts and 0.8118 on these, a **5.60×** contrast on the same model. The instrument can
move; it did not. What the adapter does do is **diffuse the output distribution**: mean
token entropy runs **1.71× to 2.78×** base across the four subsets, Cliff's *d* 0.86–1.00
on every one. That is its largest coherent behavioural fingerprint, and it is not refusal.

## 6.4 Consequence: no precision comparison, and a prediction withdrawn

We had pre-registered a prediction that this adapter's behavioural retention at INT3
would rank highest of four adapters, anchoring a test of whether output SNR predicts
behavioural retention **across** a population spanning a 3.7× predictor range.

**The instrument did not validate, so no retention number may be computed and the
prediction is withdrawn rather than tested.** We report this rather than substituting a
weaker instrument or relaxing the gate to obtain a pass. Relaxing a threshold after
seeing the data is the precise failure the gate exists to prevent, and our own history
with it (`METHODOLOGY.md` M.6) is why we did not.

The withdrawal is not a quantization result. **No precision comparison was run on this
adapter**, and nothing here says anything about whether its behaviour survives
quantization — only that we could not certify an instrument capable of asking.

## 6.5 A second measurement proxy that fails to track what it names

`p_refuse` scores the base model at 0.812 on surface-harmful prompts while that model
complies with **16 of 16**, and 0.857 on the fiction-framed prompt where the aligned model
**complied**. In both directions the graded propensity tracks how harmful the prompt
*looks*, not what the model does. Within a fixed prompt set it remains a valid
across-condition comparison, so §6.2's verdict stands; its absolute level must never be
read as "probability the model refuses". **This is the paper's through-line at a second
level of the stack.** In §5.4 a weight-space proxy failed to track the behaviour it was
built to predict; here a behavioural proxy fails to track the behaviour it is named after.
Both were caught the same way: by validating the proxy against a contrast whose answer was
already known, and by reading trajectories rather than trusting an aggregate.
