# Appendix A: `ar.predict` — computing effective adapter magnitude from a published checkpoint

*The tool exists because §4.1 identifies `|Δ|/s` as the quantity that governs retention,
and **no adapter card publishes effective magnitude**. It closes that gap for weights.
It does not close it for behaviour, and it says so in its own output (§A.4).*

---

## A.1 What it does and what it costs

Given a HuggingFace adapter identifier, `ar.predict` computes the adapter's per-weight
delta magnitude against the quantization step size of the base model it targets, and
reports the weight-space consequences.

It needs **no GPU, no training, and no base-model download.** Base weights are
range-read from the remote safetensors shards — roughly **150 MB of network** rather
than the 16 GB the full model would require — and adapter tensors are small. A run takes
about 30 seconds.

```bash
python -m ar.predict --adapter adamkarvonen/Qwen3-8B-taboo-smile_50_mix \
                     --bits 4 --group-size 128
```

## A.2 Output

```
  effective magnitude  mean|delta|     9.747e-05
                       mean step s     9.222e-03
                       mean|delta|/s   0.01060

  predicted bit-flip rate            0.0106   [0.0090, 0.0122]
  predicted cosine(delta, delta_eff) 0.1427
  predicted weight-space SNR         0.1442
  predicted layer-output SNR         1.749   [1.487, 2.011]  +/-15%

  NEAR-TOTAL WEIGHT-SPACE EROSION: the deployed weights barely move.

      module   mean|d|/s     flip   cosine     amp  SNR_out
------------------------------------------------------------
      v_proj     0.00874   0.0087   0.1300   11.22    1.471
      k_proj     0.00979   0.0098   0.1335   11.23    1.513
      o_proj     0.01068   0.0107   0.1401   11.22    1.588
      q_proj     0.01088   0.0109   0.1426   11.22    1.617
     up_proj     0.01205   0.0121   0.1555   11.21    1.765
   gate_proj     0.01568   0.0157   0.1856   11.19    2.114
   down_proj     0.00635   0.0064   0.1115   19.39    2.175

  [elided: a note on what the two weight-space columns assume, then the two
   banners of A.4, which print here on every run and are quoted there in full]
```

**In weight space the module ordering is mostly a magnitude effect and partly a
tail-shape one** (§4.3, B.5): `gate_proj` has the largest `|Δ|/s` and the highest cosine,
`down_proj` the smallest and the lowest, and in *this* table the cosine column happens to
be monotone in the ratio column. It is not monotone in general, and an earlier draft
called the effect "entirely" magnitude two bullets above the measurement that refutes it.
Pooled over the nine adapters (B.5), `k_proj` has the second-highest flip rate and the
fourth-lowest cosine while `up_proj` has the fourth-highest flip and the second-highest
cosine — which under `cosine = sqrt(τ · |Δ|/s)` requires `k_proj`'s τ to sit **17.5%
below** `up_proj`'s, and the τ bullet below measures exactly that spread. Magnitude sets
the scale; tail shape reorders within it.

**In output space the ordering reverses at the bottom, and the reason is architectural.**
`down_proj` has the *lowest* weight-space cosine (0.1115) and the *highest* predicted
output SNR (2.175), because its input dimension is 12288 rather than 4096, so its
amplification is 19.39 against 11.2. The two columns answer different questions and
disagree about which module fares worst; a reader taking "retains least" from the ratio
column and applying it to the layer's output would get it backwards. This is the same
distinction as §4.4 and §5.4, visible inside a single table.

**Three notes on reading these numbers against the paper's.**

- **They are predictions, not the paper's measurements, and they do not match to the last
  digit.** The tool samples three layers of 36 and averages seven modules unweighted; the
  weight-space runs measure four layers and 28 module-instances (B.1). Predicted flip
  here is `0.0106`, against a
  measured `0.01093` for the same adapter in B.1 — **3.0% apart**, which is within the
  tool's stated ±15% band but *outside* the 2.3% figure quoted for the closed form. The
  2.3% is the model's error against a full measurement; 3.0% is what layer sampling adds.
- **`predicted layer-output SNR 1.749` is not the `1.6286` B.13 attributes to this
  adapter.** B.13's value is *measured*, by projecting onto an orthonormal basis of Δ's
  right singular vectors; this one is *predicted* from Equation 5. They differ by 7.4%,
  which is the amplification law's error on this adapter, and PG-1's 1.6200–1.6728 range
  is a range of measured values only.
- **`predicted weight-space SNR` is `cos / sqrt(1 − cos²)`, and that is not the
  definition B.13 measures.** Here it is the ratio of `Δ_eff`'s component *along* `Δ` to
  its component *orthogonal* to `Δ`, computed from the predicted cosine: at cosine 0.1427
  it gives 0.1442. B.13 reports `||Δ|| / ||Δ_eff − Δ||`, signal over total error,
  measured per layer and averaged, and that is the quantity the abstract's 6.2–16.5×
  amplification range is denominated in. **The gap between what this tool prints and what
  B.13 measures is 0.1442 against 0.1341, 7.5%.** An earlier draft quoted 3.4% here,
  which is the gap between the two *formulas* evaluated at the same measured cosine
  (0.1387 against 0.1341) — a real quantity, and not the one a reader of the tool's
  output is asking about. Every number the tool prints is a claim, and substituting a
  friendlier recomputation for the printed one is the failure this appendix exists to
  prevent. The formulas agree closely because both reduce to approximately `cos` when the
  projection coefficient is near 1 and `cos` is small; the remaining 4.1% is the cosine
  prediction's own error. Neither validates the other.
- **The `cosine` column is not the fixed-`τ` formula A.3 tabulates.** The tool computes
  `sqrt(mean(Δ²) / (s · mean|Δ|))` per module, which is `sqrt(τ_module · |Δ|/s)` with each
  module's *own* tail-shape statistic. Backing it out of the printed columns gives
  1.82–2.20 across the seven modules, against the 1.5962 the synthetic Gaussian sweep
  measures. A.3's table describes **Figure 5's** predictor, which deliberately uses one
  fixed constant so that it has no per-module free parameter; applying that formula to
  the row above gives 0.118 where the tool prints 0.130. Two different predictors, and
  an earlier draft implied they were the same one.
- **The `amp` column uses each module's measured error concentration, not the `1 + c/r`
  form.** Equation 5 with `c = 0.87` and `r = 32` gives `sqrt(128/1.0272) = 11.16`; the
  tool prints `11.22`, implying a measured concentration of 1.017 rather than the fitted
  1.027. Both are in the paper: the closed form is the claim, the measured value is what
  the tool has available per module. **The 1% gap is not unexplained.** `c ≈ 0.87` is
  fitted across ranks 16–128; on this adapter at `r = 32` the measured concentration
  corresponds to `c = 0.017 × 32 = 0.54`, so the closed form over-states concentration
  here and under-states amplification by 0.5%. The abstract's 6.2–16.5× range is the span
  of B.13's *measured* per-adapter ratios and does not pass through `c` at all.

## A.3 Accuracy

Validated against directly measured records on **nine** published adapters (the
validation figure at the end of this appendix):

| quantity | prediction | max relative error across nine adapters |
|---|---|---|
| code-flip rate | `mean(min(\|Δ\|/s, 1))` | **2.3%** (safety) |
| cosine | per module, `sqrt(τ · \|Δ\|/s)` with τ = 1.5962, then averaged | **10.4%** (latentqa) |

The prediction is a closed form with no fitted parameters (§3.5), so this is
out-of-sample in the only sense available: nothing about these nine adapters was used to
construct the model. The cosine row carries one measured constant, the tail-shape
statistic `τ = mean(Δ²)/mean|Δ|²`, measured at **1.5962 on the synthetic Gaussian sweep**
and flat there across three decades of adapter magnitude.

**That constant is a property of the synthetic generator, and trained adapters do not
satisfy it.** 1.5962 is 1.6% above `π/2 = 1.5708`, the exact value for a Gaussian, which
is what the sweep draws. Backing `τ` out per module from A.2's table gives **1.82–2.20**
on a real adapter — 16% to 38% higher, because products of trained `B·A` matrices are
heavier-tailed than products of Gaussians. Since `cosine ∝ √τ`, that alone is a 7% to 17%
under-prediction, which is the right size to be most of the 10.4% in the table above; the
code-flip row carries no such constant, which is why it is at 2.3%. **The cosine
predictor's error is dominated by a constant fitted on a generator its targets do not
match**, and stating that is more useful than the error figure alone.

**These errors are for the quantity the model predicts: the adapter's own contribution
under a fixed grid** (§3.3). A deployment toolchain also moves the grid, which is a second
effect the closed form does not model; read as a prediction of the combined outcome its
error is 32–47%. The tool prints this scope in its own output.

> **An earlier version of this table read "six adapters" and "cosine 5.0%", and both were
> wrong; the panel itself printed "max error 0.0%".** Three numbers for one quantity, none
> of them the measured 10.4%. The figure plots nine. The cosine panel was computing its "prediction" as
> the measured projection coefficient `<Δ_eff, Δ>/||Δ||²` divided by the measured
> magnitude ratio `||Δ_eff||/||Δ||`. Those differ by exactly a factor of
> `cos(Δ, Δ_eff)`, so their quotient **is** the cosine — it plotted
> cosine against cosine, drew a perfect line, and printed *max error 0.0%*. It rendered
> without error for the whole draft and every cross-check passed, because every value in
> it was correct. It was simply not a test. The panel now uses the channel model's own
> cosine, and the figure's cross-check asserts that prediction and measurement **differ**,
> so the vacuous form cannot come back silently (`METHODOLOGY.md` M.3, `METHODOLOGY.md` M.5).

**It reads `use_rslora` from each adapter's config rather than assuming a convention.**
For a rank-128 rsLoRA adapter the two conventions differ by `√128 ≈ 11.3×`, which is
enough to move an adapter from worst to best in a nine-adapter ranking (`METHODOLOGY.md` M.4). The
computed delta is verified against peft's own `merge_and_unload` by a ground-truth
fixture (§3.8).

## A.4 What it cannot do, stated in its own output

The tool prints the following, unconditionally, on every run:

> **LIMIT OF THIS TOOL, measured not hypothetical.** Six adapters matched on rank,
> scaling, base model and training recipe, whose output SNR agreed to within 3.3%,
> showed behavioural retention at 3-bit spanning 28.7% to 86.4% (28.4% to 84.4% with the
> instrument's floor subtracted; the split is 2 of 6 below half uncorrected, 3 of 6
> corrected). The outcome varied 27x more than the predictor did, once the outcome's own
> measurement error is netted out of its spread — 30x on the raw comparison, which is a
> noisy outcome against a deterministic predictor. Of the 7 adapter
> pairs whose difference was statistically resolved, 6 ran OPPOSITE to output SNR.
>
> So: these numbers do not discriminate between similar adapters. If you are choosing
> between two adapters of comparable rank and magnitude, this tool cannot tell you which
> will survive quantization better, and a difference it reports between them carries no
> information. Whether it discriminates ACROSS dissimilar adapters is untested.

It also prints, unconditionally:

> **WHAT THESE NUMBERS SCOPE TO.** The flip rate above is THE ADAPTER'S OWN
> CONTRIBUTION: given one grid derived from the base weights, the fraction of codes this
> delta pushes across a boundary. That is what the model predicts, parameter-free, to
> within 2.3% on nine published adapters.
>
> A deployment toolchain also recomputes the grid from the merged tensor, and the grid
> then moves under almost every weight. That is a SECOND effect this model does not
> describe. With both acting, measured code flips run 1.5–1.9× the number above and
> 83.6–87.4% of dequantized VALUES differ rather than the 1–15% of codes. So: read the
> flip rate as what the adapter did, not as the fraction of your deployed checkpoint that
> differs from the base.

We include both rather than a softer caveat because each failure is measured, not
anticipated (§5.4, §3.3), and because a tool that reports a number invites the inference
that the number ranks things. **The honest use is as a description of what happens to
stored weights, at a single adapter, under a grid that isolates the adapter's own
contribution — not as a comparison between adapters and not as a deployment forecast.**

## A.5 Interpreting the output

The two banners in A.4 are the tool's own statement of what it does and does not
support, and they are quoted above verbatim rather than paraphrased here. Two additions
they do not carry. **The tool answers only the merged case**: §2.5
measures the unmerged one on 756 records and finds it entirely different, `|Δ|/s` rising
from 0.011–0.149 to 2.31–2.38 and cosine from 0.14–0.51 to 0.9948–0.9952. And **the
predictive-gap limit is about discrimination, not level** — output SNR fails to separate
adapters whose predictor range is 3.3% wide, which is range restriction, and is
compatible with the same quantity setting the absolute level, which is what §4.4
measures and what the amplification law is for (§5.4).

## A.6 Reproduction

Full commands and pinned versions in Appendix D. The tool alone:

```bash
pip install -r requirements.txt          # no torch CUDA build needed for ar.predict
PYTHONPATH=src python -m ar.predict --adapter <hf-id> --bits 4 --group-size 128
```

Supported: `--bits {3,4,8}`, `--group-size {any positive int, or -1 for per-channel}`,
`--scheme {asymmetric, symmetric_gptq, symmetric_awq}`, `--base-model` to override the
adapter's declared base.
