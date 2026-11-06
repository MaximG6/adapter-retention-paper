# Abstract

*Draft. Written after Method, Results, Limitations and Conclusion, per the practice of
not letting the framing precede the findings.*

---

Merging a LoRA adapter and quantizing to INT4 at group size 128 delivers a weight update
with **cosine 0.14** to the one the adapter trained, at **7.5 times its magnitude**. We
lead with cosine because it does not depend on which tensor sets the quantization grid,
where the fraction of weights that looks changed runs from 1.1% to 85.5% (§3.3).

Across **six rank-32 adapters sharing one base model and one training recipe**, quantized
on the deployment path, the two sides of behaviour move differently. Elicitation
*capability* shows **no detectable loss**: retention 99.2%, enumerated 95% interval
**[90.7%, 107.6%]**, spanning parity and excluding losses beyond about 9%. The trained
*constraint* does move, and **tightens**: the adversarial leak rate falls from 16.7% to
8.3%, a paired difference of **+8.3 points** with an enumerated 95% interval of
**[+4.2, +12.5]** — four of the six adapters leak one fewer prompt and none leaks more,
on cells that are counts out of 8. An exact sign-flip permutation test gives **p = 0.125**:
the direction is consistent and n=6 cannot resolve it (§5.1). §3.7 defines behaviour as
two-sided, so a single "unchanged" would be wrong on both sides.

Both halves follow from one ratio, `|Δ|/s` — the adapter's per-weight magnitude against
the quantization step. Read per weight, a channel model with **no fitted parameters**
predicts each adapter's code-flip rate to within **2.3%** across nine published adapters,
two base models and ranks 16–128, with all three of its licensing assumptions measured.
Read on the adapter's `r`-dimensional active subspace, layer-output fidelity is
*measured* **6.2–16.5× higher** than weight-space fidelity — the span of B.13's
per-adapter values, not a prediction from the concentration constant. Weight-space
erasure and layer-output survival are one measurement read at two levels of the same map;
behaviour is a third level and follows from neither (§5.3).

Capability degrades at coarser grids — 77.2% at INT4 per-channel, 57.8% at INT3 — and
degrades **benignly**, the constraint holding throughout (§5.2). Weight-space measurement
cannot say **which** adapter survives. Across six adapters matched on rank, scaling, base
model, recipe and measured output SNR to within 3.3%, retention spans **28.7% to 86.4%**
on the pre-registered instrument (28.4–84.4% floor-corrected), and six of the seven pairs
the data can resolve run opposite to the predictor.

---

## Notes for revision

- **Length.** ~280 words. Priority if trimmed further: keep the capability/constraint
  split with both intervals, the channel model's 2.3%, and the predictive gap.
- The opening contrast mirrors **Figure 1** (twin panel) by design.
- "99.2% retention" is elicitation score relative to the same adapter's own BF16 score —
  stated precisely in §5.1. The abstract must not drift into implying a broader
  behavioural claim than §8.1 licenses, and must not restate 99.2% without the interval:
  the interval is what makes it a bounded non-detection rather than a measured equality.
- **The capability claim may not be stated as a claim about behaviour.** An earlier
  version read "no detectable change in their trained behaviour", which §3.7's own
  two-sided definition falsifies: the constraint side is measured and it moves. The
  non-detection is on capability alone, and the constraint result goes beside it.
