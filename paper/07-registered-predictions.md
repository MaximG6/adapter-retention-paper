# Appendix: Registered predictions

*Every prediction registered before the run it concerns, with its outcome. The methodological practices that produced them are in `METHODOLOGY.md` in the accompanying repository, which is inside the same claim audit, count-word gate and cross-reference gate as this document.*

We pre-registered eleven predictions before the runs they concern: **P1 to P9**, nine in a
dated planning document, and **P10** and **P11** later, each in the `EXPERIMENTS.md` entry
for the run it concerns and before that run. **All eleven are below.** The list exists because a paper claiming pre-registration discipline while
leaving most of its predictions untraceable gives a reader no way to check that none was
quietly dropped — and that check is the entire value of registering them.

An earlier version of this appendix said "the complete list" over the P1–P9 table alone
and closed with "nothing is missing: nine were registered and nine appear," while the
body cited P11 twice and B.11 routed readers to P11.1, P11.2 and P11.3 — labels that
appeared nowhere in the document. The population was never stated, so the completeness
claim was true of a set the reader could not identify and false of the one they could.

Of the nine in the planning document, **two were confirmed outright** (P2, P6) and one
was confirmed on synthetic adapters but does not transfer to trained ones (P1). **The
remaining six were not confirmed as registered**, in four different ways. P4 appears in
that six because *some* clause of it failed; its other clause held, and the count is over
predictions while the taxonomy below is over clauses:

- **withdrawn on evidence before being tested** — P7, and the safety clause of P8;
- **superseded by a measurement bug in its own inputs** — the at-risk clause of P4;
- **never run, by a decision recorded with its reasoning beforehand** — P9;
- **untested because the adapters it needs do not exist publicly** — P3, P5, and the
  remainder of P8.

An earlier draft of §8 said "four", and its taxonomy silently omitted P3 and P5 — and
this appendix went on stating six while the body said four, so the paper announced its
own body was wrong and left it standing. That is precisely the failure this table exists
to make impossible, committed in the prose introducing the table, and it is recorded here
rather than corrected away.

| | prediction (final registered form) | outcome | where resolved |
|---|---|---|---|
| **P1** | Under `α = 2r`, weight-space fidelity rises as `r^{+1/4}` while subspace output fidelity falls as `r^{−1/4}` — the two spaces disagree in sign | **Confirmed on synthetic adapters; does not transfer to trained ones** | §4.3 |
| **P2** | Under fixed `α`, `SNR_w ∝ r^{−1/4}` and `SNR_out ∝ r^{−3/4}` | **Confirmed** (fitted −0.275 / −0.744 against −0.25 / −0.75) | §4.3 |
| **P3** | A calibrated absolute prediction of output SNR at `r=32` | **Untested.** Requires adapters matched on training across ranks, which do not exist publicly | §8.2, FW-2 |
| **P4** | Behaviour substantially preserved for adapters with output SNR > 1.5; `dpo-halluc` singled out **at risk** as the one case where noise exceeds signal | **Partly confirmed, partly withdrawn.** Behaviour preserved for all six at INT4 g128. The at-risk clause was **withdrawn**: it rested on an output SNR of 0.958 produced by an rsLoRA scaling bug; the corrected value is 3.757 and **no adapter in the set has output SNR below 1** | §5.1; withdrawal in `METHODOLOGY.md` M.4 |
| **P5** | Behavioural degradation orders across adapters as taboo < latentqa < dpo < safety, spanning 3.7× in output SNR | **Not tested.** Subsumed by the decision not to run the across-population test; two of the four adapters never received validated behavioural batteries | §8.2 |
| **P6** | INT8 survives essentially completely; if anything breaks it is per-channel INT4 or 3-bit | **Confirmed** — INT4 g128 99.2%, per-channel 77.2%, INT3 57.8% | §5.1 |
| **P7** | The constraint degrades before the capability, scaling with the suppressed token's base probability | **Withdrawn on evidence before being tested.** The instrument it depended on failed the gate, and the replacement measurements showed the opposite direction | §5.3, `METHODOLOGY.md` M.2 |
| **P8** | At INT3, retention ranks in output-SNR order; safety adapter > 85% | **Safety clause withdrawn** (its instrument did not validate); remainder untested | §6.4, §8.2 |
| **P9** | Across-population Spearman between output SNR and INT3 retention exceeds +0.6 | **Not run.** Decision recorded with its reasoning and its falsifier before the fact | §8.2 |

**Registered later, each before its own run.** These are not in the planning document
because the questions they answer did not exist when it was written; both were registered
in `EXPERIMENTS.md` with a falsifier before the measurement, on the same terms as P1–P9.

| | prediction (final registered form) | outcome | where resolved |
|---|---|---|---|
| **P10** | Quantizing `Δ` on its own grid rather than `W`'s puts `mean(\|Δ\|/s_Δ)` in 0.1–1.0, cosine above 0.99 for all nine, relative error under 0.15, and compresses the across-adapter spread below 1.1× | **Three of four clauses confirmed.** Clause 1's range was wrong and its direction right: measured 2.31–2.38, not 0.1–1.0 | §7 |
| **P11** | Equation 4's third licensing assumption: `P(δ<0)` within 0.5 ± 0.01 per module-instance (**P11.1**); `\|r(sign δ, u)\| < 0.01` (**P11.2**); the sign-aware flip prediction within 2% of the 50/50 two-tail average at every `t ≥ 0.005` (**P11.3**) | **All three confirmed**, and by margins of 42×, 9× and 3× on the registered bounds | §4.1, B.11 |

Two properties of this listing are the point of it. **First, the failures are legible**:
a reader can see that P4's headline clause did not survive its own input data, that P7
was abandoned rather than quietly reinterpreted, and that P10's first clause was off by
an order of magnitude in a direction that did not change its conclusion. **Second,
nothing is missing**: eleven were registered and eleven appear.
