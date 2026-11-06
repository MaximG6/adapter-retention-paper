# Methodological practice

*Companion to “Weight-Space Erasure Without Behavioural Collapse in Quantized LoRA Adapters”. Every entry below is evidenced by an error of ours that measurement caught before publication, and each says what it changed. They are here rather than in the paper because they are a different paper's argument — the paper's appendix keeps the registered predictions and their outcomes, which is the part its own claims depend on.*

*This document is inside the paper's checks. `analysis/audit_draft_numbers.py` verifies its numbers against `results/raw/**`, `analysis/countcheck.py` resolves its count words against the structures they count, and `analysis/xref.py` resolves its references to the paper against the paper's structure and the paper's references here against these headings. A number that leaves the PDF does not leave the audit.*

---

## M.1 Nothing checks meaning, and a green verification block is evidence about coverage


**Claim.** Every automated check in this project compares a number to a number. The
claim audit compares a printed value to a recomputation from raw; the cross-artifact
check compares two printed values; the table check compares two cells; the
cross-reference gate compares a reference against the set of labels that exist. None of
them checks *meaning*. **A green block is evidence about what the checks cover, not
evidence that the document is right**, and the gap between those two is where the
remaining errors live.

**Evidence.** Five defects shipped in a built PDF while every check passed. All five were
found by reading.

| defect | why every check passed |
|---|---|
| 8 of 54 renumbered references resolved to the wrong section | the gate proves a target *exists*; §3.10 was mapped to §3.7, which exists |
| §2 asserted "all three concern activations" two paragraphs below a sentence explaining one of them does not | no check reads a sentence against the paragraph above it |
| the body said six adapters where the abstract and tables said nine | the count is a row *tally*, not a printed value with a recomputation |
| §5.1's heading said "monotone" directly above a caption saying individual adapters are not | headings and captions are not quantities |
| the tool's validation figure plotted cosine against cosine and printed "max error 0.0%" | every plotted value was verified against raw and every one was correct; nothing asked whether the two sides could differ (M.5) |

The renumbering case is the sharp one, because the fix made it worse: before, the
references dangled and a reader could see it. **A checker that turns a dangling
reference into a resolving but wrong one has removed the symptom and kept the disease.**

**Three holes, named as holes.** `refs.bib` is under no gate — citation identifiers and
the claims attributed to them are verified by hand, in session, and nothing enforces it.
Prose-to-table consistency is under no gate: a table's row count disagreeing with the
sentence introducing it passes everything. Semantic reference validity is under no gate;
what exists instead is a one-off manual audit of the rewritten references, which is not a
standing check and will not fire next time.

**This entry organises the ones that follow.** Each of them closes one numeric hole, and
none of them closes the semantic one: the generated-view rule (M.4) verifies that derived
documents match raw, and the shared-code-path rule (M.5) verifies that a check does not
share its subject's assumptions and has failed on a known-bad input. Both are worth
having. Neither would have caught any row of the table above.

*This paragraph said "All three are worth having" after a list of two, for one round,
inside the entry arguing that count-versus-prose defects pass every check. It is now
under one: `analysis/countcheck.py` resolves each count word in the body against the
structure it counts and fails the build on a disagreement.*


## M.2 Validate an instrument against a known contrast before registering a prediction on it


**Claim.** An instrument that has not been shown to separate a contrast whose answer
you already know cannot support a prediction about a contrast whose answer you do not.
The check is cheap and it belongs before pre-registration, not after a null.

**Evidence.** Our first capability probe was a forced-reveal continuation — "The secret
word is" — and it returned near-identical values on models with obviously different
behaviour. The probe asks the model to complete the single frame its training
suppresses, so it measures the constraint at maximum strength and reads approximately
zero whether or not the model knows the word. We had registered a prediction on it.

We now require every instrument to separate `aligned_bf16` from `base_bf16` — a
contrast confirmable by reading the text — before it may be used in a precision
comparison (§3.9). Applying this rule cost one instrument (the reveal probe,
deprecated), rescued two (the graded constraint trace and fixed-guesser elicitation,
both of which passed), and caused one registered prediction to be **withdrawn on
evidence rather than tested** (Appendix C of the paper).


## M.3 A validation gate must itself be tested against something already known to be broken


**Claim.** A gate is an instrument. Build it, then verify it rejects a case you have
independently established is bad — otherwise you have a filter of unknown selectivity
sitting at the most consequential point in the pipeline.

**Evidence.** Our first gate passed instruments on `|Cohen's d| > 1.0 OR ratio > 3.0`,
and it **certified the reveal probe** — the one instrument already documented as
broken. Two independent defects:

- Cohen's `d` returns `inf` when pooled variance is zero and the means differ, and
  non-finite was being read as "enormous effect" rather than "undefined".
- The disjunction. Either clause alone is satisfiable by arithmetic: the probe's
  "5757× ratio" was a ratio between 10⁻⁶ and 6.4 × 10⁻³, two numbers that are both
  approximately nothing.

The rebuilt gate is conjunctive on a rank-based effect size (Cliff's delta, always
finite), a ratio, and an absolute floor, and it ships with a self-test asserting that
it rejects the reveal probe. The self-test is run on every invocation.


**The rule caught a defect in the check written to enforce the rule.** A table-level
cross-artifact check was added after two tables were found printing the same three
intervals with different values. It ran clean and reported no disagreement — while the
disagreement was still live in the shipped PDF. LaTeX writes `int4\_g128` and the
markdown writes `int4_g128`; the label-normalising sweep stripped `\word` escapes but
not `\_`, so the two labels never joined and every comparison silently had one side.
It looked correct, exited zero, and would have produced a third consecutive green block
over the same defect.

It was caught only because the rule in this entry required it to fire on a case whose
answer was already known — the historical disagreement, reintroduced deliberately. This
is the sharpest form the evidence for this practice can take: not a check that failed,
but **the rule catching a fault in the check written to enforce it**. The previous
sharpest was the vacuous-comparison guard, which had the same shape one level down.


## M.4 An append-only record needs a generated view, or it will leak superseded values


**Claim.** An append-only log is the right format for a research record: it preserves
what was believed and when, and corrections are added rather than substituted. That
property is also a hazard. A superseded number stays on the page looking exactly as
authoritative as it did before it was corrected, and anything drafted *from the log*
inherits it. **Derived documents must regenerate from raw records, never from the
notebook.**

**Evidence.** Our notebook records, plainly and in the correct place, that a scaling bug
made one adapter's weight delta 11.3× too small and that every affected row in four
earlier entries is superseded. The correction is unambiguous. The paper's results
section was nonetheless drafted with the *old* values, because it was written from the
entry that contained the original measurement rather than from the entry that corrected
it — and the stale range reached the abstract.

It was caught by building the appendix tables as a **generated** artifact: a script that
emits every table directly from the raw records, whose first run disagreed with the
draft. Extending the same approach to a claim-by-claim audit of the remaining sections
then caught a second instance in a table we had already corrected once — an output-SNR
row still carrying the pre-fix value.

**The second instance is the more instructive one, because a stale number there carried a
false sentence rather than a wrong digit.** The draft asserted that one adapter had
output-space noise exceeding signal. At the corrected value that adapter's output SNR is
3.76 rather than 0.96, and **no adapter in the set has noise exceeding signal** — a
statement our own notebook had recorded explicitly, in the entry that made the
correction, two weeks earlier.

**That claim also had institutional momentum behind it, which is why it survived.** It
was not an incidental number. It had been singled out for emphasis, written into the
plan as a *registered prediction* — one adapter "singled out as at risk, being the one
case where output-space noise exceeds signal" — and it was the most quotable finding in
that section at the time. When the underlying value was corrected, the prediction was
formally withdrawn; the *framing* it had established was not, and the framing is what the
draft reproduced.

There is a sharper detail. The same request that asked us to foreground that result also
asked us to check whether the adapter's unusual scaling was deliberate — and it was that
check which uncovered the bug and invalidated the result. **The instruction to emphasise
a finding and the instruction that destroyed it arrived together**, and only the first
left a durable trace in the prose.

The general form: **a number that has been promoted is harder to retract than one that
has not**, because promotion creates dependent sentences elsewhere. When a value is
corrected, the correction has to propagate to everything the old value licensed, not just
to the cell it occupied. Digit-level staleness is recoverable at proof stage.
Claim-level staleness reads as a normal, confident sentence and survives review.

Two properties made the difference, and both are cheap:

1. **Generated, not transcribed.** Tables and figures are emitted from raw by script;
   the only hand-written numbers in the paper are ones an audit re-derives.
2. **The audit is a regression test, not a one-off.** Each claim the prose makes is
   encoded with its expected value and recomputed from raw, so a later change to the
   records fails loudly instead of silently disagreeing with the text.

A useful diagnostic: our public README carried the correct values throughout, because it
is *rewritten* at each gate rather than appended to. **The format that preserves history
is the one that leaks stale values; the format that overwrites is the one that stays
current.** A project wants both, and should not confuse which is which.


## M.5 A check that shares an assumption or a code path with the thing it checks is not a check


**Claim.** The value of a verification lies entirely in what it does *not* share with its
subject. A check built on the same assumption will confirm that assumption; a check
calling the same code will confirm that the code is deterministic. Neither tests what it
appears to test, and both report success. **Independence has to be structural, and it has
to be deliberate, because the convenient implementation is almost always the dependent
one.**

We state this as a rule rather than an anecdote because we have two instances from
opposite ends of the project, arising in completely different work.

**Instance one, measurement (Phase 0).** To measure how a low-rank adapter behaves on its
own subspace, we drew probe vectors as `coef @ A` — through the adapter's own factor
matrix. The probes therefore carried covariance `AᵀA` and inherited `A`'s spectrum: the
instrument shared its geometry with the object under test. It returned amplification
values of 33–75 that were nearly rank-insensitive, and we concluded that the
`√(d_in/r)` law had failed. It had not. **The probe had imported the very structure it
was supposed to interrogate**, and an orthonormal basis of `Δ`'s right singular vectors —
constructed to be uninformative about `A`'s spectrum — recovered the law to within 1% at
r=32 (§4.4).

**Instance two, verification (write-up).** Building the figure cross-checks, the obvious
implementation was to import the figure's own data loader and compare. That would have
been worthless in precisely the case we needed it for: the predictive-gap figure's
marking bug lived in logic the figure owned, and a shared loader would have reproduced
the same 2 marked adapters and reported agreement. The checker therefore re-reads the raw records by a
separate route, and recomputes the resolvable-pair set independently. That is why it
disagreed, and why the disagreement was informative.

**The two cases have the same shape at different levels.** In one, an instrument shared
geometry with its measurement target. In the other, a verifier shared code with its
verification target. Both produced confident, plausible, wrong agreement, and in both the
dependent version was the one that was easier to write.

**The practical test we now apply**: *if the thing being checked were wrong in the way I
most fear, would this check still pass?* If the answer is yes, the check is measuring
something other than what it claims. This is the same question the instrument gate asks
of behavioural probes (M.2) and the gate self-test asks of the gate (M.3), applied to
verification generally.

### The verification half: a check is not trusted until it has failed on a known-bad input

Structural independence is the **design** principle. It is not self-verifying, because an
implementation that fails to achieve it looks exactly like one that achieves it — both
print success. **The way you establish that a check is independent is to feed it an input
you have independently established is bad, and require it to fail.**

We applied this three times before naming it, and in all three cases the check looked
correct by inspection and only the known-bad input distinguished working from vacuous:

| check | known-bad input it must reject | what would have happened without the test |
|---|---|---|
| the instrument gate (M.3) | the deprecated reveal probe, already documented as broken | the gate had already certified it once under a disjunctive rule |
| the figure cross-checker | the actual predictive-gap pair values (2 vs 4 marked adapters) and the dissociation figure's estimator split (0.7810 vs 0.7716) | a checker sharing the figures' loader would have reproduced the bug and reported agreement |
| the vacuous-comparison guard | a call passing one expression twice | the first implementation matched the wrong AST node, **never fired, and printed PASS on the exact bug it was written to catch** |

The third is the sharpest, because the guard was written specifically to enforce this
section and was implemented in violation of it. It was not distinguishable from a working
guard by reading it. It was distinguishable only by running it against a case whose answer
was already known.

**Both halves are needed and they are cheap.** Design for structural independence;
verify it by requiring a failure on a case you already understand. A check that has never
failed has not been shown to be capable of failing.

*(A caution that follows: this makes "our checks all pass" a much weaker statement than it
appears, unless each check has a recorded known-bad case it rejects. We keep those as
tests — `tests/test_figcheck.py`, `instrument_gate.py --self-test` — so the property is
re-verified on every run rather than asserted once.)*

### A comparison can be vacuous without any value in it being wrong

The instances above are guards with a wrong model of the world. **This one is a check
with no model at all**, and it is the sharpest form the failure takes.

The `ar.predict` validation figure plots predicted against measured on two panels.
Its cosine panel computed the "prediction" as the measured projection coefficient
`<Δ_eff, Δ>/||Δ||²` divided by the measured magnitude ratio `||Δ_eff||/||Δ||`. Those two
differ by exactly a factor of `cos(Δ, Δ_eff)`, so their quotient **is** the cosine.
**The panel plotted cosine against cosine.** It drew a perfect diagonal,
printed *max error 0.0%*, and shipped in the built PDF for the whole draft.

Every cross-check passed, and correctly: the figure's guard verifies that every plotted
value matches an independent recomputation from raw, and every plotted value did. The
values were right. The *comparison* carried no information, and nothing in the project
was looking at that. A prose table beside it claimed 5.0%, so the same quantity appeared
as 0.0%, 5.0% and — once actually computed — **10.4%**.

**The general form: a check that verifies its inputs is not the same as a check that can
fail.** Ask of any comparison whether the two sides could differ *in principle*, before
asking whether they do. The figure guard now asserts that prediction and measurement
differ by more than machine precision, so the vacuous form cannot return silently, and
both error figures are registered claims so the figure and the prose cannot drift apart
again.

The nearest relative is row 3 below — the vacuous-comparison guard that passed one
expression twice — which is the same shape one level down: a check on checks, itself
unable to fail.

**Seven instances, seven unrelated causes, no shared mechanism.** Each row is a guard
whose own model of the world was wrong, and the last column records whether it fired when
it should not have or failed to fire when it should.

| # | guard | its own defect | direction |
|---|---|---|---|
| 1 | adapter-config audit | expected default guessed (`None`) where the library uses `16` | false positive |
| 2 | figure coverage guard | one call covering six values counted as one | false positive |
| 3 | vacuous-check guard | matched the innermost AST call, not its own | false negative |
| 4 | cross-reference checker | regex required whitespace where the text has a period | false positive |
| 5 | appendix-letter mapper | a guard skipping subsection refs also skipped every reference ending a sentence | false negative |
| 6 | appendix-letter mapper | ran after the section pass and rewrote references that pass had just created | false positive |
| 7 | cross-reference gate | keyed on the literal word "Appendix", so a bare `D.1.2` was not a reference to it | false negative |

This is now **the most-repeated failure in the project** — more frequent than any error
in the science itself. What the seven share is only the consequence: **a check whose model
of the world is wrong is worse than no check, because it consumes the attention a working
check would receive and teaches its author to discount it.** In every case the fix was to
correct the guard's model — read defaults at runtime, count what a call covers, match the
right node, match the actual heading format — never to loosen the threshold, which is the
tempting move and the wrong one.

The uncomfortable corollary: **guards are code, and this project's guards have had a
higher defect rate than its measurements.** That is an argument for testing them against
known-bad input (M.3), not for having fewer of them.



## M.6 Price the caveat against the measurement that would remove it


**Claim.** When a limitation is about to be written into the text, ask what it would cost
to *eliminate* rather than describe. Careful wording is the reflex, and it is often more
expensive than the experiment it substitutes for — because a caveat is permanent, appears
everywhere the claim appears, and weakens the claim forever, while the measurement is
paid for once.

**Evidence.** Our headline paired "98.8% of stored weights unchanged" with "99.2% of
behaviour retained." The first was measured on **three** adapters, the second on **six**.
The obvious fix was a sample-size statement at each of the four places the pair appears.

Instead we checked the cost of closing it. All six adapters were already cached, the
measurement is inference-only, and a prior run of the identical configuration had taken
68 seconds. Running the three missing adapters cost **about three and a half minutes**,
and the headline became **98.9% / 99.2% on the same six adapters** — no caveat needed
anywhere.

**The comparison is stark enough to be worth stating as a ratio:** three and a half
minutes of compute against four hedged repetitions of the paper's central claim, carried
permanently.

**The episode also shows why the caveat would have been the worse outcome even if
honest.** Closing the gap surfaced a *second* mismatch hidden underneath the first: the
old figure had silently pooled one adapter's 36-layer run with the others' 4-layer runs,
so the population differed in layer coverage as well as in size. A caveat about sample
size would have made the paper scrupulous about one mismatch while a second sat
undisclosed beneath it. **Describing a limitation can conceal its neighbours; removing it
cannot.**

This is not an argument against caveats — §8 is long and every entry in it is load-
bearing. It is an argument for pricing them first. The test: *if this limitation were
someone else's, what would I ask them to run?* If the answer is under a day, run it.


## M.7 Tooling reports success on the operation, not on the outcome


**Claim.** A zero exit code means a command completed, not that it accomplished anything.
In this ecosystem the gap between the two is wide enough to be a standing hazard, and it
is widest exactly where the operation is bulk data movement — the case where verifying by
eye is least practical and most necessary. **Check what landed on disk, not what the tool
said.**

**Evidence, two instances, neither detectable from the exit status.**

*Downloading model weights.* `snapshot_download` returned **exit 0** while leaving five
**0-byte files** in the cache. Nothing in the output distinguished this from success. The
failure surfaced only downstream, as a model that would not load, and was diagnosed by
listing file sizes. Our download path now fetches file-by-file and asserts each size
against the Hub's own metadata before proceeding.

*Cloning the repository.* On Windows, `git clone` of this repository fails on record paths
of 157 characters. The output is:

```
fatal: cannot create directory at 'results/raw/...': Filename too long
warning: Clone succeeded, but checkout failed.
```

**The line that says "Clone succeeded" comes after the line that says it failed**, and
the working tree is left partially populated. A reader skimming for the word "succeeded"
proceeds with an incomplete repository. This was found by running the reproduction
appendix against a fresh clone rather than the working tree.

**What the two share** is that the tool correctly reports on the *operation it performed*
— a request was issued, a clone object was created — while the *outcome the user wanted*
did not occur. Both were caught the same way, and it is the only way that works: **inspect
the artifact, not the return value.** Our download path asserts byte counts; our release
check regenerates every derived document and requires byte-identical output.

The positive form of this practice is worth stating, because it is what we now close a
release on: **regenerating the committed tables, prompt sets and README from the committed
records and getting byte-identical files** is a statement about the artifact rather than
about any command's exit status, and it is the strongest such statement available.



## M.8 The explanation of a measurement is a claim, and nothing was checking explanations

**Claim.** A number can be right while the sentence explaining it is wrong, and every
check this project built reads numbers. Claim audits compare printed values against raw
records. Cross-artifact checks compare one occurrence against another. Count-word gates
resolve cardinals against structures. **None of them can see a causal account**, and a
causal account written in the same session as the measurement is at its most persuasive
and least tested.

**Evidence, two sentences attached to one correct measurement.** B.11 measures the
within-bin position distribution over 42 module-instances. Every number in it was right
through four review rounds. Two sentences about those numbers were not.

*The mechanism.* We wrote that the exact-zero mass is Equation 2 pinning each group's
extrema onto codes `0` and `2^b−1`. An external reader did the arithmetic we had not:
that mechanism puts `2/128 = 1.56%` of weights on a boundary and we measured 0.20%, eight
times over. Only 5% of the exact-zero mass is extrema at all. What it actually is: base
weights are bf16, a group of 128 holds about 121 distinct values, and `w/s + z + 0.5`
lands exactly on an integer for roughly one in five hundred of them — a perturbation four
orders of magnitude below bf16's own resolution removes 99% of it.

*The replacement, which was also wrong, and this is the sharper half.* Having refuted the
boundary account we wrote a new one in the same session: because `z` is rounded, the
extrema land on the *centres* of those codes, `u = 0.494` measured, so they are pinned to
the safest position in the bin rather than the most dangerous one. **A mean of 0.494 is
what a uniform population gives.** It refutes the boundary account and licenses nothing in
its place, and we quoted it as though it licensed the opposite. Measured a round later,
when a second reader asked for the dispersion: SD 0.2887 against the uniform
`1/√12 = 0.2887`, IQR 0.500 against 0.500. **The extrema are not pinned anywhere.** The
practice below says an explanation gets a falsifier and a control — and the explanation
written to *replace* a falsified one is the one most in need of both and least likely to
get them, because the work feels finished when the wrong account has been removed.

*The corroboration.* We observed that the residual sub-uniformity implies Equation 4
should over-predict slightly, that B.2's measured/predicted is below 1 for all nine
adapters, and reported the agreement. Checked: `F_u(t)/t` reads 0.985 over most of the
range the adapters occupy, so we inferred a near-constant 1.3–1.5% over-prediction for
**every** adapter. The observed over-prediction is 0.1% for each of the taboo six and
2.3% for the safety adapter, and the ordering does not track. Sign agreement, magnitude
off eightfold, ordering uncorrelated. **That is not corroboration**, and the paragraph
was withdrawn rather than hedged.

*And the withdrawal was not the end of it, which is the part worth keeping.* Withdrawing
a claim leaves the discrepancy standing, and a standing eightfold discrepancy between two
of your own appendices is a defect, not a resolution. A second external reader said so.
Measured (EXP-052): the departure is **a function of `|Δ|/s`, not a constant** — true
flip over `min(t,1)` runs 1.12 at `t = 0.0024`, 0.97–0.98 through the middle and 0.95 at
`t = 0.124` — so a single licensing budget was never the right *shape* of statement, and
both appendices were right about their own populations. There is a second reason the two
never reconciled: B.11's argument runs on the two-sided `u` proxy and B.2's numbers are
the actual integer code flip, and those differ by 1.7% on one population. **Withdrawing a
wrong explanation and measuring the right one are different acts, and we did the first
and stopped.**

**What makes this different from the other entries.** The others are failures of
verification — something was not checked, or was checked by a route that shared the defect.
This one is a failure of *scope*: the verification was complete for the class of thing it
covers, and the defective claim was outside that class. A green block over three hundred numeric
claims says nothing about the sentences between them, and we had been reading it as though
it did (M.1).

**The practice.** *When a measurement is explained, the explanation gets a falsifier and a
control, the same as a prediction.* Three controls settled both sentences here and each is
a few lines: where do the allegedly-pinned weights actually sit; how much of the mass is
the alleged cause; does the effect survive a perturbation that a structural cause would be
immune to. The cost was minutes. The cost of not doing it was four rounds of a wrong
mechanism in a section whose entire purpose is that a structural worry was taken
seriously.

**What we do not claim.** We have no gate for this. Explanations are prose and we are not
going to regex them. What we have is a rule — an explanatory sentence in a results section
names its control — and two worked instances, which is weaker than every other entry here
and is marked as such rather than dressed up. M.9 gates the *retraction* of a wrong
explanation, which is a different and much easier problem.

---

## M.9 A correction lands where it was discovered, not where the claim is asserted

**Claim.** When a measurement retracts a claim, the retraction gets written into the
appendix that did the measuring. The two or three body sites that assert the retracted
claim are not in front of you at that moment, and nothing in the build is looking for
them. **The result is a document that argues against itself**, with the correct version in
the appendix and the retracted version in the abstract.

**Evidence.** Four findings in one review round, all this shape:

| retracted where | still asserted at | rounds it survived |
|---|---|---|
| "roughly 16 independent units" (B.12) | §3.11, §9, and four docstrings | 3 |
| "no detectable change in trained behaviour" (§3.7's own definition) | abstract, introduction, Figure 1's caption, Conclusion | 2 |
| the completeness of the registered-prediction table (P11 exists) | Appendix C's closing sentence | 2 |
| the boundary-pinning mechanism (M.8 above) | §4.1 | 1 |

Every check passed on every one of them. The claim audit compares numbers to raw records
and a retracted *sentence* has no number in it. `countcheck` resolves cardinals. `xref`
resolves references. **The class of thing being wrong was outside the class of thing being
checked**, which is M.1 with a specific mechanism attached.

**The practice, in two halves.**

*The procedure.* Before editing a claim that is being retracted or revised, grep its
wording across the paper, the appendices, the figure scripts, the tool's own output
strings and this document, and enumerate every site that asserts it. Fix all of them in
one commit or none. A grep for "roughly 16", "0.13" and "every site" would have caught
three of the four above, and each takes seconds.

*The gate.* `analysis/retracted.py` holds every retracted wording with what replaced it
and where the retraction is recorded, and fails the build if any of them is asserted
anywhere in the perimeter again. The convention that makes it work is that **a retraction
quotes the retired wording and an assertion does not**, which is how every correction in
this project is already written, so a match inside a quoted span is sanctioned and a bare
one is not. The Python half of the perimeter is parsed rather than scanned, because a
figure's in-panel header is a string literal and on raw bytes every character of it looks
like a quotation — which would have made the gate structurally unable to see the one
defect on the list that a reader would have taken away backwards.

Each entry carries the wording it retired, and a test asserts that every pattern flags its
own exemplar and no other entry's. A pattern with a typo in it otherwise sits in the table
matching nothing while the gate reports clean, which is M.5's family: a check that
cannot fail.

### The same failure has a second mode, and building the gate for the first did not close it

**Claim.** Propagation can fail on **retraction** — a claim is withdrawn where it was
measured and still asserted where it is summarised — and it can fail on **addition** — a
claim reaches every summary and the section that should source it is never touched. The
gate above closes the first. It is *structurally incapable* of closing the second, and the
difference is worth stating precisely: a retraction leaves a wording behind to search for,
and an addition leaves nothing at all. There is no string whose presence is wrong. The
defect is an absence, in a file nobody edited.

**Evidence, from the round that named partial propagation.** In the same commits that
built `retracted.py` and wrote the entry above, `[+4.2, +12.5]` — this paper's constraint
result, and half of its headline — went into the abstract, the introduction, Figure 1's
caption and the Conclusion. **§5.1, the section whose entire subject is that contrast,
never stated it.** Every gate passed. The claim audit recomputed the number from raw
records and found it correct. `countcheck` had no cardinal to resolve. `xref` had a
reference, and it resolved. `retracted.py` had nothing retracted. The number was right,
and it was asserted only in the four places whose job is to restate something said
elsewhere.

It was found by reading the built PDF, which is the fourth consecutive round in which the
read-through found what no gate could — and the first in which what it found was the
round's own work, committed alongside the entry explaining the class it belongs to.

**The practice.** *A summary may assert a number only if a section that measures it also
states it.* `analysis/forward.py` extracts every claim from the abstract, the
introduction, every figure and table caption, and the conclusion, and fails the build if
one is stated nowhere else. Two rules, because scalars and intervals fail differently:

- A **scalar** resolves if some measured value rounds to it at the precision the summary
  chose. Writing 0.363 for a measured 0.3634 is ordinary prose, and a gate demanding
  digit-identical restatement would force every summary to quote at appendix precision.
- An **interval** resolves only if both ends appear within one sentence of the body.
  Otherwise `[+4.2, +12.5]` is satisfied by a 4.2 in one appendix and a 12.5 in another —
  which is how a gate on bare numbers would have passed the exact defect it was written
  for. That case is a test.

Fed the tree as it stood at the offending commit — from git, not a reconstruction — it
names the interval at all four sites and nowhere else.

**What is deliberately not checked, and what is deliberately not allowed.** It does not
check that the summary *cites* the source: a missing pointer is a readability complaint
and a missing source is an unsupported claim, and only the second should fail a build.
And it has **no exemption mechanism**. One case wanted one — Figure 1's caption named a
superseded value, 97.9%, in the course of saying it was superseded — and the sentence was
rewritten to drop the number instead, because it was making the point about the header
rather than about the value. An exemption list is a gate that can be widened quietly, and
this project has enough evidence about what happens to checks nobody can fail (M.5).

**Both entries share a diagnosis and neither is the general fix.** The general fix is that
nothing here reads meaning (M.1). What these two do is convert two specific, recurring,
mechanically-detectable shapes of that failure into build errors, and the honest reading
of the second is that it exists because the round which understood the problem best still
committed a version of it.

---

## M.10 An instrument that sizes a decision is a gate, and needs the same known-bad-input check

**Claim.** M.3 says a validation gate must be tested against something already known to be
broken. That rule was applied to gates — things that pass or fail a build — and not to
*measurement instruments*, which merely produce a number that a human then acts on. The
distinction is not real. **An instrument whose output sizes a decision is a gate whose
verdict happens to be delivered in prose**, and it needs the same treatment, for the same
reason: otherwise it is a filter of unknown selectivity at the most consequential point.

**Evidence.** The instrument that measured this paper's per-section page cost computed a
section's extent as the distance from its own heading to *the next heading*. Nothing
labelled follows the last one. The bibliography — **1.83 pages** — was therefore charged
to the Conclusion, which read 2.35 pages of what is half a page of prose.

The consequences are the whole of the last three rounds of page arithmetic:

| what was sized against it | what it was told |
|---|---|
| round 8's seven cuts, and their targets | body 12 pp when it was 10 |
| round 9's re-derivation of those cuts | body 11.6 pp when it was 9.7 |
| an external reviewer's cut plan, twice | the same, at one remove |
| round 10's three costed scenarios | body 12.56 pp when it was 10.73 |

Every one of those arguments was made about a body **1.8 pages larger than it is**, and in
the direction that makes cutting look more necessary than it was.

**Why nothing caught it.** Not one of the reasons is exotic. The number *looked plausible*
— a long conclusion is unusual, not absurd, and the row sat in a table of forty others.
The instrument lived in a scratch directory rather than the repository, so it was outside
`pytest`, outside the claim audit, and outside every perimeter this project has spent five
rounds extending. And **no check compared the sum of the parts against the document**: a
span that overruns inflates exactly one row, and no other row disagrees with it. Row by
row the table is unfalsifiable; only the total can see the error.

That last point is the general one. A per-item measurement with no conservation check is
the same shape as a validation gate with no known-bad input: internally consistent, and
consistent with being wrong.

**The practice.**

- *An instrument that sizes a decision goes in the repository, under test, like any gate.*
  `analysis/pagecost.py`, with `tests/test_pagecost.py`.
- *It is fed a layout whose right answer is known.* `span_of` is separated out as a pure
  function of (label positions, terminator positions, document end) precisely so that it
  can be. The old behaviour is pinned as a test in the form that quantifies it: with the
  bibliography invisible, the last section reads exactly 3× its true extent.
- *It checks that its parts add up.* The measurement now reports the unattributed
  remainder — the title block, 0.45 pages — and fails if the parts exceed the document.

**What this does not fix.** Nothing here would have caught an instrument that is wrong by
a *constant factor*, since the parts would still sum correctly. Conservation catches
misattribution, not miscalibration, and the distinction is worth keeping in view: this
entry buys one specific class of error, not confidence in the number.

---

## M.11 A universal quantifier is an obligation across the whole document, and fixing the site that motivated it does not discharge it

**Claim.** The three propagation variants already recorded (M.9) are about a *particular*
claim reaching, or failing to reach, a *particular* site. This is the fourth and it is
different in kind. When a sentence says **every**, **all**, **never** or **in this paper**,
it creates an obligation over every population the document contains — including the ones
the author was not looking at when they wrote it. **A single-site fix does not discharge a
universally quantified claim**, and the site that motivated the sentence is exactly the one
the author *did* check.

**Evidence, both from rounds that were trying to be more precise.**

*Layer indices.* Round 9 added a disclosure of layer sampling because a reviewer asked what
the four-layer figure cost: "Every per-adapter number in this paper is measured on **layers
0, 12, 24 and 35**." True of Qwen3-8B, which has 36 layers. **Layer 35 does not exist on
the 32-layer Llama-3.1-8B-Instruct**, which carries the safety adapter — and §4.1 named
that adapter's real layers, 0/10/21/31, four paragraphs later, in the same document, added
in the same round.

*Adapter count.* "The correlation between delta magnitude and bin position is below 0.0011
across **all nine adapters**." The control covers **six** (EXP-009). The released tool's
own banner has said *six trained adapters* correctly the whole time; the paper drifted away
from its own tool.

**Two properties make this variant its own entry.**

First, **the sentence advertising a standard is what invites the check that finds the
violation.** Neither of these would have been caught if the paper had been vaguer. Round 9
wrote the layer sentence to be more specific than "we sample four layers", and specificity
is what made it falsifiable and false. That is not an argument for vagueness — it is a
reason to expect the *newest and most careful* sentences to be where the next defect is.

Second, **it is not gateable**. Whether "every per-adapter number is measured on layers 0,
12, 24 and 35" is true depends on two base models' layer counts, and no regex reaches that.
Where M.9's two gates fail a build, this one cannot.

**The practice.** `analysis/quantifiers.py` lists every universally quantified sentence in
the methods text that also names a value, and `--since <rev>` lists only those a round
added. It is **a worklist, not a verdict**: it is deliberately not wired into the build,
prints no pass or fail, and is marked as such in its own output, because a check that
cannot decide should not be allowed to look like one that can (M.3, M.5). What it buys is
that the list is *stable and diffable*, so the obligation a round creates is visible in the
round that creates it.

Run against this round's own edits, it returned seven new sentences and **three were
over-broad**, all written in the same session as this entry:

| sentence | scope it claimed | scope it has |
|---|---|---|
| "No paired contrast in this paper can clear a three-way Holm correction" | every paired contrast | the six-adapter ones; B.3 is paired on 252 cells |
| "Where this paper tests a paired contrast it uses the exact randomization test…" | every paired contrast | the same; B.3 carries no test |
| "Every per-adapter number in this paper is measured on four layers" | every per-adapter number | the main grid; the 36-layer run is the exception the same paragraph names |

All three are now scoped. **The entry describing this failure mode contained three
instances of it**, which is the most direct evidence available that the worklist is worth
having and that reading it is not optional.
