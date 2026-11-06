"""One bootstrap, shared by every artifact that reports an interval.

Eight copies of this function existed, and they did not agree. `phase1_pooled.py` --
the script Appendix D names as the source for section 5.1 -- defaulted to n=5000 while
the appendix generator used n=20000, same seed. Both were correct computations of the
same quantity; they printed different numbers, and the body table and Appendix B.6
disagreed in the last digit of three confidence intervals for a whole draft cycle. The
claim audit could not catch it: it checked each number against the raw records
independently and never checked two occurrences against each other.

Two changes follow from that.

**The interval over adapters is now exact, not sampled.** With six adapters the
bootstrap distribution has 6**6 = 46,656 distinct resamples, which enumerate in about
20 ms. Enumerating removes Monte Carlo error rather than shrinking it, so there is no
seed, no n, and no way for two callers to disagree. It also settles which of the two
published values was right: neither. At n=20000 the Monte Carlo standard deviation of
an endpoint is 0.04-0.17 percentage points, comparable to the last digit both artifacts
printed, so that digit was never supported by the estimator that produced it.

**Ratios still resample.** `ratio_ci` draws numerator and denominator independently,
so its space is 6**12 and cannot be enumerated. Its n and seed are pinned here, in one
place, which is the property that was actually missing.
"""
from __future__ import annotations

import random
from itertools import product

#: Above this sample size, enumeration costs more than the Monte Carlo error it removes:
#: 7 samples is 823,543 resamples, 8 is 16.7 million.
EXACT_MAX = 7
#: Used only when a sample is too large to enumerate. Pinned so callers cannot drift.
MC_DRAWS = 20000
MC_SEED = 0
ALPHA = 0.025


def _pct(sorted_vals: list[float]) -> tuple[float, float]:
    n = len(sorted_vals)
    return sorted_vals[int(ALPHA * n)], sorted_vals[int((1 - ALPHA) * n)]


def ci(xs: list[float]) -> tuple[float, float]:
    """Two-sided 95% bootstrap interval for the mean of `xs`.

    Exact by enumeration when `len(xs) <= EXACT_MAX`, which is every interval the paper
    reports over the six-adapter population. Deterministic: no seed argument exists,
    because a seed argument is what let two artifacts disagree.
    """
    k = len(xs)
    if k < 2:
        return (float("nan"), float("nan"))
    if k <= EXACT_MAX:
        return _pct(sorted(sum(c) / k for c in product(xs, repeat=k)))
    rng = random.Random(MC_SEED)
    return _pct(sorted(sum(xs[rng.randrange(k)] for _ in range(k)) / k
                       for _ in range(MC_DRAWS)))


def ratio_ci(num: list[float], den: list[float]) -> tuple[float, float]:
    """Interval for mean(num)/mean(den), resampling the two groups independently.

    Not enumerable at these sizes, so this stays Monte Carlo -- with n and seed fixed
    here rather than at each call site.
    """
    if len(num) < 2 or len(den) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(MC_SEED)
    out = []
    for _ in range(MC_DRAWS):
        a = sum(num[rng.randrange(len(num))] for _ in range(len(num))) / len(num)
        b = sum(den[rng.randrange(len(den))] for _ in range(len(den))) / len(den)
        out.append(a / b if b else float("nan"))
    return _pct(sorted(out))


def cluster_ratio_ci(
    clusters: list[tuple[str, list[float], list[float]]],
) -> tuple[float, float]:
    """Interval for mean(num)/mean(den) resampling CLUSTERS, stratified, paired.

    Each entry is (stratum, numerator values, denominator values) for one cluster.
    Clusters are drawn with replacement within each stratum, so the number drawn from a
    stratum always equals the number it has.

    Three differences from `ratio_ci`, each of which matters here:

    * **Clusters, not observations.** The hint battery is 8 intents x 3 paraphrases, so
      treating 32 prompts as 32 independent draws counts clustered draws as independent
      ones. Measured (B.12), the ICC is 0.175-0.303 and the battery carries 23-26
      effective units, so the prompt-level SE is too small by 11-18%. Not the sqrt(2)
      that an earlier version of this docstring assumed from "~16 independent units";
      that is the ICC = 1 case and it is not what the paraphrases do.
    * **Stratified.** Hint and adversarial prompts are different instruments; resampling
      them in one pool lets a draw contain almost none of either.
    * **Paired.** Numerator and denominator are the same prompts under two precisions, so
      one draw of clusters indexes both. `ratio_ci` resamples them independently, which
      discards the pairing.
    """
    if len(clusters) < 2:
        return (float("nan"), float("nan"))
    by_stratum: dict[str, list[int]] = {}
    for i, (s, _, _) in enumerate(clusters):
        by_stratum.setdefault(s, []).append(i)
    rng = random.Random(MC_SEED)
    out: list[float] = []
    for _ in range(MC_DRAWS):
        num_sum = den_sum = 0.0
        num_n = den_n = 0
        for idx in by_stratum.values():
            for _ in range(len(idx)):
                _, nv, dv = clusters[idx[rng.randrange(len(idx))]]
                num_sum += sum(nv)
                num_n += len(nv)
                den_sum += sum(dv)
                den_n += len(dv)
        if num_n and den_n and den_sum:
            out.append((num_sum / num_n) / (den_sum / den_n))
    if len(out) < 2:
        return (float("nan"), float("nan"))
    return _pct(sorted(out))


def is_exact(xs: list[float]) -> bool:
    """Whether `ci` enumerated rather than sampled. Reported next to intervals so a
    reader knows which kind of number they are looking at."""
    return 2 <= len(xs) <= EXACT_MAX


def endpoint_sd(xs: list[float], reps: int = 8) -> tuple[float, float]:
    """Monte Carlo standard deviation of each endpoint, over `reps` independent runs.

    Zero for an enumerated interval. For a sampled one this is the quantity that decides
    how many digits may be printed: publishing an endpoint to a finer resolution than
    its own resampling noise is what let two artifacts disagree in the last digit while
    both were correct.
    """
    if is_exact(xs):
        return (0.0, 0.0)
    k = len(xs)
    los, his = [], []
    for seed in range(reps):
        rng = random.Random(seed)
        lo, hi = _pct(sorted(sum(xs[rng.randrange(k)] for _ in range(k)) / k
                             for _ in range(MC_DRAWS)))
        los.append(lo)
        his.append(hi)
    return (_sd(los), _sd(his))


def _sd(v: list[float]) -> float:
    m = sum(v) / len(v)
    return (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0


def fmt_ci(xs: list[float], scale: float = 1.0, unit: str = "") -> str:
    """Format an interval at the resolution its estimator supports.

    Exact intervals print in full and are marked; sampled ones are rounded to the digit
    the Monte Carlo standard deviation supports and carry their draw count, so a reader
    can tell the two apart without reading the method section.
    """
    lo, hi = ci(xs)
    lo, hi = lo * scale, hi * scale
    if is_exact(xs):
        return f"[{lo:.4g}, {hi:.4g}]{unit}"
    sd_lo, sd_hi = endpoint_sd(xs)
    worst = max(sd_lo, sd_hi) * scale
    # One decimal beyond the noise, floored at nothing finer than the noise itself.
    dec = 0 if worst >= 1 else max(0, min(4, -int(f"{worst:e}".split("e")[1])))
    return (f"[{lo:.{dec}f}, {hi:.{dec}f}]{unit}"
            f" (MC n={MC_DRAWS}, endpoint SD {worst:.{dec + 1}f})")
