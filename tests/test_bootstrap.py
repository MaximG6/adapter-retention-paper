"""Tests for the shared bootstrap, including the cluster estimator PG-2 now uses.

The standing rule in this project is that a check is not trusted until it has failed on
an input whose answer is already known. `cluster_ratio_ci` exists to WIDEN intervals
relative to the observation-level bootstrap, so the tests that matter are the ones that
would catch it silently not doing that: a version that ignored the cluster structure
would pass a "returns a plausible interval" test and fail every test below.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import bootstrap  # noqa: E402


def _flat(vals: list[float], stratum: str = "a") -> list[tuple[str, list[float], list[float]]]:
    """One observation per cluster: the degenerate case with no clustering at all."""
    return [(stratum, [v], [1.0]) for v in vals]


def test_zero_width_when_every_cluster_identical() -> None:
    clusters = [("hint", [2.0, 2.0, 2.0], [1.0, 1.0, 1.0]) for _ in range(8)]
    lo, hi = bootstrap.cluster_ratio_ci(clusters)
    assert lo == pytest.approx(2.0)
    assert hi == pytest.approx(2.0)


def test_singleton_clusters_track_the_observation_bootstrap() -> None:
    """With one observation per cluster there IS no clustering, so the two estimators
    must agree. If they diverge here the cluster version has a resampling bug, not a
    clustering effect."""
    vals = [0.2, 0.5, 0.9, 1.4, 1.6, 2.1, 2.4, 3.0, 3.3, 3.9, 4.1, 4.8]
    c_lo, c_hi = bootstrap.cluster_ratio_ci(_flat(vals))
    r_lo, r_hi = bootstrap.ratio_ci(vals, [1.0] * len(vals))
    assert c_lo == pytest.approx(r_lo, abs=0.15)
    assert c_hi == pytest.approx(r_hi, abs=0.15)


def test_clustering_widens_relative_to_ignoring_it() -> None:
    """The defect this estimator exists to fix, stated as a test.

    Twelve observations in four clusters of three, with all the variation BETWEEN
    clusters and none within. Treating them as twelve independent draws must give a
    visibly narrower interval than resampling the four clusters.
    """
    groups = [[1.0] * 3, [2.0] * 3, [3.0] * 3, [10.0] * 3]
    clustered = [("g", g, [1.0] * 3) for g in groups]
    flat = [v for g in groups for v in g]
    c_lo, c_hi = bootstrap.cluster_ratio_ci(clustered)
    f_lo, f_hi = bootstrap.cluster_ratio_ci(_flat(flat))
    assert (c_hi - c_lo) > (f_hi - f_lo) * 1.2


def test_strata_are_resampled_separately() -> None:
    """A draw must always contain both strata in their original proportions.

    Pooling them would let a resample consist almost entirely of one stratum, which for
    an 8-hint/8-adversarial split shows up as a much wider interval.
    """
    clusters = ([("hint", [1.0], [1.0])] * 8) + ([("adv", [9.0], [1.0])] * 8)
    s_lo, s_hi = bootstrap.cluster_ratio_ci(clusters)
    pooled = [("all", n, d) for _, n, d in clusters]
    p_lo, p_hi = bootstrap.cluster_ratio_ci(pooled)
    assert s_hi - s_lo == pytest.approx(0.0, abs=1e-9)
    assert p_hi - p_lo > 1.0


def test_pairing_is_preserved_within_a_draw() -> None:
    """Numerator and denominator move together, so a cluster that is extreme in both
    cancels. An unpaired estimator cannot produce a zero-width interval here."""
    clusters = [("s", [k], [k]) for k in (1.0, 5.0, 20.0, 100.0)]
    lo, hi = bootstrap.cluster_ratio_ci(clusters)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)


def test_degenerate_inputs_do_not_silently_return_a_number() -> None:
    lo, hi = bootstrap.cluster_ratio_ci([("s", [1.0], [1.0])])
    assert lo != lo and hi != hi          # NaN, not a fabricated point interval


def test_ci_enumerates_at_six_and_is_seedless() -> None:
    xs = [0.287, 0.44, 0.61, 0.70, 0.81, 0.864]
    assert bootstrap.is_exact(xs)
    assert bootstrap.ci(xs) == bootstrap.ci(xs)
