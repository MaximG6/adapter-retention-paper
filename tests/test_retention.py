"""Tests for the retention metrics, including the fixed/adaptive scale distinction.

A note on the sub-threshold claim, because it is easy to state wrongly and the
whole interpretation of the step-ratio metric depends on it:

    |D_i| < s/2 does NOT imply the weight's code is unchanged, even on a fixed
    grid.

Whether a weight flips depends on where it sits inside its bin. With s = 1 and
z = 0, w = 0.40 has code 0; adding D = 0.40 gives 0.80, code 1 - a flip, despite
|D|/(s/2) = 0.8 < 1. What |D| < s/2 does guarantee is that the code moves by at
most one step.

For a weight uniformly positioned within its bin, P(code changes) = min(|D|/s, 1),
verified to four decimals in EXP-004. So at step_ratio == 1 exactly half of those
weights flip. The guaranteed-zero-flip condition is positional: bin-centred
weights with |D| < s/2 cannot flip, and that is what the test below constructs.
"""

from __future__ import annotations

import math

import pytest
import torch

from ar.quantsim import QuantConfig, compute_params
from ar.retention import compare_regimes, compute_retention, lora_delta

ASYM4_G128 = QuantConfig(bits=4, group_size=128, scheme="asymmetric")


def _bin_centred_base(n_out: int, n_in: int, config: QuantConfig) -> torch.Tensor:
    """A base weight whose every element sits exactly at the centre of its bin.

    Built by quantizing a random tensor and keeping the dequantized result: those
    values are on grid points by construction, and a grid point is the centre of
    its own rounding interval.
    """
    torch.manual_seed(0)
    raw = torch.randn(n_out, n_in)
    from ar.quantsim import quantize_dequantize

    return quantize_dequantize(raw, config).dequant


def test_fixed_scale_gives_zero_flips_for_bin_centred_subthreshold_delta() -> None:
    # The correctly-stated version of "sub-threshold deltas do not flip": weights
    # exactly on grid points, every |D| strictly below s/2, fixed grid from W.
    # Nothing can cross a boundary, so flips must be exactly zero.
    w = _bin_centred_base(8, 256, ASYM4_G128)
    step = compute_params(w, ASYM4_G128).step_per_weight()

    torch.manual_seed(1)
    frac = torch.empty(w.shape).uniform_(-0.49, 0.49)
    d = frac * step

    assert (d.abs() < step / 2).all()
    m = compute_retention(w, d, ASYM4_G128, "fixed_scale")
    assert m.code_flip_rate == 0.0
    assert m.value_change_rate == 0.0
    assert m.retention_ratio == 0.0
    assert m.subthreshold_fraction == 1.0


def test_subthreshold_delta_can_still_flip_when_weights_sit_near_bin_edges() -> None:
    # The counter-example that makes the test above need its careful construction.
    # Offsetting the base by 0.49 steps puts every weight just inside a boundary;
    # a delta of 0.4 steps then pushes many across it, though all are < s/2.
    w0 = _bin_centred_base(8, 256, ASYM4_G128)
    step = compute_params(w0, ASYM4_G128).step_per_weight()
    w = w0 + 0.49 * step

    d = 0.4 * step
    assert (d.abs() < step / 2).all()
    m = compute_retention(w, d, ASYM4_G128, "fixed_scale")
    assert m.code_flip_rate > 0.3, m.code_flip_rate
    assert m.subthreshold_fraction == 1.0


def test_predicted_flip_rate_matches_measured_under_uniform_bin_positions() -> None:
    # P(flip) = min(|D|/s, 1) for uniformly-positioned weights. This ties the
    # step-ratio distribution to the bit-flip rate analytically, so a divergence
    # here means either a bug or a real correlation between delta and bin position.
    torch.manual_seed(2)
    w = torch.randn(16, 512)
    step = compute_params(w, ASYM4_G128).step_per_weight()
    for frac in (0.05, 0.25, 0.5):
        d = frac * step * torch.where(torch.rand(w.shape) < 0.5, -1.0, 1.0)
        m = compute_retention(w, d, ASYM4_G128, "fixed_scale")
        assert m.predicted_flip_rate == pytest.approx(frac, abs=1e-6)
        assert m.code_flip_rate == pytest.approx(frac, abs=0.02), (
            frac, m.code_flip_rate
        )


def test_fixed_and_adaptive_diverge_so_the_distinction_cannot_collapse() -> None:
    # If these two ever agreed exactly on a realistic input, the grid-shift
    # diagnostic would be vacuous and someone could quietly drop one regime.
    torch.manual_seed(3)
    w = torch.randn(32, 512)
    d = 0.05 * torch.randn(32, 512)

    cmp = compare_regimes(w, d, ASYM4_G128)
    assert cmp.fixed.retention_ratio != cmp.adaptive.retention_ratio
    assert cmp.grid_shift_fraction > 0.0
    assert cmp.scale_shift_fraction > 0.0
    assert cmp.retention_gap != 0.0


def test_grid_shift_changes_weights_the_delta_did_not_touch() -> None:
    # The artifact in its purest form. One weight in a group carries a large delta
    # that moves the group's max; every other weight in the group has D_i exactly
    # zero. Under fixed scale those untouched weights cannot change. Under adaptive
    # scale the group's step size shifts and some of them do.
    cfg = QuantConfig(bits=4, group_size=128, scheme="asymmetric")
    torch.manual_seed(4)
    w = torch.randn(4, 128)
    d = torch.zeros(4, 128)
    d[:, 0] = 3.0 * w.abs().max()  # blow out the group maximum

    fixed = compute_retention(w, d, cfg, "fixed_scale")
    adaptive = compute_retention(w, d, cfg, "adaptive_scale")
    cmp = compare_regimes(w, d, cfg)

    # 127 of every 128 weights have exactly zero delta.
    assert cmp.n_zero_delta == 4 * 127
    # Under fixed scale, at most the one perturbed weight per row can change.
    assert fixed.value_change_rate <= 4 / d.numel() + 1e-12
    # Under adaptive scale, many more change, and they are the untouched ones.
    assert adaptive.value_change_rate > fixed.value_change_rate
    assert cmp.grid_shift_fraction_zero_delta > 0.0


def test_zero_delta_gives_zero_retention_under_fixed_scale() -> None:
    # A delta of exactly zero is rejected, since retention would be 0/0 and the
    # usual cause is an adapter that failed to load.
    w = torch.randn(4, 128)
    with pytest.raises(ValueError, match="exactly zero"):
        compute_retention(w, torch.zeros(4, 128), ASYM4_G128, "fixed_scale")


def test_huge_delta_is_fully_retained_under_adaptive_scale_only() -> None:
    # A delta far above the step size comes through intact only under adaptive
    # scale. Under fixed scale, W + D exceeds the range of W's own grid and
    # saturates against the clamp, so retention is understated. This bounds where
    # fixed_scale is a valid instrument: it isolates the step-size mechanism only
    # while D stays small relative to W's range, which is our regime of interest.
    torch.manual_seed(5)
    w = torch.randn(8, 256)
    d = 5.0 * torch.randn(8, 256)

    adaptive = compute_retention(w, d, ASYM4_G128, "adaptive_scale")
    assert adaptive.retention_ratio > 0.95, adaptive.retention_ratio
    assert adaptive.cosine > 0.99, adaptive.cosine

    fixed = compute_retention(w, d, ASYM4_G128, "fixed_scale")
    assert fixed.retention_ratio < 0.7, fixed.retention_ratio
    assert fixed.retention_ratio < adaptive.retention_ratio


def test_cosine_rises_monotonically_with_delta_magnitude() -> None:
    # The dose-response curve the project rests on. Cosine is the metric that
    # behaves: it is bounded, monotone, and zero when the delta is destroyed.
    torch.manual_seed(6)
    w = torch.randn(8, 512)
    base_d = torch.randn(8, 512)
    for regime in ("fixed_scale", "adaptive_scale"):
        cosines = [
            compute_retention(w, m * base_d, ASYM4_G128, regime).cosine  # type: ignore[arg-type]
            for m in (0.001, 0.01, 0.1, 1.0)
        ]
        assert cosines == sorted(cosines), (regime, cosines)


def test_retention_ratio_is_unbounded_and_nonmonotone_for_small_deltas() -> None:
    # Pins the trap in the plan's headline metric so nobody reports it bare.
    # When |D| << s, each flipped weight contributes a full step to D_eff, so
    # ||D_eff|| >> ||D|| while pointing almost nowhere near D. A naive reading of
    # retention_ratio = 20 as "excellent retention" would be exactly backwards.
    torch.manual_seed(6)
    w = torch.randn(8, 512)
    base_d = torch.randn(8, 512)

    tiny = compute_retention(w, 0.001 * base_d, ASYM4_G128, "fixed_scale")
    large = compute_retention(w, 1.0 * base_d, ASYM4_G128, "fixed_scale")

    # Ratio is far above 1 for the tiny delta and near 1 for the large one:
    # non-monotone, and decreasing over this range.
    assert tiny.retention_ratio > 10.0, tiny.retention_ratio
    assert tiny.retention_ratio > large.retention_ratio

    # The metrics that read correctly all agree the tiny delta was destroyed.
    assert tiny.cosine < 0.1, tiny.cosine
    assert tiny.relative_error > 1.0, tiny.relative_error
    assert large.cosine > 0.9, large.cosine
    assert large.relative_error < 1.0, large.relative_error


def test_relative_error_equals_one_when_delta_is_fully_erased() -> None:
    # The erasure baseline: D_eff = 0 gives ||0 - D||/||D|| = 1 exactly. This is
    # what makes relative_error readable, unlike retention_ratio.
    w = _bin_centred_base(8, 256, ASYM4_G128)
    step = compute_params(w, ASYM4_G128).step_per_weight()
    torch.manual_seed(9)
    d = torch.empty(w.shape).uniform_(-0.49, 0.49) * step

    m = compute_retention(w, d, ASYM4_G128, "fixed_scale")
    assert m.retention_ratio == 0.0
    assert m.relative_error == pytest.approx(1.0, abs=1e-6)
    assert m.projection_coefficient == pytest.approx(0.0, abs=1e-6)


def test_projection_stays_near_one_where_cosine_collapses() -> None:
    # Quantization acts as a noisy but roughly unbiased channel for small deltas:
    # the delta biases which way each weight rounds, so it survives in projection
    # even when per-weight fidelity is gone. Relevant to Phase 1, where aggregate
    # behaviour may persist despite low cosine.
    torch.manual_seed(6)
    w = torch.randn(8, 512)
    d = 0.01 * torch.randn(8, 512)
    m = compute_retention(w, d, ASYM4_G128, "adaptive_scale")
    assert m.cosine < 0.3, m.cosine
    assert 0.5 < m.projection_coefficient < 1.6, m.projection_coefficient


def test_regime_is_required_and_validated() -> None:
    w = torch.randn(4, 128)
    d = 0.1 * torch.randn(4, 128)
    with pytest.raises(TypeError):
        compute_retention(w, d, ASYM4_G128)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="Unknown regime"):
        compute_retention(w, d, ASYM4_G128, "whatever")  # type: ignore[arg-type]


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="!="):
        compute_retention(
            torch.randn(4, 128), torch.randn(4, 64), ASYM4_G128, "fixed_scale"
        )


def test_lora_delta_matches_manual_product() -> None:
    # (alpha/r) * B @ A with PEFT's storage layout: A is (r, in), B is (out, r).
    torch.manual_seed(7)
    r, alpha = 8, 16.0
    a = torch.randn(r, 64)
    b = torch.randn(32, r)
    got = lora_delta(a, b, alpha=alpha, rank=r)
    assert got.shape == (32, 64)
    torch.testing.assert_close(got, (alpha / r) * (b @ a))


def test_lora_delta_rslora_scaling_matches_peft() -> None:
    # peft's LoraLayer.update_layer: scaling = alpha/sqrt(r) if use_rslora else
    # alpha/r. Getting this wrong is silent and large -- at r=128 the two differ
    # by sqrt(128) = 11.3x, which is enough to invert an adapter's ranking.
    torch.manual_seed(14)
    r, alpha = 128, 16.0
    a = torch.randn(r, 256)
    b = torch.randn(64, r)
    plain = lora_delta(a, b, alpha=alpha, rank=r, use_rslora=False)
    rs = lora_delta(a, b, alpha=alpha, rank=r, use_rslora=True)
    torch.testing.assert_close(plain, (alpha / r) * (b @ a))
    torch.testing.assert_close(rs, (alpha / math.sqrt(r)) * (b @ a))
    ratio = (torch.linalg.norm(rs) / torch.linalg.norm(plain)).item()
    assert ratio == pytest.approx(math.sqrt(r), rel=1e-5), ratio
    # Default must stay non-rsLoRA, matching peft's own default.
    torch.testing.assert_close(lora_delta(a, b, alpha=alpha, rank=r), plain)


def test_lora_delta_rejects_inconsistent_rank() -> None:
    with pytest.raises(ValueError, match="rank"):
        lora_delta(torch.randn(8, 64), torch.randn(32, 4), alpha=16.0, rank=8)


def test_cosine_times_retention_is_exactly_projection() -> None:
    # This is an algebraic identity, not an approximation:
    #   cos * ret = [<De,D>/(||De|| ||D||)] * [||De||/||D||] = <De,D>/||D||^2
    # So "assert cos * ret ~= 1" is not a test of two quantities agreeing; it is
    # precisely a test of the channel being unbiased. Asserted exactly here, and
    # the unbiasedness claim is tested separately below.
    torch.manual_seed(10)
    w = torch.randn(64, 1024)
    for m in (0.001, 0.01, 0.1, 1.0):
        r = compute_retention(w, m * torch.randn(64, 1024), ASYM4_G128, "fixed_scale")
        assert r.cosine * r.retention_ratio == pytest.approx(
            r.projection_coefficient, rel=1e-5
        )


def test_channel_is_unbiased_given_enough_flipped_weights() -> None:
    # E[D_eff] = D, so projection -> 1. The estimate is noisy in the number of
    # FLIPPED weights, not the number of weights, so this needs a large tensor at
    # a small delta. With ~4,800 flips the estimate lands within a few percent.
    torch.manual_seed(11)
    w = torch.randn(2048, 1024)
    d = 0.001 * torch.randn(2048, 1024)
    r = compute_retention(w, d, ASYM4_G128, "fixed_scale")
    assert r.code_flip_rate * d.numel() > 1000
    assert r.projection_coefficient == pytest.approx(1.0, abs=0.1), (
        r.projection_coefficient
    )


def test_cosine_follows_the_closed_form_sqrt_law() -> None:
    # Distribution-free form: ||D_eff||^2 = N*s*mean|D| and ||D||^2 = N*mean(D^2),
    # so with projection ~ 1,  cosine = sqrt(mean(D^2) / (s * mean|D|)).
    #
    # The simpler sqrt(mean|D|/s) understates cosine by a constant factor of
    # sqrt(pi/2) ~ 1.2533 for Gaussian deltas, because it drops the shape term
    # mean(D^2)/mean|D|^2. Departures from the Gaussian constant measure the
    # delta's tail shape, so the distribution-free form is the one to test.
    torch.manual_seed(12)
    w = torch.randn(512, 1024)
    step = compute_params(w, ASYM4_G128).step_per_weight()
    s_mean = step.mean().item()
    for m in (0.001, 0.003, 0.01, 0.03):
        d = m * torch.randn(512, 1024)
        r = compute_retention(w, d, ASYM4_G128, "fixed_scale")
        predicted = (
            (d**2).mean().item() / (s_mean * d.abs().mean().item())
        ) ** 0.5
        assert r.cosine == pytest.approx(predicted, rel=0.10), (m, r.cosine, predicted)


@pytest.mark.parametrize(
    ("alpha_scales_with_rank", "exponent"), [(True, 0.25), (False, -0.25)]
)
def test_alpha_convention_sets_the_sign_of_the_rank_trend(
    alpha_scales_with_rank: bool, exponent: float
) -> None:
    # Delta = (alpha/r) B A with iid factors gives std((BA)_ij) ~ sqrt(r), so
    # alpha = 2r yields |D| ~ sqrt(r) and cosine ~ r^(+1/4), while fixed alpha
    # yields |D| ~ 1/sqrt(r) and cosine ~ r^(-1/4). The rank trend REVERSES on a
    # hyperparameter convention, which is why the sweep must cover both.
    torch.manual_seed(13)
    w = torch.randn(1024, 1024) * 0.02
    ranks = (4, 16, 64)
    cosines = []
    for rank in ranks:
        torch.manual_seed(100 + rank)
        a = torch.randn(rank, 1024) * 0.01
        b = torch.randn(1024, rank) * 0.01
        alpha = 2.0 * rank if alpha_scales_with_rank else 16.0
        d = lora_delta(a, b, alpha=alpha, rank=rank)
        cosines.append(compute_retention(w, d, ASYM4_G128, "fixed_scale").cosine)

    if alpha_scales_with_rank:
        assert cosines == sorted(cosines), cosines
    else:
        assert cosines == sorted(cosines, reverse=True), cosines

    observed = cosines[-1] / cosines[0]
    predicted = (ranks[-1] / ranks[0]) ** exponent
    assert observed == pytest.approx(predicted, rel=0.15), (observed, predicted)


def test_conventions_are_reported_and_can_differ() -> None:
    # Convention is a measured factor, not just a config flag: if the two
    # symmetric conventions give different retention on the same delta, that is a
    # result. This asserts the machinery keeps them distinguishable.
    torch.manual_seed(8)
    w = torch.randn(8, 256)
    d = 0.02 * torch.randn(8, 256)
    out = {}
    for scheme in ("asymmetric", "symmetric_awq", "symmetric_gptq"):
        cfg = QuantConfig(bits=4, group_size=128, scheme=scheme)  # type: ignore[arg-type]
        m = compute_retention(w, d, cfg, "fixed_scale")
        assert m.scheme == scheme
        out[scheme] = m.retention_ratio
    assert len(set(out.values())) > 1, out

