"""Hand-computed validation of the group-wise affine quantizer.

Every expected value here is derived by hand and the arithmetic is shown in a
comment. No expected value is copied from the implementation's own output, which
would make these tests a regression harness rather than a correctness proof.

Asymmetric convention under test (GPTQ):
    xmax <- max(group, 0), xmin <- min(group, 0)      zero always representable
    all-zero group      -> range forced to [-1, 1]
    scale = (xmax - xmin) / (2^b - 1)
    zero  = round(-xmin / scale)
    code  = clamp(round(w / scale) + zero, 0, 2^b - 1)
    deq   = (code - zero) * scale

symmetric_awq convention under test (AWQ/torch style, NOT gptqmodel):
    qmax = 2^(b-1) - 1, qmin = -2^(b-1)
    scale = absmax / qmax        (all-zero group -> absmax forced to 1)
    code  = clamp(round(w / scale), qmin, qmax)
    deq   = code * scale
"""

from __future__ import annotations

import pytest
import torch

from ar.quantsim import QuantConfig, quantize_dequantize

ASYM4_G4 = QuantConfig(bits=4, group_size=4, scheme="asymmetric")
SYM4_G4 = QuantConfig(bits=4, group_size=4, scheme="symmetric_awq")


def test_torch_round_is_half_to_even() -> None:
    # The whole test suite's expected codes depend on this rounding mode. If a
    # torch release ever changes it, fail here rather than in a retention number.
    got = torch.round(torch.tensor([0.5, 1.5, 2.5, 3.5, -0.5, -1.5]))
    assert got.tolist() == [0.0, 2.0, 2.0, 4.0, -0.0, -2.0]


def test_asym4_exact_grid_nonnegative() -> None:
    # group [0,1,2,3]: xmax=3, xmin=min(0,0)=0, scale=3/15=0.2, zero=round(0)=0
    # codes = round([0,1,2,3]/0.2) = [0,5,10,15]; deq = codes*0.2 = input exactly
    w = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    r = quantize_dequantize(w, ASYM4_G4)
    assert r.codes.tolist() == [[0.0, 5.0, 10.0, 15.0]]
    torch.testing.assert_close(r.dequant, w)
    torch.testing.assert_close(r.scale, torch.tensor([[0.2]]))
    assert r.zero is not None and r.zero.tolist() == [[0.0]]


def test_asym4_exact_grid_spanning_zero() -> None:
    # group [-1,0,1,2]: xmax=2, xmin=-1, scale=3/15=0.2, zero=round(1/0.2)=5
    # codes = round(w/0.2)+5 = [-5,0,5,10]+5 = [0,5,10,15]
    # deq = (codes-5)*0.2 = [-1,0,1,2] exactly
    w = torch.tensor([[-1.0, 0.0, 1.0, 2.0]])
    r = quantize_dequantize(w, ASYM4_G4)
    assert r.codes.tolist() == [[0.0, 5.0, 10.0, 15.0]]
    assert r.zero is not None and r.zero.tolist() == [[5.0]]
    torch.testing.assert_close(r.dequant, w)


def test_asym4_half_way_values_round_to_even() -> None:
    # Chosen so scale is exactly 1.0 and the halves are exact in binary:
    # group [0,0.5,1.5,2.5,15]: xmax=15, xmin=0, scale=15/15=1.0, zero=0
    # round-half-to-even -> [0, 0, 2, 2, 15]
    cfg = QuantConfig(bits=4, group_size=5, scheme="asymmetric")
    w = torch.tensor([[0.0, 0.5, 1.5, 2.5, 15.0]])
    r = quantize_dequantize(w, cfg)
    assert r.codes.tolist() == [[0.0, 0.0, 2.0, 2.0, 15.0]]
    torch.testing.assert_close(r.dequant, torch.tensor([[0.0, 0.0, 2.0, 2.0, 15.0]]))


def test_asym4_constant_negative_group_is_exact() -> None:
    # group [-3,-3,-3,-3]: xmax=max(-3,0)=0, xmin=-3, scale=3/15=0.2
    # zero=round(3/0.2)=15; codes=round(-3/0.2)+15=-15+15=0
    # deq=(0-15)*0.2=-3.0 exactly. A constant group must survive intact.
    w = torch.full((1, 4), -3.0)
    r = quantize_dequantize(w, ASYM4_G4)
    assert r.codes.tolist() == [[0.0, 0.0, 0.0, 0.0]]
    torch.testing.assert_close(r.dequant, w)


def test_asym4_all_zero_group_is_exactly_zero_and_scale_is_finite() -> None:
    # Degenerate group: range forced to [-1,1], scale=2/15, zero=round(1/scale).
    # codes = round(0/scale)+zero = zero, so deq = (zero-zero)*scale = 0 exactly,
    # whatever zero rounds to. The point is no division by zero and no NaN.
    w = torch.zeros((1, 4))
    r = quantize_dequantize(w, ASYM4_G4)
    assert torch.all(r.dequant == 0.0)
    assert torch.isfinite(r.scale).all() and (r.scale > 0).all()


def test_sym4_all_zero_group_is_exactly_zero_and_scale_is_finite() -> None:
    # absmax forced to 1 -> scale = 1/7, codes = 0, deq = 0.
    w = torch.zeros((1, 4))
    r = quantize_dequantize(w, SYM4_G4)
    assert torch.all(r.dequant == 0.0)
    assert r.zero is None
    torch.testing.assert_close(r.scale, torch.tensor([[1.0 / 7.0]]))


def test_sym4_exact_grid() -> None:
    # group [-7,-1,0,7]: absmax=7, qmax=7, scale=1.0, codes=round(w)=[-7,-1,0,7]
    w = torch.tensor([[-7.0, -1.0, 0.0, 7.0]])
    r = quantize_dequantize(w, SYM4_G4)
    assert r.codes.tolist() == [[-7.0, -1.0, 0.0, 7.0]]
    torch.testing.assert_close(r.dequant, w)
    torch.testing.assert_close(r.scale, torch.tensor([[1.0]]))


def test_sym4_clamps_to_signed_range() -> None:
    # group [-8,7,0,0]: absmax=8, scale=8/7. round(-8/(8/7))=round(-7)=-7,
    # round(7/(8/7))=round(6.125)=6. deq = [-8, 6*8/7, 0, 0].
    w = torch.tensor([[-8.0, 7.0, 0.0, 0.0]])
    r = quantize_dequantize(w, SYM4_G4)
    assert r.codes.tolist() == [[-7.0, 6.0, 0.0, 0.0]]
    torch.testing.assert_close(
        r.dequant, torch.tensor([[-8.0, 6.0 * 8.0 / 7.0, 0.0, 0.0]])
    )


def test_symgptq4_exact_grid() -> None:
    # gptqmodel's sym convention. Values chosen so scale is exactly 1.0.
    # group [-7.5, 0.5, 1.5, 7.5]: xmin=-7.5, xmax=7.5 (already mirrored)
    # scale = (7.5 - -7.5)/15 = 1.0, zero = (15+1)//2 = 8
    # codes = clamp(round(w) + 8, 0, 15):
    #   round(-7.5) = -8 -> 0;  round(0.5) = 0 -> 8
    #   round(1.5)  =  2 -> 10; round(7.5) = 8 -> 16 -> clamped to 15
    # deq = (codes - 8) * 1.0 = [-8, 0, 2, 7]
    cfg = QuantConfig(bits=4, group_size=4, scheme="symmetric_gptq")
    w = torch.tensor([[-7.5, 0.5, 1.5, 7.5]])
    r = quantize_dequantize(w, cfg)
    assert r.codes.tolist() == [[0.0, 8.0, 10.0, 15.0]]
    torch.testing.assert_close(r.dequant, torch.tensor([[-8.0, 0.0, 2.0, 7.0]]))
    torch.testing.assert_close(r.scale, torch.tensor([[1.0]]))
    assert r.zero is not None and r.zero.tolist() == [[8.0]]


def test_symgptq_clips_all_nonnegative_groups_like_the_reference() -> None:
    # Faithful replication of a gptqmodel quirk, asserted so it cannot silently
    # change. For an all-non-negative group gptqmodel leaves xmin at 0 (it only
    # mirrors where xmin < 0) while still placing the zero point at 8, so the top
    # half of the group's range is unreachable.
    # group [0,5,10,15]: xmin=0, xmax=15, scale=1.0, zero=8
    # codes = clamp(round(w)+8, 0, 15) = [8, 13, 18->15, 23->15]
    # deq = [0, 5, 7, 7]   <- 10 and 15 both collapse to 7
    cfg = QuantConfig(bits=4, group_size=4, scheme="symmetric_gptq")
    w = torch.tensor([[0.0, 5.0, 10.0, 15.0]])
    r = quantize_dequantize(w, cfg)
    assert r.codes.tolist() == [[8.0, 13.0, 15.0, 15.0]]
    torch.testing.assert_close(r.dequant, torch.tensor([[0.0, 5.0, 7.0, 7.0]]))


def test_symgptq_all_zero_group_is_exactly_zero() -> None:
    cfg = QuantConfig(bits=4, group_size=4, scheme="symmetric_gptq")
    r = quantize_dequantize(torch.zeros((1, 4)), cfg)
    assert torch.all(r.dequant == 0.0)
    assert torch.isfinite(r.scale).all() and (r.scale > 0).all()


def test_asym8_exact_grid() -> None:
    # 8-bit: maxq=255. group [0,255]: scale=255/255=1.0, zero=0, codes=[0,255].
    cfg = QuantConfig(bits=8, group_size=2, scheme="asymmetric")
    w = torch.tensor([[0.0, 255.0]])
    r = quantize_dequantize(w, cfg)
    assert r.codes.tolist() == [[0.0, 255.0]]
    torch.testing.assert_close(r.dequant, w)


def test_sym8_exact_grid() -> None:
    # 8-bit symmetric: qmax=127. group [-127,127]: absmax=127, scale=1.0.
    cfg = QuantConfig(bits=8, group_size=2, scheme="symmetric_awq")
    w = torch.tensor([[-127.0, 127.0]])
    r = quantize_dequantize(w, cfg)
    assert r.codes.tolist() == [[-127.0, 127.0]]
    torch.testing.assert_close(r.dequant, w)


def test_ragged_final_group_gets_its_own_scale() -> None:
    # n_in=5, g=4 -> group0=[1,1,1,1], group1=[8] (ragged tail of length 1).
    # group0: xmax=1, xmin=0, scale=1/15, zero=0, code=round(15)=15, deq=1.0
    # group1: xmax=8, xmin=0, scale=8/15, zero=0, code=round(15)=15, deq=8.0
    # The tail's step size must come only from the tail. If padding leaked into
    # the range, or the tail were folded into group0, scale[0,1] would not be 8/15.
    cfg = QuantConfig(bits=4, group_size=4, scheme="asymmetric")
    w = torch.tensor([[1.0, 1.0, 1.0, 1.0, 8.0]])
    r = quantize_dequantize(w, cfg)
    assert r.scale.shape == (1, 2)
    torch.testing.assert_close(r.scale, torch.tensor([[1.0 / 15.0, 8.0 / 15.0]]))
    torch.testing.assert_close(r.dequant, w)
    assert r.codes.shape == w.shape


def test_ragged_tail_scale_is_independent_of_the_full_group() -> None:
    # Same tail value, wildly different leading group. The tail's scale must be
    # identical across both cases; a leak would change it.
    cfg = QuantConfig(bits=4, group_size=4, scheme="asymmetric")
    a = quantize_dequantize(torch.tensor([[0.0, 0.0, 0.0, 0.0, 8.0]]), cfg)
    b = quantize_dequantize(torch.tensor([[-50.0, 3.0, 99.0, 7.0, 8.0]]), cfg)
    torch.testing.assert_close(a.scale[:, 1], b.scale[:, 1])


def test_group_size_minus_one_is_one_group_per_row() -> None:
    cfg = QuantConfig(bits=4, group_size=-1, scheme="asymmetric")
    w = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]])
    r = quantize_dequantize(w, cfg)
    assert r.scale.shape == (2, 1)
    # Row 1 is 10x row 0, so its step size must be 10x as well.
    torch.testing.assert_close(r.scale[1, 0] / r.scale[0, 0], torch.tensor(10.0))


def test_rows_are_quantized_independently() -> None:
    # Row 0 alone and row 1 alone must give the same result as both together.
    cfg = QuantConfig(bits=4, group_size=4, scheme="asymmetric")
    r0 = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    r1 = torch.tensor([[-100.0, 5.0, 0.5, 62.0]])
    both = quantize_dequantize(torch.cat([r0, r1]), cfg)
    alone0 = quantize_dequantize(r0, cfg)
    alone1 = quantize_dequantize(r1, cfg)
    torch.testing.assert_close(both.dequant[0:1], alone0.dequant)
    torch.testing.assert_close(both.dequant[1:2], alone1.dequant)


@pytest.mark.parametrize("scheme", ["asymmetric", "symmetric_awq", "symmetric_gptq"])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("group_size", [4, 32, 128, -1])
def test_codes_stay_in_range_and_quantization_is_idempotent(
    scheme: str, bits: int, group_size: int
) -> None:
    torch.manual_seed(0)
    w = torch.randn(8, 260)  # 260 is not a multiple of 32 or 128: ragged on purpose
    cfg = QuantConfig(bits=bits, group_size=group_size, scheme=scheme)  # type: ignore[arg-type]
    r = quantize_dequantize(w, cfg)

    if scheme == "symmetric_awq":
        lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    else:
        lo, hi = 0, 2**bits - 1
    assert r.codes.min() >= lo and r.codes.max() <= hi

    # Re-quantizing an already-quantized tensor must be a no-op: the values are
    # already exactly on the grid. This catches off-by-one scale/zero errors that
    # exact-value tests on small tensors can miss.
    #
    # symmetric_gptq is excluded, and not because it is broken: gptqmodel's sym
    # scheme is genuinely non-idempotent. Code 0 reaches -8s while +8s clips to
    # +7s, so the mirrored range grows on the second pass and the scale shifts.
    # Measured on the reference itself at int4 g128: max|Q(Q(w)) - Q(w)| =
    # 4.456093e-01 for gptqmodel, and 4.456093e-01 for ours. See EXP-003.
    if scheme != "symmetric_gptq":
        again = quantize_dequantize(r.dequant, cfg)
        torch.testing.assert_close(again.dequant, r.dequant, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("group_size", [4, 7, 32, 128, -1])
def test_step_per_weight_matches_group_membership(group_size: int) -> None:
    torch.manual_seed(1)
    w = torch.randn(4, 130)
    cfg = QuantConfig(bits=4, group_size=group_size, scheme="asymmetric")
    r = quantize_dequantize(w, cfg)
    step = r.step_per_weight()
    assert step.shape == w.shape
    g = w.shape[1] if group_size == -1 else group_size
    # Every weight's step must equal its own group's scale.
    for j in (0, 1, g - 1, g, w.shape[1] - 1):
        if j < w.shape[1]:
            torch.testing.assert_close(step[:, j], r.scale[:, j // g])


def test_dequant_error_is_bounded_by_half_a_step() -> None:
    # A correct affine quantizer never errs by more than s/2 on any weight that is
    # inside the group range. This is the property the whole retention argument
    # rests on, so assert it directly.
    torch.manual_seed(2)
    w = torch.randn(16, 512)
    cfg = QuantConfig(bits=4, group_size=128, scheme="asymmetric")
    r = quantize_dequantize(w, cfg)
    step = r.step_per_weight()
    err = (r.dequant - w).abs()
    assert torch.all(err <= step / 2 + 1e-6)


def test_finer_group_size_never_has_a_larger_mean_step() -> None:
    # Smaller groups cannot have a wider dynamic range than the group containing
    # them, so the mean step size must be monotone non-increasing in group size.
    torch.manual_seed(3)
    w = torch.randn(8, 512)
    means = []
    for g in (512, 128, 64, 32):
        cfg = QuantConfig(bits=4, group_size=g, scheme="asymmetric")
        means.append(quantize_dequantize(w, cfg).scale.mean().item())
    assert means == sorted(means, reverse=True), means


def test_symgptq_is_not_idempotent_and_that_is_faithful() -> None:
    # Asserted so nobody later "fixes" symmetric_gptq into idempotence and thereby
    # silently stops matching gptqmodel. The reference value below was measured by
    # running gptqmodel's own Quantizer twice on this exact input (EXP-003).
    torch.manual_seed(0)
    w = torch.randn(8, 260)
    cfg = QuantConfig(bits=4, group_size=128, scheme="symmetric_gptq")
    once = quantize_dequantize(w, cfg).dequant
    twice = quantize_dequantize(once, cfg).dequant
    drift = (twice - once).abs().max().item()
    assert drift == pytest.approx(4.456093e-01, rel=1e-5), drift


def test_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="NaN"):
        quantize_dequantize(torch.tensor([[float("nan"), 1.0, 2.0, 3.0]]), ASYM4_G4)
    with pytest.raises(ValueError, match="inf"):
        quantize_dequantize(torch.tensor([[float("inf"), 1.0, 2.0, 3.0]]), ASYM4_G4)
    with pytest.raises(ValueError, match="2-D"):
        quantize_dequantize(torch.zeros(4), ASYM4_G4)
    with pytest.raises(ValueError, match="2-D"):
        quantize_dequantize(torch.zeros(2, 2, 2), ASYM4_G4)


def test_rejects_bad_config() -> None:
    # 3-bit became supported for the coarse Phase 1 conditions (EXP-013); 2-bit
    # has never been validated against gptqmodel and must stay refused.
    assert QuantConfig(bits=3, group_size=128, scheme="asymmetric").bits == 3
    with pytest.raises(ValueError, match="bits"):
        QuantConfig(bits=2, group_size=128, scheme="asymmetric")
    with pytest.raises(ValueError, match="bits"):
        QuantConfig(bits=16, group_size=128, scheme="asymmetric")
    with pytest.raises(ValueError, match="group_size"):
        QuantConfig(bits=4, group_size=0, scheme="asymmetric")
    with pytest.raises(ValueError, match="group_size"):
        QuantConfig(bits=4, group_size=-2, scheme="asymmetric")


def test_config_name_is_stable_for_result_paths() -> None:
    assert QuantConfig(bits=4, group_size=128, scheme="asymmetric").name == "int4_g128_asym"
    assert QuantConfig(bits=8, group_size=-1, scheme="symmetric_awq").name == "int8_per_channel_symawq"
    assert QuantConfig(bits=4, group_size=32, scheme="symmetric_gptq").name == "int4_g32_symgptq"


