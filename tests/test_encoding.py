"""Portable checks of the equations in the frozen representation adapters.

Reference equations are written directly here; the tests do not import the
third-party deblurring backbone or depend on private datasets/checkpoints.
"""

import hashlib
import struct

import pytest
import torch
import torch.nn.functional as F

from structured_spike.encoding import (
    StructuredSpikeEncoder,
    TemporalValueLayer,
    fixed_projection,
    make_legacy_carrier,
    orderless_permutation,
    permutation_set_sha256,
)
from structured_spike.initialization import (
    REGISTERED_FIXED_SHA256,
    apply_initialization,
    tensor_sha256,
)
from structured_spike.portable_initialization import build_published_projection


MODES = ("fixed", "shared", "full", "countgrid", "summary", "sum_dense", "shuffled")


def frozen_basis(dtype=torch.float32):
    """rgb_representation.py::_time_basis and TemporalValueLayer.trilinear."""
    timestamps = torch.linspace(0.0, 1.0, 56, dtype=dtype)
    centers = torch.linspace(0.0, 1.0, 9, dtype=dtype)
    delta = timestamps.unsqueeze(0) - centers.unsqueeze(1)
    kernel = F.relu(1.0 - 8 * delta.abs())
    return timestamps, delta, kernel


@pytest.fixture
def spikes():
    generator = torch.Generator().manual_seed(19)
    return torch.randint(0, 2, (2, 56, 3, 5), generator=generator).float()


def test_registered_fixed_projection_hash():
    assert tensor_sha256(fixed_projection()) == REGISTERED_FIXED_SHA256


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fixed_matches_direct_frozen_equation(spikes, dtype):
    timestamps, _, kernel = frozen_basis(dtype)
    expected_weights = timestamps.unsqueeze(0) * kernel
    assert torch.equal(fixed_projection(dtype=dtype), expected_weights)
    expected = torch.einsum("bthw,kt->bkhw", spikes.to(dtype), expected_weights)
    actual = StructuredSpikeEncoder("fixed").to(dtype)(spikes.to(dtype))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.count_nonzero(expected_weights[:, 0]) == 0


@pytest.mark.parametrize("mode,expected", [
    ("fixed", 0), ("shared", 1021), ("full", 504), ("countgrid", 0),
    ("summary", 0), ("sum_dense", 0), ("shuffled", 0),
])
def test_trainable_parameter_counts(mode, expected):
    encoder = StructuredSpikeEncoder(mode)
    assert sum(p.numel() for p in encoder.parameters() if p.requires_grad) == expected
    assert sum(p.numel() for p in encoder.value_layer.parameters()) == 1021


@pytest.mark.parametrize("mode", ["shared", "full"])
def test_learnable_modes_start_at_fixed(spikes, mode):
    fixed = StructuredSpikeEncoder("fixed")(spikes)
    actual = StructuredSpikeEncoder(mode)(spikes)
    torch.testing.assert_close(actual, fixed, rtol=0, atol=0)


def test_shared_matches_direct_frozen_mlp_and_timestamp(spikes):
    encoder = StructuredSpikeEncoder("shared")
    with torch.no_grad():
        encoder.value_layer.mlp[-1].weight.fill_(0.25)
        encoder.value_layer.mlp[-1].bias.fill_(-0.125)
    timestamps, delta, kernel = frozen_basis()
    mlp = encoder.value_layer.mlp
    hidden = F.leaky_relu(F.linear(delta.unsqueeze(-1), mlp[0].weight, mlp[0].bias), 0.1)
    hidden = F.leaky_relu(F.linear(hidden, mlp[2].weight, mlp[2].bias), 0.1)
    residual = F.linear(hidden, mlp[4].weight, mlp[4].bias).squeeze(-1)
    weights = timestamps.unsqueeze(0) * (kernel + residual)
    expected = torch.einsum("bthw,kt->bkhw", spikes, weights)
    torch.testing.assert_close(encoder(spikes), expected, rtol=0, atol=0)
    impulse = torch.zeros_like(spikes)
    impulse[:, 0] = 1
    assert torch.count_nonzero(encoder(impulse)) == 0
    assert not torch.equal(encoder(spikes), StructuredSpikeEncoder("fixed")(spikes))


def test_shared_gradients_flow_through_residual_mlp(spikes):
    encoder = StructuredSpikeEncoder("shared")
    # Zero final-layer initialization intentionally blocks earlier layers on
    # the first step. Nonzero final weights exercise the complete MLP path.
    with torch.no_grad():
        encoder.value_layer.mlp[-1].weight.fill_(0.2)
    encoder(spikes).square().mean().backward()
    assert all(p.grad is not None for p in encoder.value_layer.parameters())
    assert all(torch.isfinite(p.grad).all() for p in encoder.value_layer.parameters())
    assert encoder.value_layer.mlp[0].weight.grad.abs().sum() > 0


def test_full_matches_direct_frozen_residual_equation(spikes):
    encoder = StructuredSpikeEncoder("full")
    with torch.no_grad():
        encoder.full_projection_residual.copy_(torch.arange(504).reshape(9, 56) / 504)
    timestamps, _, kernel = frozen_basis()
    weights = timestamps.unsqueeze(0) * kernel + encoder.full_projection_residual
    expected = torch.einsum("bthw,kt->bkhw", spikes, weights)
    torch.testing.assert_close(encoder(spikes), expected, rtol=0, atol=0)


def test_all_504_full_parameters_receive_gradients_including_first_column():
    encoder = StructuredSpikeEncoder("full")
    spikes = torch.ones(2, 56, 3, 5)
    encoder(spikes).sum().backward()
    gradient = encoder.full_projection_residual.grad
    assert gradient.shape == (9, 56)
    assert torch.count_nonzero(gradient) == 504
    assert torch.equal(gradient, torch.full((9, 56), 30.0))
    impulse = torch.zeros_like(spikes)
    impulse[:, 0] = 1
    with torch.no_grad():
        encoder.full_projection_residual[:, 0].fill_(2.0)
    assert torch.equal(encoder(impulse), torch.full((2, 9, 3, 5), 2.0))


@pytest.mark.parametrize("kind,replicate", [
    ("analytical", 0), ("registered_row", 0), ("independent_row", 1),
    ("independent_row", 2), ("common_column", 1), ("common_column", 2),
])
def test_full_initialization_uses_audited_projection(kind, replicate, spikes):
    encoder = StructuredSpikeEncoder("full", initialization=kind, replicate=replicate)
    target, metadata = build_published_projection(fixed_projection(), kind, replicate)
    residual = target - fixed_projection()
    assert torch.equal(encoder.full_projection_residual, residual)
    effective = fixed_projection() + residual
    expected = torch.einsum("bthw,kt->bkhw", spikes, effective)
    torch.testing.assert_close(encoder(spikes), expected, rtol=0, atol=0)
    assert encoder.initialization_metadata["target_projection_sha256"] == metadata["target_projection_sha256"]
    assert encoder.full_projection_residual.requires_grad


def test_full_remains_compatible_with_frozen_initialization_helper():
    encoder = StructuredSpikeEncoder("full")
    parameter_ids = [id(p) for p in encoder.parameters()]
    metadata = apply_initialization(encoder, "analytical", 0)
    assert metadata["target_projection_sha256"] == REGISTERED_FIXED_SHA256
    assert parameter_ids == [id(p) for p in encoder.parameters()]


def test_countgrid_matches_frozen_calibration_and_dc(spikes):
    timestamps, _, kernel = frozen_basis()
    active = (timestamps != 0).to(kernel.dtype).unsqueeze(0)
    gains = (timestamps.unsqueeze(0) * kernel).sum(1) / (kernel * active).sum(1)
    weights = gains.unsqueeze(1) * kernel * active
    expected = torch.einsum("bthw,kt->bkhw", spikes, weights)
    encoder = StructuredSpikeEncoder("countgrid")
    torch.testing.assert_close(encoder(spikes), expected, rtol=0, atol=0)
    assert torch.equal(weights != 0, fixed_projection() != 0)
    constant = torch.ones(1, 56, 2, 3)
    torch.testing.assert_close(encoder(constant), StructuredSpikeEncoder("fixed")(constant))
    impulse = torch.zeros_like(constant)
    impulse[:, 0] = 1
    assert torch.count_nonzero(encoder(impulse)) == 0


@pytest.mark.parametrize("mode", ["summary", "sum_dense"])
def test_summary_modes_match_frozen_formulas_without_computing_tfi_tfp(spikes, mode):
    tfi = torch.full((2, 1, 3, 5), 0.25)
    tfp = torch.full((2, 1, 3, 5), 0.75)
    if mode == "summary":
        expected = torch.cat((tfi, tfp, torch.zeros(2, 7, 3, 5)), dim=1)
    else:
        position = torch.arange(1, 10, dtype=torch.float32)
        matrix = torch.stack((position, 10.0 - position), dim=1) / 45.0
        expected = torch.einsum("bchw,kc->bkhw", torch.cat((tfi, tfp), dim=1), matrix)
    encoder = StructuredSpikeEncoder(mode)
    actual = encoder(spikes, tfi=tfi, tfp=tfp)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual.sum(1, keepdim=True), tfi + tfp)
    assert torch.equal(actual, encoder(torch.zeros_like(spikes), tfi=tfi, tfp=tfp))


def frozen_permutation(sample_id):
    salt = "iclr2027-orderless9-fixed-v1"
    keyed = [(hashlib.sha256(f"{salt}\0{sample_id}\0{frame}".encode()).digest(), frame)
             for frame in range(56)]
    permutation = tuple(frame for _, frame in sorted(keyed))
    return permutation[1:] + permutation[:1] if permutation == tuple(range(56)) else permutation


def test_shuffled_matches_frozen_permutation_without_touching_rng(spikes):
    sample_ids = ["sample-A", "synthetic/0002"]
    encoder = StructuredSpikeEncoder("shuffled")
    expected_spikes = torch.stack([spikes[i, list(frozen_permutation(sid))]
                                   for i, sid in enumerate(sample_ids)])
    expected = StructuredSpikeEncoder("fixed")(expected_spikes)
    # Constructing a module initializes its frozen MLP; only permutation and
    # encoding are RNG-free. Capture state after the reference constructor.
    before = torch.random.get_rng_state().clone()
    for sid in sample_ids:
        permutation = orderless_permutation(sid)
        assert permutation == frozen_permutation(sid)
        assert sorted(permutation) == list(range(56))
        assert permutation != tuple(range(56))
    actual = encoder(spikes, sample_ids=sample_ids)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(actual, encoder(spikes, sample_ids=sample_ids))


def test_permutation_set_hash_matches_frozen_serialization():
    ids = ["sample-A", "synthetic/0002"]
    digest = hashlib.sha256()
    for sid in ids:
        encoded = sid.encode("utf-8")
        digest.update(struct.pack(">I", len(encoded)))
        digest.update(encoded)
        digest.update(bytes(frozen_permutation(sid)))
    assert permutation_set_sha256(ids) == digest.hexdigest()
    # Sealed by executing the frozen 2026-10-26 function separately, not by
    # the public implementation used by this test.
    assert digest.hexdigest() == "5a07bdc55c8060facc3b793a97c408d31cdabc369a845a928f24fda4fd71f6b4"
    assert permutation_set_sha256(ids) != permutation_set_sha256(ids[::-1])


@pytest.mark.parametrize("mode", MODES)
def test_legacy_carrier_order_and_background_resize(spikes, mode):
    encoder = StructuredSpikeEncoder(mode)
    summaries = dict(tfi=torch.zeros(2, 1, 3, 5), tfp=torch.ones(2, 1, 3, 5))
    kwargs = {**summaries, "sample_ids": ["a", "b"]}
    background = torch.arange(16, dtype=torch.float32).view(2, 1, 2, 4)
    temporal = encoder(spikes, **kwargs)
    carrier = encoder.legacy_carrier(spikes, background, **kwargs)
    assert carrier.shape == (2, 26, 3, 5)
    assert torch.count_nonzero(carrier[:, :16]) == 0
    assert torch.equal(carrier[:, 16:25], temporal)
    expected_bg = F.interpolate(background, size=(3, 5), mode="bilinear", align_corners=False)
    assert torch.equal(carrier[:, 25:26], expected_bg)
    assert torch.equal(carrier, make_legacy_carrier(temporal, background))


@pytest.mark.parametrize("shape", [(56, 2, 3), (1, 55, 2, 3), (1, 57, 2, 3), (1, 56, 0, 3)])
def test_invalid_spike_shapes_raise(shape):
    with pytest.raises(ValueError, match="spike"):
        StructuredSpikeEncoder()(torch.zeros(shape))


@pytest.mark.parametrize("dtype", [torch.int32, torch.bool, torch.complex64])
def test_non_real_floating_spikes_raise(dtype):
    with pytest.raises(TypeError, match="floating"):
        StructuredSpikeEncoder()(torch.zeros(1, 56, 2, 3, dtype=dtype))


def test_non_tensor_spikes_raise():
    with pytest.raises(TypeError, match="Tensor"):
        StructuredSpikeEncoder()([1, 2, 3])


@pytest.mark.parametrize("mode", ["summary", "sum_dense"])
@pytest.mark.parametrize("bad_summary", [None, torch.zeros(2, 2, 3, 5),
    torch.zeros(1, 1, 3, 5), torch.zeros(2, 1, 4, 5), torch.zeros(2, 1, 3, 5).double()])
def test_invalid_summaries_raise(spikes, mode, bad_summary):
    with pytest.raises((TypeError, ValueError), match="TFI|TFP"):
        StructuredSpikeEncoder(mode)(spikes, tfi=bad_summary, tfp=torch.zeros(2, 1, 3, 5))


@pytest.mark.parametrize("ids", [None, [], ["a"], ["a", ""], ["a", 3], "ab"])
def test_invalid_sample_ids_raise(spikes, ids):
    with pytest.raises(ValueError, match="sample ID"):
        StructuredSpikeEncoder("shuffled")(spikes, sample_ids=ids)


@pytest.mark.parametrize("background", [torch.zeros(2, 3, 3, 5),
    torch.zeros(1, 1, 3, 5), torch.zeros(2, 1, 3, 5).double(), torch.zeros(2, 1, 0, 5)])
def test_invalid_legacy_background_raises(spikes, background):
    with pytest.raises((TypeError, ValueError), match="blur_gray"):
        StructuredSpikeEncoder().legacy_carrier(spikes, background)


def test_invalid_configuration_raises():
    with pytest.raises(ValueError, match="mode"):
        StructuredSpikeEncoder("unknown")
    with pytest.raises(ValueError, match="only.*full"):
        StructuredSpikeEncoder("fixed", initialization="registered_row")
    with pytest.raises(ValueError, match="replicate"):
        StructuredSpikeEncoder("full", replicate=1)
    with pytest.raises(ValueError, match="num_bins"):
        TemporalValueLayer(num_bins=1)


@pytest.mark.parametrize("frames", [0, 1, 257, True, 3.5])
def test_invalid_permutation_frame_counts_raise(frames):
    with pytest.raises(ValueError, match="frames"):
        orderless_permutation("a", frames)
    with pytest.raises(ValueError, match="frames"):
        permutation_set_sha256([], frames)


@pytest.mark.parametrize("ids", ["a", None, [3], [""]])
def test_invalid_permutation_sets_raise(ids):
    with pytest.raises(ValueError, match="sample ID"):
        permutation_set_sha256(ids)


def test_invalid_projection_dtype_and_legacy_temporal_shape_raise():
    with pytest.raises(TypeError, match="floating"):
        fixed_projection(dtype=torch.int32)
    with pytest.raises(ValueError, match="temporal"):
        make_legacy_carrier(torch.zeros(1, 8, 2, 3), torch.zeros(1, 1, 2, 3))


def test_shared_dtype_mismatch_gives_actionable_error(spikes):
    with pytest.raises(ValueError, match="move the encoder"):
        StructuredSpikeEncoder("shared")(spikes.double())


@pytest.mark.parametrize("mode", MODES)
def test_float64_and_noncontiguous_inputs_are_supported(mode):
    encoder = StructuredSpikeEncoder(mode).double()
    spike = torch.ones(1, 56, 2, 3, dtype=torch.float64).transpose(-1, -2)
    kwargs = {"tfi": torch.ones(1, 1, 3, 2, dtype=torch.float64),
              "tfp": torch.ones(1, 1, 3, 2, dtype=torch.float64), "sample_ids": ["synthetic"]}
    actual = encoder(spike, **kwargs)
    assert actual.dtype == torch.float64
    assert actual.shape == (1, 9, 3, 2)
    assert torch.equal(actual, encoder.encode(spike, **kwargs))
    assert mode in repr(encoder)


def test_state_dict_roundtrip(spikes):
    source = StructuredSpikeEncoder("full", initialization="registered_row")
    destination = StructuredSpikeEncoder("full")
    destination.load_state_dict(source.state_dict())
    assert torch.equal(source(spikes), destination(spikes))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is optional")
@pytest.mark.parametrize("mode", MODES)
def test_cuda_device_placement(mode):
    encoder = StructuredSpikeEncoder(mode).cuda()
    spike = torch.ones(1, 56, 2, 3, device="cuda")
    kwargs = {"tfi": torch.ones(1, 1, 2, 3, device="cuda"),
              "tfp": torch.ones(1, 1, 2, 3, device="cuda"), "sample_ids": ["synthetic"]}
    actual = encoder(spike, **kwargs)
    assert actual.device.type == "cuda"
    assert actual.shape == (1, 9, 2, 3)
