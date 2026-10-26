"""CPU contract tests for the frozen adapter, without checkpoints or training."""

import pytest
import torch

from structured_spike.restormer_adapter import (
    RestormerSpikeAdapter,
    _reflect_pad_to_multiple,
)


@pytest.fixture(scope="module")
def adapter():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = RestormerSpikeAdapter().cpu().eval()
    yield model
    torch.set_num_threads(previous_threads)


def make_inputs(height=4, width=4, batch=1):
    generator = torch.Generator().manual_seed(19)
    return (
        torch.randn(batch, 3, 2 * height, 2 * width, generator=generator),
        torch.randn(batch, 26, height, width, generator=generator),
        torch.randn(batch, 1, height, width, generator=generator),
    )


@pytest.mark.parametrize("height,width,batch", [(2, 2, 1), (4, 4, 1), (3, 5, 2)])
def test_cpu_forward_preserves_twice_native_resolution(adapter, height, width, batch):
    with torch.no_grad():
        outputs = adapter(*make_inputs(height, width, batch))
    assert len(outputs) == 3
    for output, channels in zip(outputs, (3, 1, 3)):
        assert output.shape == (batch, channels, 2 * height, 2 * width)
        assert output.dtype == torch.float32
        assert output.device.type == "cpu"
        assert torch.isfinite(output).all()


def test_three_output_heads_and_fusion_are_trainable(adapter):
    # One backward pass checks connectivity; there is no optimizer or update.
    adapter.zero_grad(set_to_none=True)
    outputs = adapter(*make_inputs(2, 2))
    sum(output.square().mean() for output in outputs).backward()
    for layer in (
        adapter.core.output,
        adapter.quick_head,
        adapter.fusion,
        adapter.recon_head[0],
    ):
        assert layer.weight.requires_grad
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()
        assert torch.count_nonzero(layer.weight.grad) > 0
        assert layer.bias is None
    added_parameters = sum(
        parameter.numel()
        for name, parameter in adapter.named_parameters()
        if not name.startswith("core.")
    )
    assert added_parameters == 16560
    adapter.zero_grad(set_to_none=True)


def test_quick_head_is_blur_only_but_other_heads_use_native_inputs(adapter):
    blur, spikes, guidance = make_inputs(2, 2)
    with torch.no_grad():
        original = adapter(blur, spikes, guidance)
        changed = adapter(blur, spikes + 0.5, guidance - 0.25)
    torch.testing.assert_close(original[2], changed[2], rtol=0, atol=0)
    assert not torch.equal(original[0], changed[0])
    assert not torch.equal(original[1], changed[1])


@pytest.mark.parametrize(
    "index,replacement,error,match",
    [
        (0, None, TypeError, "blur must be a torch.Tensor"),
        (0, torch.zeros(1, 1, 8, 8), ValueError, "blur must have shape"),
        (1, torch.zeros(1, 25, 4, 4), ValueError, "spike_input must have shape"),
        (2, torch.zeros(1, 2, 4, 4), ValueError, "common_guidance must have shape"),
        (1, torch.zeros(1, 26, 4, 4, dtype=torch.int64), TypeError, "floating-point"),
        (1, torch.zeros(1, 26, 4, 4, dtype=torch.float64), TypeError, "same dtype"),
        (1, torch.zeros(2, 26, 4, 4), ValueError, "same batch size"),
        (0, torch.zeros(0, 3, 8, 8), ValueError, "B >= 1"),
        (1, torch.zeros(1, 26, 1, 4), ValueError, "spatial dimensions >= 2"),
        (2, torch.zeros(1, 1, 4, 5), ValueError, "share native resolution"),
        (0, torch.zeros(1, 3, 7, 8), ValueError, "exactly 2x"),
        (1, torch.empty(1, 26, 4, 4, device="meta"), ValueError, "same device"),
        (1, torch.zeros(1, 26, 4, 4).to_sparse(), TypeError, "strided tensor layout"),
    ],
)
def test_invalid_input_contract(adapter, index, replacement, error, match):
    inputs = list(make_inputs())
    inputs[index] = replacement
    with pytest.raises(error, match=match):
        adapter(*inputs)


def test_inputs_must_match_model_device(adapter):
    inputs = tuple(value.to("meta") for value in make_inputs())
    with pytest.raises(ValueError, match="Inputs and model parameters"):
        adapter(*inputs)


def test_reflection_padding_matches_periodic_reflection():
    image = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]], requires_grad=True)
    padded = _reflect_pad_to_multiple(image, 8)
    expected = image.detach().repeat(1, 1, 4, 4)
    torch.testing.assert_close(padded, expected, rtol=0, atol=0)
    padded.sum().backward()
    torch.testing.assert_close(image.grad, torch.full_like(image, 16))


def test_aligned_padding_is_identity():
    image = torch.ones(1, 1, 8, 8)
    assert _reflect_pad_to_multiple(image, 8) is image


def test_singleton_axis_cannot_be_reflected():
    with pytest.raises(ValueError, match="spatial dimensions >= 2"):
        _reflect_pad_to_multiple(torch.ones(1, 1, 1, 2), 8)
