"""Portable 56-frame to nine-channel spike representation equations.

This module extracts the representation formulas from the frozen study's
``rgb_representation.py``, ``iclr_control_representation.py``, and the
``TemporalValueLayer`` in ``fusion_unet.py``. It is not the complete trained
deblurring model, does not reproduce its constructor RNG stream, and does not
include the third-party backbone or TFI/TFP extraction utilities. Summary
images must be supplied by the caller.

The default output is B x 9 x H x W. The optional legacy carrier prepends 16
zero channels and appends resized grayscale blur, giving B x 26 x H x W.
No normalization, image resizing, or binarization is applied to native spikes.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .portable_initialization import build_published_projection


FRAMES = 56
BINS = 9
ORDERLESS_PROTOCOL = "sha256_sample_id_frame_sort_v1"
ORDERLESS_SALT = "iclr2027-orderless9-fixed-v1"
_REPRESENTATIONS = {
    "fixed": "fixed9_bg",
    "shared": "learned9_bg",
    "full": "learned_full9_bg",
    "countgrid": "countgrid9_bg",
    "summary": "c9_tfi_tfp_bg",
    "sum_dense": "sum_dense9_bg",
    "shuffled": "orderless9_bg",
}

__all__ = [
    "BINS", "FRAMES", "StructuredSpikeEncoder", "TemporalValueLayer",
    "fixed_projection", "make_legacy_carrier", "orderless_permutation",
    "permutation_set_sha256",
]


def _time_basis(*, device: torch.device | str | None, dtype: torch.dtype):
    # Keep linspace and operation order: arange/55 is mathematically equal but
    # does not necessarily have the registered float32 tensor bytes.
    timestamps = torch.linspace(0.0, 1.0, FRAMES, device=device, dtype=dtype)
    centers = torch.linspace(0.0, 1.0, BINS, device=device, dtype=dtype)
    return timestamps, timestamps.unsqueeze(0) - centers.unsqueeze(1)


def fixed_projection(
    *, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the analytical 9 x 56 matrix ``t * relu(1 - 8*abs(t-c))``.

    The canonical registered identity uses CPU float32. Other dtypes/devices
    evaluate the same formula in their own arithmetic; they need not have the
    canonical byte hash. The first column is exactly zero because t[0] = 0.
    """
    if not dtype.is_floating_point:
        raise TypeError("projection dtype must be real floating point")
    timestamps, delta = _time_basis(device=device, dtype=dtype)
    return timestamps.unsqueeze(0) * F.relu(1.0 - (BINS - 1) * delta.abs())


class TemporalValueLayer(nn.Module):
    """Analytical triangular kernel plus the study's 1-30-30-1 residual MLP.

    Both hidden activations are LeakyReLU(0.1). The last linear layer starts
    at zero, so the initial function is analytical although the earlier
    layers use PyTorch's standard random initialization. There are 1,021
    parameters. The external timestamp factor belongs to the encoder, not
    this relative-time kernel.
    """

    def __init__(self, num_bins: int = BINS):
        super().__init__()
        if type(num_bins) is not int or num_bins < 2:
            raise ValueError("num_bins must be an integer of at least 2")
        self.num_bins = num_bins
        self.mlp = nn.Sequential(
            nn.Linear(1, 30),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(30, 30),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(30, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def trilinear(self, delta_t: torch.Tensor) -> torch.Tensor:
        """Retain the historical method name for the triangular time kernel."""
        return F.relu(1.0 - (self.num_bins - 1) * delta_t.abs())

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        residual = self.mlp(delta_t.unsqueeze(-1)).squeeze(-1)
        return self.trilinear(delta_t) + residual


def orderless_permutation(sample_id: str, frames: int = FRAMES) -> tuple[int, ...]:
    """Return the frozen sample-keyed SHA-256 frame permutation without RNG.

    Despite its historical name, this is the *Shuffled* control: one fixed
    permutation per stable sample ID, not averaging over all permutations.
    ``frames`` is bounded to 2..256 for the byte serialization used by the
    frozen permutation-set hash; the study always uses 56.
    """
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"invalid sample ID for Shuffled: {sample_id!r}")
    if type(frames) is not int or not 2 <= frames <= 256:
        raise ValueError("frames must be an integer between 2 and 256")
    keyed = []
    for frame in range(frames):
        payload = f"{ORDERLESS_SALT}\0{sample_id}\0{frame}".encode("utf-8")
        keyed.append((hashlib.sha256(payload).digest(), frame))
    permutation = tuple(frame for _, frame in sorted(keyed))
    if permutation == tuple(range(frames)):
        permutation = permutation[1:] + permutation[:1]
    return permutation


def permutation_set_sha256(sample_ids: Sequence[str], frames: int = FRAMES) -> str:
    """Seal an ordered sample-ID list and its frozen frame permutations."""
    if isinstance(sample_ids, (str, bytes)) or not isinstance(sample_ids, Sequence):
        raise ValueError("sample IDs must be a sequence of strings")
    if type(frames) is not int or not 2 <= frames <= 256:
        raise ValueError("frames must be an integer between 2 and 256")
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        permutation = orderless_permutation(sample_id, frames=frames)
        encoded = sample_id.encode("utf-8")
        digest.update(struct.pack(">I", len(encoded)))
        digest.update(encoded)
        digest.update(bytes(permutation))
    return digest.hexdigest()


def _validate_image(tensor: torch.Tensor, name: str, channels: int) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.layout != torch.strided or not tensor.is_floating_point():
        raise TypeError(f"{name} must be a dense real floating-point tensor")
    if tensor.ndim != 4 or tensor.shape[1] != channels or any(size <= 0 for size in tensor.shape):
        raise ValueError(f"{name} must have nonempty Bx{channels}xHxW shape, got {tuple(tensor.shape)}")


def _validate_matching_image(
    tensor: torch.Tensor, reference: torch.Tensor, name: str, *, spatial: bool,
) -> None:
    _validate_image(tensor, name, channels=1)
    if (
        tensor.shape[0] != reference.shape[0]
        or tensor.dtype != reference.dtype
        or tensor.device != reference.device
        or (spatial and tensor.shape[-2:] != reference.shape[-2:])
    ):
        dimensions = "batch/H/W/dtype/device" if spatial else "batch/dtype/device"
        raise ValueError(f"{name} must match spike {dimensions}")


def make_legacy_carrier(temporal: torch.Tensor, blur_gray: torch.Tensor) -> torch.Tensor:
    """Build ``[16 zeros, 9 temporal channels, 1 grayscale blur]``.

    ``blur_gray`` is a B x 1 x Hb x Wb tensor matching temporal batch,
    dtype, and device. Bilinear resizing uses ``align_corners=False``.
    This carrier alone does not implement or supply a downstream backbone.
    """
    _validate_image(temporal, "temporal", channels=BINS)
    _validate_matching_image(blur_gray, temporal, "blur_gray", spatial=False)
    background = F.interpolate(
        blur_gray, size=temporal.shape[-2:], mode="bilinear", align_corners=False,
    )
    zeros = temporal.new_zeros(temporal.shape[0], 16, *temporal.shape[-2:])
    return torch.cat((zeros, temporal, background), dim=1)


class StructuredSpikeEncoder(nn.Module):
    """Encode native B x 56 x H x W spikes into B x 9 x H x W.

    Modes:
        fixed: analytical timestamp-weighted triangular projection.
        shared: same kernel plus a shared MLP residual (1,021 trainable).
        full: analytical matrix plus an unconstrained 9 x 56 residual
            (504 trainable, including its first column).
        countgrid: remove the within-bin timestamp factor while preserving
            Fixed's nonzero support and per-bin response to constant spikes.
        summary: externally supplied [TFI, TFP, 0, ..., 0].
        sum_dense: spread the same two summaries through the fixed rank-two,
            nonnegative, column-normalized 9 x 2 matrix.
        shuffled: permute each sample's 56 frames using its stable ID, then
            apply Fixed. This is not a permutation-invariant representation.

    ``initialization`` and ``replicate`` configure Full via the
    published-permutation initialization helper. See
    ``build_published_projection`` for accepted families. The separate
    byte-preserved strict helper retains its original runtime-dependent gates.
    Full initialization is on CPU in float32; use standard ``.to(device, dtype)``
    afterwards. The MLP follows PyTorch's default device/dtype at construction.
    ``initialization_metadata`` describes construction only, not
    later training, dtype conversion, or loading another state dict.

    All modes retain the original value-layer parameter inventory, frozen
    outside Shared. Thus Fixed has zero *trainable* parameters but still
    contains the frozen 1,021-parameter MLP. Model construction consumes RNG;
    encoding and sample permutations do not. This isolated module does not
    claim RNG-equivalent construction of the complete experimental network.
    """

    def __init__(
        self, mode: str = "fixed", *, initialization: str = "analytical", replicate: int = 0,
    ):
        super().__init__()
        if mode not in _REPRESENTATIONS:
            raise ValueError(f"unknown mode {mode!r}; expected {sorted(_REPRESENTATIONS)}")
        if mode != "full" and (initialization != "analytical" or replicate != 0):
            raise ValueError("custom initialization and replicate are only available for full")
        self.mode = mode
        self.representation = _REPRESENTATIONS[mode]
        self.length_spike = FRAMES
        self.bins = BINS
        self.value_layer = TemporalValueLayer(num_bins=BINS)
        if mode != "shared":
            self.value_layer.requires_grad_(False)
        position = torch.arange(1, 10, dtype=torch.float32)
        self.register_buffer("summary_dense_matrix", torch.stack((position, 10.0 - position), dim=1) / 45.0)
        self.register_parameter("full_projection_residual", None)
        self.initialization_metadata = None
        if mode == "full":
            fixed = fixed_projection(device="cpu")
            target, metadata = build_published_projection(fixed, initialization, replicate)
            self.full_projection_residual = nn.Parameter(target - fixed)
            self.initialization_metadata = metadata

    def _time_basis(self, spike: torch.Tensor):
        # Compatibility interface for the frozen apply_initialization helper.
        return _time_basis(device=spike.device, dtype=spike.dtype)

    def _permuted_spikes(self, spike: torch.Tensor, sample_ids: Sequence[str] | None):
        if (
            isinstance(sample_ids, (str, bytes))
            or not isinstance(sample_ids, Sequence)
            or len(sample_ids) != spike.shape[0]
        ):
            raise ValueError("Shuffled requires one stable sample ID per batch element")
        indices = torch.tensor(
            [orderless_permutation(sample_id) for sample_id in sample_ids],
            device=spike.device, dtype=torch.long,
        ).view(spike.shape[0], FRAMES, 1, 1)
        return torch.gather(spike, 1, indices.expand_as(spike))

    def encode(
        self, spike: torch.Tensor, *, tfi: torch.Tensor | None = None,
        tfp: torch.Tensor | None = None, sample_ids: Sequence[str] | None = None,
    ) -> torch.Tensor:
        """Encode spikes; summaries/IDs are used only by their relevant modes.

        TFI and TFP must each be B x 1 x H x W with matching dtype/device in
        Summary and SumDense. The module deliberately does not compute them.
        Shared requires its MLP dtype/device to match the input; move the
        module with ``.to(...)`` before calling it.
        """
        _validate_image(spike, "spike", channels=FRAMES)
        if self.mode in ("summary", "sum_dense"):
            _validate_matching_image(tfi, spike, "TFI", spatial=True)
            _validate_matching_image(tfp, spike, "TFP", spatial=True)
            summaries = torch.cat((tfi, tfp), dim=1)
            if self.mode == "summary":
                return torch.cat((summaries, spike.new_zeros(spike.shape[0], BINS - 2, *spike.shape[-2:])), dim=1)
            matrix = self.summary_dense_matrix.to(device=spike.device, dtype=spike.dtype)
            return torch.einsum("bchw,kc->bkhw", summaries, matrix)

        timestamps, delta = self._time_basis(spike)
        if self.mode == "shared":
            parameter = next(self.value_layer.parameters())
            if parameter.device != spike.device or parameter.dtype != spike.dtype:
                raise ValueError("Shared parameters must match spike dtype/device; move the encoder with .to(...)")
            kernel = self.value_layer(delta)
        else:
            kernel = self.value_layer.trilinear(delta)

        if self.mode == "countgrid":
            active = (timestamps != 0).to(dtype=kernel.dtype).unsqueeze(0)
            denominator = (kernel * active).sum(dim=1)
            gains = (timestamps.unsqueeze(0) * kernel).sum(dim=1) / denominator
            weights = gains.unsqueeze(1) * kernel * active
        else:
            weights = timestamps.unsqueeze(0) * kernel
            if self.mode == "full":
                if self.full_projection_residual.device != spike.device:
                    raise ValueError("Full parameters must match spike device; move the encoder with .to(...)")
                # Do not multiply this residual by timestamps: all 504 entries,
                # including the first column, must remain free to learn.
                weights = weights + self.full_projection_residual.to(dtype=weights.dtype)
        native = self._permuted_spikes(spike, sample_ids) if self.mode == "shuffled" else spike
        return torch.einsum("bthw,kt->bkhw", native, weights)

    def forward(
        self, spike: torch.Tensor, *, tfi: torch.Tensor | None = None,
        tfp: torch.Tensor | None = None, sample_ids: Sequence[str] | None = None,
    ) -> torch.Tensor:
        return self.encode(spike, tfi=tfi, tfp=tfp, sample_ids=sample_ids)

    def legacy_carrier(
        self, spike: torch.Tensor, blur_gray: torch.Tensor, *,
        tfi: torch.Tensor | None = None, tfp: torch.Tensor | None = None,
        sample_ids: Sequence[str] | None = None,
    ) -> torch.Tensor:
        """Encode and return the optional 26-channel legacy carrier."""
        temporal = self.encode(spike, tfi=tfi, tfp=tfp, sample_ids=sample_ids)
        return make_legacy_carrier(temporal, blur_gray)

    def extra_repr(self) -> str:
        return f"mode={self.mode!r}, frames={FRAMES}, bins={BINS}"
