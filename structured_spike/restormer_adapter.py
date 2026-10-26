"""Restormer-based task adapter with three independent, trainable output heads.

The unmodified official Restormer modules live in ``source/restormer_official.py``.
This is a task adapter, not an off-the-shelf RGB Restormer or a new method.
Inputs are already prepared by the caller; no spike encoding, TFI generation,
target access, normalization, or gradient detachment is performed here.
"""

import torch
from torch import nn

from .third_party.restormer_arch import Restormer


def _reflect_axis_right(x: torch.Tensor, amount: int, axis: int) -> torch.Tensor:
    """Append reflected values without ReflectionPad's CUDA backward.

    For ``amount < size`` this equals ordinary right/bottom reflection padding.
    Larger padding uses the same periodic reflection of the original axis.
    Each index_select has distinct indices, including for very small images;
    repeated reflected positions are put in separate concatenated branches.
    All operations remain differentiable. A singleton axis cannot be reflected.
    """
    if amount == 0:
        return x
    size = x.shape[axis]
    if size < 2:
        raise ValueError("Reflection padding requires spatial dimensions >= 2")

    if amount < size:
        indices = torch.arange(
            size - 2, size - amount - 2, -1, device=x.device, dtype=torch.long
        )
        return torch.cat((x, x.index_select(axis, indices)), dim=axis)

    # Only small edge cases reach this branch (padding is at most 7 pixels).
    # Build indices from shape metadata, never from image values.
    period = 2 * (size - 1)
    pieces = [x]
    block = []
    used = set()
    for position in range(size, size + amount):
        offset = position % period
        index = offset if offset < size else period - offset
        if index in used:
            indices = torch.tensor(block, device=x.device, dtype=torch.long)
            pieces.append(x.index_select(axis, indices))
            block = []
            used = set()
        block.append(index)
        used.add(index)
    if block:
        indices = torch.tensor(block, device=x.device, dtype=torch.long)
        pieces.append(x.index_select(axis, indices))
    return torch.cat(pieces, dim=axis)


def _reflect_pad_to_multiple(x: torch.Tensor, multiple: int) -> torch.Tensor:
    """Pad only the right and bottom; aligned inputs are returned unchanged."""
    height, width = x.shape[-2:]
    x = _reflect_axis_right(x, (-width) % multiple, axis=-1)
    return _reflect_axis_right(x, (-height) % multiple, axis=-2)


class RestormerSpikeAdapter(nn.Module):
    """Fixed full-size Restormer core plus 16,560 adapter parameters.

    ``forward(blur, spike_input, common_guidance)`` accepts floating tensors:
      * blur:            [B, 3,  H, W]
      * spike_input:     [B, 26, h, w] (16 zero + 9 temporal + background)
      * common_guidance: [B, 1,  h, w]
    where H = 2*h and W = 2*w. Inputs must share dtype and device. Spatial
    dimensions at native spike resolution must be at least two pixels.

    Returns ``(main, recon, quick)`` with shapes [B,3,H,W], [B,1,H,W],
    and [B,3,H,W]. ``quick`` is blur-only. Both main and recon can depend
    on all 26 spike channels and the supplied common guidance. As with the
    official core, autocast may change intermediate/output-head dtypes;
    no explicit dtype conversion is applied by this adapter.

    The architecture has no arm-dependent settings or extra norm/dropout.
    Added convolutions are fixed to bias=False, matching the core convention.
    """

    def __init__(self):
        super().__init__()
        self.core = Restormer(
            inp_channels=3,
            out_channels=3,
            dim=48,
            num_blocks=[4, 6, 6, 8],
            num_refinement_blocks=4,
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type="WithBias",
            dual_pixel_task=False,
        )
        self.quick_head = nn.Conv2d(48, 3, kernel_size=3, padding=1, bias=False)
        self.fusion = nn.Conv2d(96 + 26 + 1, 96, kernel_size=1, bias=False)
        self.recon_head = nn.Sequential(
            nn.Conv2d(96, 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def _validate_inputs(self, blur, spike_input, common_guidance):
        inputs = (
            ("blur", blur, 3),
            ("spike_input", spike_input, 26),
            ("common_guidance", common_guidance, 1),
        )
        for name, value, channels in inputs:
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.ndim != 4 or value.shape[1] != channels:
                raise ValueError(f"{name} must have shape [B, {channels}, H, W]")
            if not value.is_floating_point():
                raise TypeError(f"{name} must have a floating-point dtype")
            if value.layout != torch.strided:
                raise TypeError(f"{name} must have strided tensor layout")
            if value.shape[0] < 1 or min(value.shape[-2:]) < 2:
                raise ValueError(f"{name} needs B >= 1 and both spatial dimensions >= 2")
        for name, value, _ in inputs[1:]:
            if value.shape[0] != blur.shape[0]:
                raise ValueError(f"{name} and blur must have the same batch size")
            if value.dtype != blur.dtype:
                raise TypeError(f"{name} and blur must have the same dtype")
            if value.device != blur.device:
                raise ValueError(f"{name} and blur must be on the same device")
        if spike_input.shape[-2:] != common_guidance.shape[-2:]:
            raise ValueError("spike_input and common_guidance must share native resolution")
        height, width = blur.shape[-2:]
        native_height, native_width = spike_input.shape[-2:]
        if (height, width) != (2 * native_height, 2 * native_width):
            raise ValueError("blur must be exactly 2x the native spike/guidance resolution")
        if blur.device != self.core.patch_embed.proj.weight.device:
            raise ValueError("Inputs and model parameters must be on the same device")

    def forward(self, blur, spike_input, common_guidance):
        self._validate_inputs(blur, spike_input, common_guidance)
        height, width = blur.shape[-2:]
        padded_blur = _reflect_pad_to_multiple(blur, 8)
        padded_spike = _reflect_pad_to_multiple(spike_input, 4)
        padded_guidance = _reflect_pad_to_multiple(common_guidance, 4)

        # Original level 1 sees only RGB; its independent quick head cannot
        # receive a forward dependency from either spike or guidance.
        inp_enc_level1 = self.core.patch_embed(padded_blur)
        out_enc_level1 = self.core.encoder_level1(inp_enc_level1)
        quick = self.quick_head(out_enc_level1) + padded_blur

        inp_enc_level2 = self.core.down1_2(out_enc_level1)
        if inp_enc_level2.shape[-2:] != padded_spike.shape[-2:]:
            raise RuntimeError("Padded RGB and native spike resolutions are misaligned")
        inp_enc_level2 = self.fusion(
            torch.cat((inp_enc_level2, padded_spike, padded_guidance), dim=1)
        )
        out_enc_level2 = self.core.encoder_level2(inp_enc_level2)
        recon = self.recon_head(out_enc_level2)

        # The remaining encoder, latent, decoder, skips, refinement and RGB
        # residual follow the original official Restormer forward verbatim.
        inp_enc_level3 = self.core.down2_3(out_enc_level2)
        out_enc_level3 = self.core.encoder_level3(inp_enc_level3)
        inp_enc_level4 = self.core.down3_4(out_enc_level3)
        latent = self.core.latent(inp_enc_level4)

        inp_dec_level3 = self.core.up4_3(latent)
        inp_dec_level3 = torch.cat((inp_dec_level3, out_enc_level3), dim=1)
        inp_dec_level3 = self.core.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.core.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.core.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat((inp_dec_level2, out_enc_level2), dim=1)
        inp_dec_level2 = self.core.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.core.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.core.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat((inp_dec_level1, out_enc_level1), dim=1)
        out_dec_level1 = self.core.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.core.refinement(out_dec_level1)
        main = self.core.output(out_dec_level1) + padded_blur

        return (
            main[:, :, :height, :width],
            recon[:, :, :height, :width],
            quick[:, :, :height, :width],
        )
