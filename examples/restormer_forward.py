"""Randomly initialized model-component forward pass; no weights/data downloaded."""
import torch
import torch.nn.functional as F
from structured_spike.encoding import StructuredSpikeEncoder
from structured_spike.restormer_adapter import RestormerSpikeAdapter


def main():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    spikes = torch.randint(0, 2, (1, 56, 12, 16)).float()
    blur = torch.rand(1, 3, 24, 32)
    # A caller must supply the actual grayscale conversion used in its pipeline.
    # This random tensor is only an interface fixture, not image preprocessing.
    blur_gray = torch.rand(1, 1, 24, 32)
    encoder = StructuredSpikeEncoder("full")
    backbone = RestormerSpikeAdapter().eval()
    with torch.no_grad():
        carrier = encoder.legacy_carrier(spikes, blur_gray)
        guidance = F.interpolate(blur_gray, size=spikes.shape[-2:], mode="bilinear", align_corners=False)
        main, recon, quick = backbone(blur, carrier, guidance)
    print("PASS:", tuple(main.shape), tuple(recon.shape), tuple(quick.shape))


if __name__ == "__main__":
    main()
