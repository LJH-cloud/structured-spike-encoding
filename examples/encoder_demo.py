"""CPU forward/backward smoke check, not a trained restoration experiment."""
import torch
from structured_spike.encoding import StructuredSpikeEncoder


def main():
    torch.manual_seed(7)
    spikes = torch.randint(0, 2, (2, 56, 12, 16)).float()
    model = StructuredSpikeEncoder("full", initialization="common_column", replicate=1)
    result = model(spikes)
    result.square().mean().backward()
    parameters = [p for p in model.parameters() if p.requires_grad]
    assert result.shape == (2, 9, 12, 16)
    assert sum(p.numel() for p in parameters) == 504
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    print("PASS: Full/C1 output", tuple(result.shape), "with 504 trainable coefficients")


if __name__ == "__main__":
    main()
