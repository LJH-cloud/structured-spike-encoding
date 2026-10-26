# Structured Spike Encoding

Code accompanying **Structured Initialization Improves Learned Spike Encoding for Image Deblurring**.

Jiahui Lv · Dawei Weng · Han Zou · Qingjie Liu

The encoder aggregates a 56-frame spike window into nine temporal channels. A structured triangular initialization is refined by learning all 504 projection coefficients. Controlled initializations change the coefficient-to-time assignment while keeping the same learnable projection.

## What is available

- Standalone PyTorch encoders: Fixed, Shared, Full, CountGrid, Summary, SumDense, and Shuffled input.
- The original, hash-verified initialization implementation, including the registered row permutation, R1/R2, and C1/C2 controls.
- The study's Restormer task adapter and the MIT-licensed official Restormer architecture.
- Raw-RGB PSNR, SSIM, MAE, MSE, and capture-level aggregation.
- The historical training settings and executable code-provenance checks.
- CPU unit tests and small forward/backward examples. No dataset is needed for these checks.

**Release scope:** this is a reusable encoder/model-component release, not a one-command reproduction of every paper experiment. Pretrained checkpoints, datasets, sample-ID lists, experiment result tables, the end-to-end historical training/data pipeline, and the SpkDeblurNet CNN backbone are not included. Summary/SumDense accept externally prepared TFI/TFP images. See [reproducibility](docs/REPRODUCIBILITY.md) for the exact boundaries and [third-party notices](THIRD_PARTY_NOTICES.md) for source permissions.

## Installation and checks

```bash
git clone https://github.com/LJH-cloud/structured-spike-encoding.git
cd structured-spike-encoding
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,restormer]'
python -m pytest -q
python scripts/verify_release.py
python examples/encoder_demo.py
```

Python 3.10+ and PyTorch are required. Historical initialization hashes are checked rather than silently relaxed when a platform changes floating-point construction. Tests establish implementation properties, not retraining performance; consult [validation](docs/VALIDATION.md) for the environment actually tested.

## Use the encoder

```python
import torch
from structured_spike.encoding import StructuredSpikeEncoder

spikes = torch.randint(0, 2, (2, 56, 24, 32)).float()
encoder = StructuredSpikeEncoder("full")
encoded = encoder(spikes)              # [2, 9, 24, 32]
encoded.square().mean().backward()     # all 504 coefficients are trainable

# Same Full parameterization, but one common permutation of time columns:
control = StructuredSpikeEncoder("full", initialization="common_column", replicate=1)
```

`initialization="registered_row"` selects the original Permuted init.; `"independent_row", replicate=1/2` selects R1/R2; `"common_column", replicate=1/2` selects C1/C2. Additional row controls use prespecified structural screening, not outcome-based selection. Common columns preserve initial singular values and row Gram matrices, not encoded data covariance.

`legacy_carrier(spikes, blur_gray, ...)` constructs the study's 26-channel adapter input (16 zero channels, nine temporal channels, and resized blurred grayscale). Those zero channels are adapter bookkeeping, not additional observed spike channels. See [the Restormer example](examples/restormer_forward.py) for a complete randomly initialized forward pass.

## Experimental protocol and data

`configs/study.json` records the original settings and the **14 fresh + four explicitly resumed CNN runs** among the 18 added controls. It is a protocol description, not a ready-to-run training configuration. Obtain actual data from its providers under their terms. Do not substitute the ordinary GoPro RGB dataset for the paired spike/RGB data and assume identical inputs.

Result CSVs and sample-ID lists are not distributed in this code-only release. The verification script checks the distributed source identities; its optional aggregate-verification branch requires separately supplied result and split files. No score reproduction is claimed by the default code check.

## License and acknowledgments

Original code is released under [MIT](LICENSE). The bundled official Restormer implementation retains its own [MIT notice](structured_spike/third_party/RESTORMER_LICENSE.md). The work builds on spike-guided restoration and temporal aggregation research; independent code release does not replace attribution to those methods.

The SpkDeblurNet repository did not contain an explicit redistribution license at the inspected commit. It is linked as an external reference and is not redistributed here. A private fork or a citation is not treated as a substitute license.
