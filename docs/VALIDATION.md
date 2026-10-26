# Release validation — 2026-10-26

This record concerns the portable code release. It is not a new training experiment.

## Executed locally

- Environment: macOS 26.5.1, ARM64, Python 3.12.14, PyTorch 2.14.1, NumPy 2.5.4, einops 0.8.2, pytest 9.1.1.
- Editable installation with `python -m pip install -e '.[test,restormer]'`: successful.
- `python -m pytest -q -rs`: **125 passed, 18 skipped, 25 subtests passed**.
- Seven skips require CUDA, which is unavailable in this release-validation environment.
- Eleven skips belong to the original strict selector suite: historical bitwise reduction gates do not match this ARM CPU. The public loader instead passes mandatory byte-identity checks for all six published initializations. No historical acceptance criterion was weakened and no new control was selected.
- `python scripts/verify_release.py`: source identities passed. An additional **4,160 numeric checks** passed locally using the prepared but unpublished result tables and split IDs; these checks cannot be rerun from the code-only checkout without those files. The default public checkout reports zero result-table checks.
- `python examples/encoder_demo.py`: forward and backward passed, 504 trainable coefficients.
- `python examples/restormer_forward.py`: actual randomly initialized adapter forward passed, producing the main RGB, grayscale reconstruction, and quick RGB heads.
- Wheel build succeeded; package data includes selected permutation metadata and the complete Restormer license.

## Comparison with frozen implementation

The original encoding methods were extracted from the hash-verified source in memory for comparison. All seven 26-channel carriers were bitwise equal on CPU float32, including nonzero Shared MLP and Full-matrix residuals. Metric functions were likewise compared for float32/float64 and 11x11, 13x17, and 24x32 inputs: all four metric tensors were bitwise equal in all six cases. These checks preserve the formulas; they do not establish whole-model RNG identity or new training scores.

The sealed initialization module and official Restormer source retain their original bytes. The adapter differs only in one import path. No result CSVs, sample-ID lists, raw images, private history, checkpoints, cluster paths, or credentials are in the public release file set.

## Not validated here

CUDA behavior of the portable release, full-size inference with published checkpoints, data decoding/augmentation, distributed training, and end-to-end reproduction of paper scores have not been rerun. No claim of such validation is implied by passing CPU tests or the table-reconstruction checks.
