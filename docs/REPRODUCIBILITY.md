# Reproducibility and integration

## Release identity

This release separates the paper's new encoder/control code from the historical cluster execution environment. `initialization.py` is byte-for-byte the sealed initialization source. `encoding.py` extracts its temporal encoding interfaces; it does not instantiate the original CNN and therefore does not reproduce the original whole-model construction RNG stream. The Restormer adapter is the sealed adapter with one import-path relocation; its official backbone is unchanged.

The original row-control selector uses exact float32 reduction comparisons. Reduction order can differ across CPU architectures even when the initial matrix bytes agree. The public encoder therefore loads the **already selected experimental permutations** from `published_initializations.json`. It verifies the fixed, target, residual, and effective matrix SHA-256 identities. It does not choose replacement controls on the new host. The strict original selector remains available for a matching historical runtime; its compatibility tests report skips on a backend with different bitwise reductions. Portable matrix-identity tests remain mandatory.

`SOURCE_PROVENANCE.json` records source identities and adaptation boundaries. These are code/matrix provenance checks, not evidence that new training runs reproduce paper scores.

## What is tested

The CPU suite exercises encoding formulas, shared/full parameter counts, gradients, selected permutation identities, common-column invariants, raw-RGB metric formulas, and the actual Restormer task adapter. Examples use synthetic tensors with no learned checkpoints. No restoration score is inferred from those smoke checks. See `VALIDATION.md` for actual outcomes.

## Original study settings

`configs/study.json` records the settings, not an executable end-to-end training job. All initialization treatments optimize the same 9x56 Full matrix. CNN adds R1/R2/C1/C2 and Transformer adds R1/C1, each at three seeds. Analytical Full, Fixed, and the registered row-permuted baseline are historical references. All selected checkpoints are final 100k-update endpoints, not validation-best checkpoints.

Four CNN runs were resumed from 10k with the original training configuration. Their historical random states were not completely saved; they must not be described as uninterrupted fresh trajectories. The other 14 new runs completed without that recovery. This release does not rerun any experiment or distribute experiment result tables.

The 18 added initialization controls were evaluated for PSNR, SSIM, MAE, and MSE. The paper's 39-model core study has different coverage, including perceptual/real-data evaluations. Do not extrapolate LPIPS, DISTS, LIQE, BRISQUE, X4K, or RSB results to the new controls.

## Splits and input contract

- The historical splits contain 1,678 training, 425 development, and 1,111 test images. Identifier lists are not distributed in this code-only release.
- Historical stable identifiers have the form `split__sequence__blur_gamma__frame.png`. Replacing `__` with `/` locates the corresponding RGB input relative to the paired dataset root, not necessarily to a stock GoPro download.
- A parent capture is `sample_id.split('__')[1].rsplit('_', 1)[0]`. Development parents are held out from training. Test-parent overlap is explicitly represented by seen/unseen-parent strata in result tables; no all-test-parent-disjoint claim is made.
- RGB and target inputs have twice the spatial dimensions of native spikes. The centered 56 native spike frames feed the encoder. The legacy adapter carrier is `[zero16, temporal9, blurred_grayscale1]`, with common guidance supplied separately.
- Summary and SumDense require TFI/TFP from the same augmented spike window. The historical setting is center index 28, default window 50, and output divided by 0.5. TFI's legacy left index-0 sentinel and missing-event fallback affect exact replication; a generic upstream TFI implementation is not automatically interchangeable. The toolbox is excluded because its redistribution permission was not established.

Obtain paired spike/RGB data from its provider. No images, raw `.dat` recordings, pretrained weights, or sample-ID lists are included.

## Metrics and tables

PSNR/SSIM/MAE/MSE operate on raw RGB outputs, without clipping or uint8 conversion. SSIM uses an 11x11 valid Gaussian window, sigma 1.5, constants .01²/.03², then averages spatial maps and RGB channels. Compute PSNR per image before aggregation. MSE=0 yields +infinity; the released finite-score tables contain no perfect reconstruction samples.

Headline aggregation assigns equal weight to each parent capture, then summarizes three training seeds. Standard deviations of improvements are computed after pairing seeds, using sample SD (ddof=1). Stored SSIM values are raw, not the paper's displayed 10^-3 multiples. The optional result-verification branch requires separately supplied tables and sample-ID lists; with the code-only checkout it verifies source hashes only. It does not evaluate withheld checkpoints or reconstruct source images.

## Full training reproduction

The remaining work for a full end-to-end release includes a redistributable CNN backbone, permission-resolved preprocessing, portable training/data entry points, checkpoint distribution, and full GPU verification. An upstream clone plus these encoder components is **not** claimed to reproduce every historical run. Official Restormer is included under MIT; the SpkDeblurNet source is linked externally rather than relicensed. No cluster credentials, absolute cluster paths, private Git history, or execution logs are shipped.
