# Third-party sources

## Included: official Restormer

- Project: [swz30/Restormer](https://github.com/swz30/Restormer).
- Source commit: `68dc6ac472db26f16361150cb7a96a1bc87da93f`.
- Original path: `basicsr/models/archs/restormer_arch.py`.
- Distributed path: `structured_spike/third_party/restormer_arch.py`.
- SHA-256: `3be243fa3c8e2cb2c9459eeac062b04d6084dbcc328292dae1c3e52ba6f7434a`.
- License: MIT, Copyright 2022 Syed Waqas Zamir and contributors. The complete upstream notice is in `structured_spike/third_party/RESTORMER_LICENSE.md`.
- Architecture source is unchanged. The separate task adapter is study-specific code; it is not an off-the-shelf pretrained RGB Restormer.

## Referenced, not distributed: SpkDeblurNet

- Project: [hnmizuho/SpkDeblurNet](https://github.com/hnmizuho/SpkDeblurNet/tree/fe37462c6cace294f3ce1165859188e6048e6545).
- Inspected commit: `fe37462c6cace294f3ce1165859188e6048e6545`.
- No explicit redistribution license was found in that source tree. Its backbone, mixed architecture files, data utilities, and TFI/TFP toolbox are not included in this release.
- In particular, its `arch_restormer.py` contains additional cross-modal components and is not assumed to be covered in full by the license of the separate official Restormer file.

## Data and dependencies

No dataset images, spike recordings, or pretrained weights are distributed. Dataset access and reuse remain subject to provider terms. Python dependencies are installed from their providers and retain their respective licenses. The root MIT license does not relicense external datasets or excluded software.
