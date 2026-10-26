"""The study's raw-RGB image metrics and capture-balanced aggregation.

The image formulas come from the study-authored ``eval_iclr_controls.py``
(SHA-256 below), not from a third-party metrics package. Only the small metric
helpers are retained here; no model, historical trainer, or data-loader imports
are needed. Input validation and the public aggregation wrapper are new.

Evaluate normalized RGB images with nominal data range one, outside autocast.
Predictions are deliberately NOT clamped to [0, 1]. SSIM uses an 11x11 Gaussian
valid window, sigma 1.5, and constants 0.01**2 and 0.03**2. Full-image evaluation,
checkpoint choice, and dataset preparation remain the caller's responsibility.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from numbers import Real
from typing import Any

import torch
import torch.nn.functional as F


METRIC_PROTOCOL = "rgb_unclamped_valid11_v1"
METRIC_SOURCE_SHA256 = "b7fdea3707a11bbde1fdb1dea74c2ab0145519aa435859e437fa3ece2f024ca1"
METRIC_NAMES = ("psnr", "ssim", "mae", "mse")
_WINDOW_CACHE = {}


def _gaussian_window(channels, device, dtype, win=11, sigma=1.5):
    key = (channels, device.type, device.index, dtype, win, sigma)
    cached = _WINDOW_CACHE.get(key)
    if cached is None:
        values = torch.tensor(
            [math.exp(-((index - win // 2) ** 2) / (2 * sigma**2)) for index in range(win)],
            device=device,
            dtype=dtype,
        )
        values = (values / values.sum()).view(1, 1, win, 1)
        cached = (values @ values.transpose(2, 3)).repeat(channels, 1, 1, 1)
        _WINDOW_CACHE[key] = cached
    return cached


def _validate_images(pred: torch.Tensor, gt: torch.Tensor) -> None:
    if not isinstance(pred, torch.Tensor) or not isinstance(gt, torch.Tensor):
        raise TypeError("pred and gt must be torch tensors")
    if pred.layout != torch.strided or gt.layout != torch.strided:
        raise ValueError("pred and gt must be dense strided tensors")
    if pred.ndim != 4 or gt.ndim != 4 or pred.shape != gt.shape:
        raise ValueError("pred and gt must have the same Bx3xHxW shape")
    if pred.shape[0] == 0 or pred.shape[1] != 3 or min(pred.shape[-2:]) < 11:
        raise ValueError("require a nonempty RGB batch with H and W at least 11")
    if pred.dtype != gt.dtype or pred.dtype not in (torch.float32, torch.float64):
        raise TypeError("pred and gt must share float32 or float64 dtype; no implicit casting")
    if pred.device != gt.device or pred.device.type == "meta":
        raise ValueError("pred and gt must share a materialized device")
    if not bool(torch.isfinite(pred).all()) or not bool(torch.isfinite(gt).all()):
        raise ValueError("pred and gt must contain only finite values")


def metrics_per_sample(
    pred: torch.Tensor, gt: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(psnr, ssim, mae, mse)``, each a vector of length B.

    The numerical formulas and their operation order are the frozen helpers.
    RGB channels are averaged directly; no luminance conversion, border crop,
    clipping, resizing, epsilon in PSNR, or variance clamp is introduced.
    Perfect reconstruction therefore has PSNR=+inf, MAE=MSE=0, and SSIM=1.
    Nonfinite inputs or overflow in the calculated metrics raise explicitly.
    """
    _validate_images(pred, gt)
    error = pred - gt
    mse = error.square().flatten(1).mean(1)
    mae = error.abs().flatten(1).mean(1)
    psnr = -10 * torch.log10(mse)
    channels = pred.shape[1]
    window = _gaussian_window(channels, pred.device, pred.dtype)
    mu1 = F.conv2d(pred, window, padding=0, groups=channels)
    mu2 = F.conv2d(gt, window, padding=0, groups=channels)
    mu1_sq, mu2_sq, mu12 = mu1.square(), mu2.square(), mu1 * mu2
    sigma1 = F.conv2d(pred * pred, window, padding=0, groups=channels) - mu1_sq
    sigma2 = F.conv2d(gt * gt, window, padding=0, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * gt, window, padding=0, groups=channels) - mu12
    numerator = (2 * mu12 + 0.01**2) * (2 * sigma12 + 0.03**2)
    denominator = (mu1_sq + mu2_sq + 0.01**2) * (sigma1 + sigma2 + 0.03**2)
    ssim = (numerator / denominator).flatten(1).mean(1)
    if not all(bool(torch.isfinite(value).all()) for value in (ssim, mae, mse)):
        raise ValueError("metric overflow or nonfinite result; inspect image scale and dtype")
    valid_psnr = torch.isfinite(psnr) | (torch.isposinf(psnr) & (mse == 0))
    if not bool(valid_psnr.all()):
        raise ValueError("nonfinite PSNR other than exact-reconstruction +inf")
    return psnr, ssim, mae, mse


def capture_group(sample_id: str) -> str:
    """Apply the frozen GoPro parent-capture grouping to a stable sample ID.

    Example: ``train__GOPR0372_07_00__blur_gamma__000047.png`` maps to
    ``GOPR0372_07``. Other datasets should supply explicit capture labels to
    :func:`aggregate_capture_macro` rather than assume this naming convention.
    """
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError("sample_id must be a nonempty string")
    parts = sample_id.split("__")
    sequence = parts[1] if len(parts) >= 2 else sample_id
    capture = sequence.rsplit("_", 1)[0]
    if not capture.strip():
        raise ValueError("sample_id contains an empty capture identifier")
    return capture


def aggregate_capture_macro(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Average supplied per-image metrics within, then equally across, captures.

    Each record needs a unique ``sample_id`` and numeric ``psnr``, ``ssim``,
    ``mae``, and ``mse``. Optional ``capture`` overrides the GoPro naming rule.
    Returns ``image_count``, ``capture_count``, ``image_micro``, ``capture_macro``,
    and ``per_capture`` (capture label -> metrics plus ``count``).

    PSNR values are averaged in dB; PSNR is never recomputed from averaged MSE.
    Captures receive equal weight regardless of image count. Apply this function
    separately for each system/arm/training seed: it neither pools seeds nor
    estimates uncertainty. Callers must verify the intended complete split;
    this helper cannot detect absent samples or missing captures from records.
    Inputs must be finite, except PSNR=+inf is allowed when the same image's
    MSE is exactly zero. Such infinity propagates rather than being discarded.
    """
    by_capture: dict[str, list[dict[str, float]]] = {}
    all_values: list[dict[str, float]] = []
    sample_ids: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"record {index} must be a mapping")
        missing = {"sample_id", *METRIC_NAMES}.difference(record)
        if missing:
            raise ValueError(f"record {index} is missing fields: {sorted(missing)}")
        sample_id = record["sample_id"]
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f"record {index} has an invalid sample_id")
        if sample_id in sample_ids:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        sample_ids.add(sample_id)
        capture = record["capture"] if "capture" in record else capture_group(sample_id)
        if not isinstance(capture, str) or not capture.strip():
            raise ValueError(f"record {index} has an invalid capture")
        values = {}
        for name in METRIC_NAMES:
            value = record[name]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"record {index} metric {name} must be a real number")
            value = float(value)
            if not math.isfinite(value) and not (name == "psnr" and value == math.inf):
                raise ValueError(f"record {index} metric {name} is nonfinite")
            values[name] = value
        if values["mse"] < 0 or values["mae"] < 0:
            raise ValueError(f"record {index} has negative MSE or MAE")
        if values["psnr"] == math.inf and values["mse"] != 0:
            raise ValueError(f"record {index} has infinite PSNR without zero MSE")
        by_capture.setdefault(capture, []).append(values)
        all_values.append(values)
    if not all_values:
        raise ValueError("cannot aggregate an empty record collection")

    def means(rows):
        return {name: math.fsum(row[name] for row in rows) / len(rows) for name in METRIC_NAMES}

    per_capture = {
        capture: dict(count=len(rows), **means(rows))
        for capture, rows in sorted(by_capture.items())
    }
    return {
        "image_count": len(all_values),
        "capture_count": len(per_capture),
        "image_micro": means(all_values),
        "capture_macro": means(list(per_capture.values())),
        "per_capture": per_capture,
    }
