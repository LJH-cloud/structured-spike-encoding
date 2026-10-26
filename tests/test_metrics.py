"""CPU tests for raw-RGB metrics and the unit of aggregation."""

import math
import unittest

import torch

from structured_spike.metrics import (
    aggregate_capture_macro,
    capture_group,
    metrics_per_sample,
)


def valid_rgb_ssim_reference(pred, gt):
    """Independent window-slicing oracle; deliberately no padded convolution."""
    axis = torch.arange(11, dtype=pred.dtype, device=pred.device) - 5
    weights = torch.exp(-(axis.square()) / (2 * 1.5**2))
    weights = weights / weights.sum()
    window = weights[:, None] * weights[None, :]
    left = pred.unfold(2, 11, 1).unfold(3, 11, 1)
    right = gt.unfold(2, 11, 1).unfold(3, 11, 1)
    mu1 = (left * window).sum((-2, -1))
    mu2 = (right * window).sum((-2, -1))
    var1 = (left.square() * window).sum((-2, -1)) - mu1.square()
    var2 = (right.square() * window).sum((-2, -1)) - mu2.square()
    covariance = (left * right * window).sum((-2, -1)) - mu1 * mu2
    numerator = (2 * mu1 * mu2 + 0.01**2) * (2 * covariance + 0.03**2)
    denominator = (mu1.square() + mu2.square() + 0.01**2) * (var1 + var2 + 0.03**2)
    return (numerator / denominator).flatten(1).mean(1)


class ImageMetricTests(unittest.TestCase):
    def test_rgb_ssim_matches_valid_window_oracle(self):
        generator = torch.Generator().manual_seed(472)
        gt = torch.rand((2, 3, 13, 14), generator=generator, dtype=torch.float64)
        pred = gt.clone()
        pred[:, 0] += 0.3
        pred[:, 1, 0, :] = -0.8
        pred[:, 2, -1, :] = 1.4
        psnr, ssim, mae, mse = metrics_per_sample(pred, gt)
        torch.testing.assert_close(ssim, valid_rgb_ssim_reference(pred, gt), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(mse, (pred - gt).square().flatten(1).mean(1))
        torch.testing.assert_close(mae, (pred - gt).abs().flatten(1).mean(1))
        torch.testing.assert_close(psnr, -10 * torch.log10(mse))
        luminance = torch.tensor([0.2989, 0.587, 0.114], dtype=pred.dtype).view(1, 3, 1, 1)
        gray_pred = (pred * luminance).sum(1, keepdim=True).expand(-1, 3, -1, -1)
        gray_gt = (gt * luminance).sum(1, keepdim=True).expand(-1, 3, -1, -1)
        self.assertFalse(torch.allclose(ssim, valid_rgb_ssim_reference(gray_pred, gray_gt)))

    def test_predictions_are_not_clamped_and_inputs_are_unchanged(self):
        pred = torch.full((2, 3, 11, 11), 2.0, dtype=torch.float64)
        pred[1] = -1.0
        gt = torch.ones_like(pred)
        before_pred, before_gt = pred.clone(), gt.clone()
        psnr, _, mae, mse = metrics_per_sample(pred, gt)
        torch.testing.assert_close(mse, torch.tensor([1.0, 4.0], dtype=pred.dtype))
        torch.testing.assert_close(mae, torch.tensor([1.0, 2.0], dtype=pred.dtype))
        torch.testing.assert_close(psnr, torch.tensor([0.0, -10 * math.log10(4)], dtype=pred.dtype))
        self.assertTrue(torch.equal(pred, before_pred))
        self.assertTrue(torch.equal(gt, before_gt))
        self.assertFalse(torch.equal(mse, metrics_per_sample(pred.clamp(0, 1), gt)[3]))

    def test_perfect_reconstruction_preserves_positive_infinite_psnr(self):
        image = torch.rand((1, 3, 11, 11), dtype=torch.float64)
        psnr, ssim, mae, mse = metrics_per_sample(image, image)
        self.assertTrue(torch.isposinf(psnr).all())
        torch.testing.assert_close(ssim, torch.ones_like(ssim))
        self.assertEqual(mae.item(), 0)
        self.assertEqual(mse.item(), 0)

    def test_float32_dtype_and_batch_are_preserved(self):
        pred = torch.zeros((3, 3, 12, 12), dtype=torch.float32)
        gt = torch.ones_like(pred)
        first = metrics_per_sample(pred, gt)
        second = metrics_per_sample(pred, gt)
        for left, right in zip(first, second):
            self.assertEqual(left.shape, (3,))
            self.assertEqual(left.dtype, torch.float32)
            self.assertTrue(torch.equal(left, right))

    def test_invalid_shapes_dtypes_and_nonfinite_inputs_fail(self):
        valid = torch.zeros((1, 3, 11, 11), dtype=torch.float32)
        invalid_pairs = [
            (valid[0], valid[0]),
            (valid[:, :1], valid[:, :1]),
            (valid[:, :, :10], valid[:, :, :10]),
            (valid[:0], valid[:0]),
            (valid, valid.expand(2, -1, -1, -1)),
            (valid, valid.double()),
            (valid.long(), valid.long()),
            (valid.half(), valid.half()),
            (valid, torch.full_like(valid, float("nan"))),
            (torch.full_like(valid, float("inf")), valid),
            (valid, torch.empty(valid.shape, device="meta")),
        ]
        for pred, gt in invalid_pairs:
            with self.subTest(pred_shape=pred.shape, dtype=pred.dtype, gt_device=gt.device):
                with self.assertRaises((TypeError, ValueError)):
                    metrics_per_sample(pred, gt)
        with self.assertRaises(TypeError):
            metrics_per_sample(None, valid)

    def test_finite_inputs_that_overflow_metrics_fail_explicitly(self):
        pred = torch.full((1, 3, 11, 11), 1e30, dtype=torch.float32)
        with self.assertRaises(ValueError):
            metrics_per_sample(pred, torch.zeros_like(pred))


class CaptureAggregationTests(unittest.TestCase):
    @staticmethod
    def record(sample_id, capture, psnr, mse, ssim=0.8, mae=0.1):
        return dict(sample_id=sample_id, capture=capture, psnr=psnr, ssim=ssim, mae=mae, mse=mse)

    def test_capture_macro_equal_weights_and_preserves_image_psnr(self):
        records = [
            self.record("A1", "A", 10, 0.1),
            self.record("A2", "A", 30, 0.001),
            self.record("B1", "B", 40, 0.0001),
        ]
        summary = aggregate_capture_macro(records)
        self.assertEqual(summary["image_count"], 3)
        self.assertEqual(summary["capture_count"], 2)
        self.assertEqual(summary["per_capture"]["A"]["count"], 2)
        self.assertEqual(summary["per_capture"]["A"]["psnr"], 20)
        self.assertEqual(summary["capture_macro"]["psnr"], 30)
        self.assertAlmostEqual(summary["image_micro"]["psnr"], 80 / 3)
        self.assertNotAlmostEqual(summary["capture_macro"]["psnr"], -10 * math.log10(summary["capture_macro"]["mse"]))
        self.assertEqual(summary, aggregate_capture_macro(iter(reversed(records))))

    def test_capture_can_be_derived_from_frozen_sample_id(self):
        sample_id = "train__GOPR0372_07_00__blur_gamma__000047.png"
        self.assertEqual(capture_group(sample_id), "GOPR0372_07")
        record = self.record(sample_id, "GOPR0372_07", 20, 0.01)
        del record["capture"]
        summary = aggregate_capture_macro([record])
        self.assertEqual(tuple(summary["per_capture"]), ("GOPR0372_07",))
        self.assertEqual(summary["capture_macro"], summary["image_micro"])

    def test_positive_infinite_psnr_requires_exact_reconstruction(self):
        record = self.record("A1", "A", float("inf"), 0, ssim=1, mae=0)
        summary = aggregate_capture_macro([record])
        self.assertEqual(summary["capture_macro"]["psnr"], float("inf"))
        record["mse"] = 0.1
        with self.assertRaises(ValueError):
            aggregate_capture_macro([record])

    def test_invalid_records_and_duplicate_ids_fail(self):
        good = self.record("A1", "A", 20, 0.01)
        with self.assertRaises(ValueError):
            aggregate_capture_macro([])
        with self.assertRaises(ValueError):
            aggregate_capture_macro([good, good])
        with self.assertRaises(TypeError):
            aggregate_capture_macro([None])
        for field, value in [("sample_id", ""), ("capture", ""), ("psnr", float("nan")),
                             ("psnr", float("-inf")), ("ssim", float("inf")),
                             ("mse", -1), ("mae", -1), ("psnr", True), ("psnr", "20")]:
            record = dict(good, **{field: value})
            with self.subTest(field=field, value=value):
                with self.assertRaises((TypeError, ValueError)):
                    aggregate_capture_macro([record])
        for field in ("sample_id", "psnr", "ssim", "mae", "mse"):
            record = dict(good)
            del record[field]
            with self.subTest(missing=field):
                with self.assertRaises(ValueError):
                    aggregate_capture_macro([record])
        with self.assertRaises(ValueError):
            capture_group("")


if __name__ == "__main__":
    unittest.main(verbosity=2)
