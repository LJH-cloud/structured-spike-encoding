"""CPU-only math/adapter tests; no frozen repository or model/data imports.

Run: python -B -m unittest -v test_pr_initialization
The real-model encoder/VJP and CUDA initialization audits are separate gates.
"""

from __future__ import annotations

import json
import random
import unittest
from unittest import mock

import numpy as np
import torch

from structured_spike import initialization


def real_fixed_basis(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    # Frozen rgb_representation._time_basis uses linspace, not arange / 55.
    # Both represent t/55 mathematically but can differ in float32 bytes.
    timestamps = torch.linspace(0.0, 1.0, 56, dtype=dtype)
    centers = torch.linspace(0.0, 1.0, 9, dtype=dtype)
    delta = timestamps.unsqueeze(0) - centers.unsqueeze(1)
    return timestamps.unsqueeze(0) * torch.relu(1.0 - 8 * delta.abs())


class AnalyticValueLayer(torch.nn.Module):
    def trilinear(self, delta: torch.Tensor) -> torch.Tensor:
        return torch.relu(1.0 - 8 * delta.abs())


class MinimalFullModel(torch.nn.Module):
    """Adapter fixture only; not evidence for the real inherited encoder."""
    def __init__(self) -> None:
        super().__init__()
        self.representation = "learned_full9_bg"
        self.full_projection_residual = torch.nn.Parameter(torch.zeros(9, 56))
        self.value_layer = AnalyticValueLayer()
        self.backbone_weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.register_buffer("existing_buffer", torch.tensor([7.0]))

    def _time_basis(self, probe: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        timestamps = torch.linspace(0.0, 1.0, 56, dtype=probe.dtype, device=probe.device)
        centers = torch.linspace(0.0, 1.0, 9, dtype=probe.dtype, device=probe.device)
        return timestamps, timestamps.unsqueeze(0) - centers.unsqueeze(1)


class InitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixed = real_fixed_basis()
        try:
            initialization.build_projection(cls.fixed, "registered_row", 0)
        except RuntimeError as error:
            if "historical row gates failed on CPU or caller device" in str(error):
                raise unittest.SkipTest(
                    "Historical bitwise reduction gates require a matching CPU runtime; "
                    "portable published-matrix checks are tested separately"
                ) from error
            raise
        cls.arms = [("analytical", 0), ("registered_row", 0),
                    ("independent_row", 1), ("independent_row", 2),
                    ("common_column", 1), ("common_column", 2)]
        cls.results = {
            arm: initialization.build_projection(cls.fixed, *arm) for arm in cls.arms
        }

    def test_registered_exact_historical_hashes(self) -> None:
        self.assertEqual(initialization.tensor_sha256(self.fixed), initialization.REGISTERED_FIXED_SHA256)
        target, meta = self.results[("registered_row", 0)]
        self.assertEqual(meta["permutation_sha256"], initialization.REGISTERED_PERMUTATION_SHA256)
        self.assertEqual(meta["target_projection_sha256"], initialization.REGISTERED_TARGET_SHA256)
        self.assertEqual(meta["initial_residual_sha256"], initialization.REGISTERED_RESIDUAL_SHA256)
        self.assertTrue(torch.equal(self.fixed + (target - self.fixed), target))
        self.assertEqual(meta["row_nonces"], list(initialization.REGISTERED_NONCES))

    def test_analytical_is_exact_identity(self) -> None:
        target, meta = self.results[("analytical", 0)]
        self.assertTrue(torch.equal(target, self.fixed))
        self.assertTrue(torch.equal(target - self.fixed, torch.zeros_like(target)))
        self.assertTrue(meta["effective_equals_target_bitwise"])

    def test_all_multisets_reconstruction_and_json_metadata(self) -> None:
        for arm, (target, meta) in self.results.items():
            with self.subTest(arm=arm):
                self.assertEqual(target.shape, (9, 56))
                self.assertEqual(target.dtype, self.fixed.dtype)
                self.assertEqual(target.device.type, "cpu")
                self.assertTrue(torch.equal(torch.sort(target, dim=1).values,
                                            torch.sort(self.fixed, dim=1).values))
                self.assertTrue(meta["effective_equals_target_within_tolerance"])
                self.assertEqual(meta["projection_parameters"], 504)
                self.assertTrue(meta["all_row_multisets_bitwise_equal"])
                self.assertEqual(len(meta["rows"]), 9)
                json.dumps(meta, allow_nan=False)

    def test_independent_row_gates_and_distinct_realizations(self) -> None:
        hashes = {self.results[("registered_row", 0)][1]["permutation_sha256"]}
        for replicate in (1, 2):
            target, meta = self.results[("independent_row", replicate)]
            self.assertTrue(all(row["historical_row_gates_pass"] for row in meta["rows"]))
            self.assertTrue(all(row["CPU_historical_row_gates_pass"] for row in meta["rows"]))
            self.assertTrue(meta["all_row_statistics_bitwise_equal"])
            self.assertTrue(meta["effective_equals_target_bitwise"])
            self.assertTrue(all(0 <= nonce < initialization.MAX_SEARCH_NONCES for nonce in meta["row_nonces"]))
            self.assertEqual(meta["salt"], initialization.INDEPENDENT_ROW_SALTS[replicate])
            hashes.add(meta["permutation_sha256"])
            # Verify nonce minimality, not merely that the stored nonce passes.
            for row, selected_nonce in enumerate(meta["row_nonces"]):
                for nonce in range(selected_nonce):
                    permutation = initialization._sha_permutation(meta["salt"], row, nonce)
                    indices = torch.tensor(permutation, dtype=torch.long)
                    earlier = initialization._row_gate_record(self.fixed[row], self.fixed[row, indices], indices)
                    self.assertFalse(earlier["historical_row_gates_pass"])
        self.assertEqual(len(hashes), 3)

    def test_common_columns_first_candidate_and_invariants(self) -> None:
        hashes = set()
        for replicate in (1, 2):
            target, meta = self.results[("common_column", replicate)]
            permutation = initialization._sha_permutation(initialization.COMMON_COLUMN_SALTS[replicate], None, 0)
            self.assertEqual(meta["permutations"], [permutation] * 9)
            self.assertNotEqual(permutation, list(range(56)))
            self.assertEqual(meta["common_nonce"], 0)
            self.assertEqual(meta["common_permutation_fixed_points"],
                             sum(index == value for index, value in enumerate(permutation)))
            for invariant in ("row_gram_close", "singular_values_close", "rank_equal"):
                self.assertTrue(meta["target_float64_gram_spectral"][invariant])
            source, candidate = self.fixed.double(), target.double()
            self.assertTrue(torch.allclose(source @ source.T, candidate @ candidate.T, atol=1e-12, rtol=1e-12))
            # A column permutation does not keep the time-coordinate Gram.
            self.assertFalse(torch.allclose(source.T @ source, candidate.T @ candidate, atol=1e-12, rtol=1e-12))
            hashes.add(meta["permutation_sha256"])
        self.assertEqual(len(hashes), 2)

    def test_common_columns_accept_fixed_points_without_reselection(self) -> None:
        permutation = list(range(56))
        permutation[0], permutation[1] = permutation[1], permutation[0]
        with mock.patch.object(initialization, "_sha_permutation", return_value=permutation) as generate:
            _, meta = initialization.build_projection(self.fixed, "common_column", 1)
        generate.assert_called_once_with(initialization.COMMON_COLUMN_SALTS[1], None, 0)
        self.assertEqual(meta["common_permutation_fixed_points"], 54)
        self.assertFalse(meta["rows"][0]["historical_row_gates_pass"])

    def test_common_identity_is_rejected_without_retry(self) -> None:
        with mock.patch.object(initialization, "_sha_permutation", return_value=list(range(56))) as generate:
            with self.assertRaisesRegex(RuntimeError, "identity"):
                initialization.build_projection(self.fixed, "common_column", 1)
        self.assertEqual(generate.call_count, 1)

    def test_build_and_apply_do_not_mutate_rng_or_input(self) -> None:
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        fixed_before = self.fixed.clone()
        for arm in self.arms:
            rebuilt, meta = initialization.build_projection(self.fixed, *arm)
            self.assertTrue(torch.equal(rebuilt, self.results[arm][0]))
            self.assertEqual(meta, self.results[arm][1])
            initialization.apply_initialization(MinimalFullModel(), *arm)
        self.assertEqual(random.getstate(), python_state)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_after[0], numpy_state[0])
        self.assertTrue(np.array_equal(numpy_after[1], numpy_state[1]))
        self.assertEqual(numpy_after[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertTrue(torch.equal(self.fixed, fixed_before))

    def test_apply_changes_only_existing_residual_and_retains_gradient(self) -> None:
        for arm in self.arms:
            with self.subTest(arm=arm):
                model = MinimalFullModel()
                before_parameters = dict(model.named_parameters())
                before_buffers = dict(model.named_buffers())
                before_state = {name: value.clone() for name, value in model.state_dict().items()}
                meta = initialization.apply_initialization(model, *arm)
                self.assertEqual(model.representation, "learned_full9_bg")
                self.assertEqual(dict(model.named_parameters()).keys(), before_parameters.keys())
                self.assertEqual(dict(model.named_buffers()).keys(), before_buffers.keys())
                for name, value in model.named_parameters():
                    self.assertIs(value, before_parameters[name])
                for name, value in model.named_buffers():
                    self.assertIs(value, before_buffers[name])
                for name, value in model.state_dict().items():
                    if name != "full_projection_residual":
                        self.assertTrue(torch.equal(value, before_state[name]))
                effective = self.fixed + model.full_projection_residual
                self.assertTrue(torch.allclose(effective, self.results[arm][0],
                                               atol=meta["effective_target_atol"], rtol=meta["effective_target_rtol"]))
                effective.sum().backward()
                self.assertTrue(torch.equal(model.full_projection_residual.grad, torch.ones(9, 56)))
                self.assertTrue(meta["residual_definition_bitwise_equal"])
                self.assertTrue(meta["parameter_inventory_unchanged"])
                self.assertTrue(meta["buffer_inventory_unchanged"])
                self.assertEqual(model.full_projection_residual.numel(), 504)

    def test_validation_and_fail_closed_search_bound(self) -> None:
        for invalid in (torch.zeros(8, 56), torch.zeros(9, 55), -torch.ones(9, 56),
                        torch.full((9, 56), float("nan")), torch.full((9, 56), float("inf")),
                        torch.ones(9, 56, dtype=torch.int64)):
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype):
                with self.assertRaises(ValueError):
                    initialization.build_projection(invalid, "analytical", 0)
        for kind, replicate in (("bogus", 0), ("analytical", 1), ("registered_row", 1),
                                ("independent_row", 0), ("common_column", 3), ("common_column", True)):
            with self.assertRaises(ValueError):
                initialization.build_projection(self.fixed, kind, replicate)
        with self.assertRaisesRegex(RuntimeError, "exact registered"):
            initialization.build_projection(self.fixed.double(), "registered_row", 0)
        with mock.patch.object(initialization, "MAX_SEARCH_NONCES", 0):
            with self.assertRaisesRegex(RuntimeError, "no gate was weakened"):
                initialization.build_projection(self.fixed, "independent_row", 1)
        wrong_model = MinimalFullModel()
        wrong_model.representation = "scratch_full9_bg"
        old = wrong_model.full_projection_residual.clone()
        with self.assertRaisesRegex(ValueError, "dispatch"):
            initialization.apply_initialization(wrong_model, "analytical", 0)
        self.assertTrue(torch.equal(wrong_model.full_projection_residual, old))
        frozen_model = MinimalFullModel()
        frozen_model.full_projection_residual.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "trainable"):
            initialization.apply_initialization(frozen_model, "analytical", 0)

    def test_generic_common_float64_basis(self) -> None:
        fixed = real_fixed_basis(torch.float64)
        target, meta = initialization.build_projection(fixed, "common_column", 1)
        self.assertEqual(target.dtype, torch.float64)
        self.assertTrue(meta["target_float64_gram_spectral"]["row_gram_close"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
