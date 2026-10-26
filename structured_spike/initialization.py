"""RNG-free, parameter-matched FullMatrix initialization controls.

Only ``apply_initialization`` mutates a model, by copying into its existing
504-element residual.  It never changes the learned_full9_bg dispatch tag,
registers a parameter/buffer, or imports the frozen scientific repository.

Replicate 0 identifies analytical / the historical registered_row treatment;
replicates 1 and 2 identify new independent_row / common_column realizations.
Row-family selection uses the canonical CPU float32 matrix and the historical
gates. Selected candidates are also audited on the supplied tensor's device;
device disagreement raises rather than silently choosing another realization.
Common-column selection uses no row/correlation/statistics acceptance gate.

For common columns, W P (W P)^T = W W^T in exact arithmetic. This says nothing
about preserving W^T W in the original time coordinates, data covariance, or
optimization geometry. The float64 target audit and actual effective-matrix
audit are deliberately separate.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch


PROTOCOL = "spkdeblurnet_pr_initialization_v1"
LEARNED_FULL_REPRESENTATION = "learned_full9_bg"
FULL_PROJECTION_PARAMETERS = 9 * 56
REGISTERED_SALT = "spkdeblurnet-scratch-full-row-permutation-v1"
REGISTERED_NONCES = (70, 66, 8, 22, 231, 291, 24, 398, 342)
REGISTERED_PERMUTATION_SHA256 = (
    "851102b9eaaaca524473e756aac163e473c935e29cc1abf8b7025e229a14146f"
)
REGISTERED_FIXED_SHA256 = (
    "da6f0657cc755c645eefd65124a5a99dac859ffe2260c82c6e0955b967cbf854"
)
REGISTERED_TARGET_SHA256 = (
    "a1b3995c1f7f6bd125b7406347746c9ddfd2aec8d96aabea0f52a75c31a3bdd0"
)
REGISTERED_RESIDUAL_SHA256 = (
    "a155b657d5f62a9a5506229ff0afae42c5f449e93f6966fa34584480eecc2e80"
)
INDEPENDENT_ROW_SALTS = {
    1: "spkdeblurnet-pr-upgrade-independent-row-v1-replicate-1",
    2: "spkdeblurnet-pr-upgrade-independent-row-v1-replicate-2",
}
COMMON_COLUMN_SALTS = {
    1: "spkdeblurnet-pr-upgrade-common-column-v1-replicate-1",
    2: "spkdeblurnet-pr-upgrade-common-column-v1-replicate-2",
}
MAX_SEARCH_NONCES = 100_000
MAX_ABS_ROW_WEIGHT_CORRELATION = 0.10
MAX_ABS_ROW_COORDINATE_CORRELATION = 0.10
MAX_ROW_SUPPORT_INTERSECTION = 3
FLOAT64_ATOL = 1e-12
FLOAT64_RTOL = 1e-12


def tensor_sha256(value: torch.Tensor) -> str:
    """Match the frozen helper: seal dtype, shape and contiguous tensor bytes."""
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def permutation_sha256(rows: list[list[int]]) -> str:
    return hashlib.sha256(
        json.dumps(rows, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _validate_fixed(fixed: torch.Tensor) -> None:
    if not isinstance(fixed, torch.Tensor):
        raise TypeError("fixed must be a torch.Tensor")
    if tuple(fixed.shape) != (9, 56):
        raise ValueError(f"fixed must have shape (9, 56), got {tuple(fixed.shape)}")
    if fixed.layout != torch.strided or fixed.dtype not in (torch.float32, torch.float64):
        raise ValueError("fixed must be a dense float32 or float64 tensor")
    if not bool(torch.isfinite(fixed).all().item()) or bool((fixed < 0).any().item()):
        raise ValueError("fixed must be finite and nonnegative")


def _validate_kind(kind: str, replicate: int) -> None:
    if type(replicate) is not int:
        raise ValueError("replicate must be an integer, not a bool")
    if kind in ("analytical", "registered_row"):
        if replicate != 0:
            raise ValueError(f"{kind} uses replicate=0 only")
    elif kind in ("independent_row", "common_column"):
        if replicate not in (1, 2):
            raise ValueError(f"{kind} supports replicate=1 or 2 only")
    else:
        raise ValueError(f"unknown initialization kind: {kind!r}")


def _sha_permutation(salt: str, row: int | None, nonce: int) -> list[int]:
    # The row-present payload is byte-for-byte the historical family.
    prefix = f"{salt}\0{nonce}" if row is None else f"{salt}\0{row}\0{nonce}"
    keyed = [
        (hashlib.sha256(f"{prefix}\0{column}".encode("utf-8")).digest(), column)
        for column in range(56)
    ]
    return [column for _, column in sorted(keyed)]


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left.detach().double().flatten()
    right = right.detach().double().flatten()
    left = left - left.mean()
    right = right - right.mean()
    denominator = (left.square().sum() * right.square().sum()).sqrt()
    if denominator.item() == 0.0:
        return None
    return float((left * right).sum().item() / denominator.item())


def _strict_row_stats(row: torch.Tensor) -> dict[str, torch.Tensor]:
    # Do not replace these reductions with mathematically equivalent formulas:
    # historical gates compare their float32 values bitwise.
    minimum, maximum = row.amin(), row.amax()
    return {
        "sum": row.sum(),
        "dc": row.sum(),
        "l1": row.abs().sum(),
        "squared_l2": row.square().sum(),
        "l2": torch.linalg.vector_norm(row),
        "min": minimum,
        "max": maximum,
        "range": maximum - minimum,
        "nonzero": (row != 0).sum(),
        "zero": (row == 0).sum(),
    }


def _row_gate_record(fixed: torch.Tensor, target: torch.Tensor,
                     permutation: torch.Tensor) -> dict[str, Any]:
    source_stats, target_stats = _strict_row_stats(fixed), _strict_row_stats(target)
    statistics_equal = {
        name: torch.equal(source_stats[name], target_stats[name]) for name in source_stats
    }
    coordinates = torch.arange(56, dtype=torch.float64, device=fixed.device)
    fixed_points = int((permutation == coordinates).sum().item())
    correlation = _correlation(fixed, target)
    coordinate_correlation = _correlation(coordinates, permutation)
    intersection = int(((fixed != 0) & (target != 0)).sum().item())
    multiset_equal = torch.equal(torch.sort(fixed).values, torch.sort(target).values)
    changed = not torch.equal(fixed, target)
    roundtrip_equal = torch.equal(fixed + (target - fixed), target)
    return {
        "statistics_bitwise_equal": statistics_equal,
        "row_multiset_bitwise_equal": multiset_equal,
        "row_changed": changed,
        "permutation_fixed_points": fixed_points,
        "permutation_derangement": fixed_points == 0,
        "support_intersection": intersection,
        "aligned_weight_correlation": correlation,
        "coordinate_correlation": coordinate_correlation,
        "effective_roundtrip_bitwise_equal": roundtrip_equal,
        "historical_row_gates_pass": bool(
            multiset_equal and changed and fixed_points == 0
            and intersection <= MAX_ROW_SUPPORT_INTERSECTION
            and correlation is not None
            and abs(correlation) < MAX_ABS_ROW_WEIGHT_CORRELATION
            and coordinate_correlation is not None
            and abs(coordinate_correlation) < MAX_ABS_ROW_COORDINATE_CORRELATION
            and roundtrip_equal and all(statistics_equal.values())
        ),
    }


def _select_independent_rows(fixed_cpu: torch.Tensor, salt: str
                             ) -> tuple[list[list[int]], list[int]]:
    rows, nonces = [], []
    for row_index, source in enumerate(fixed_cpu):
        for nonce in range(MAX_SEARCH_NONCES):
            permutation = _sha_permutation(salt, row_index, nonce)
            # Cheap early rejection; it is exactly one historical gate.
            if any(index == value for index, value in enumerate(permutation)):
                continue
            indices = torch.tensor(permutation, dtype=torch.long)
            target = source[indices]
            if _row_gate_record(source, target, indices)["historical_row_gates_pass"]:
                rows.append(permutation)
                nonces.append(nonce)
                break
        else:
            raise RuntimeError(
                f"independent_row search exhausted {MAX_SEARCH_NONCES} nonces "
                f"for row {row_index}, salt={salt!r}; no gate was weakened"
            )
    return rows, nonces


def _locality(row: torch.Tensor) -> dict[str, Any]:
    row = row.detach().cpu().double()
    support = torch.nonzero(row != 0).flatten().tolist()
    mass = float(row.sum().item())
    coordinates = torch.arange(56, dtype=torch.float64) / 55.0
    centroid = float((row * coordinates).sum().item() / mass) if mass else None
    variance = (
        float((row * (coordinates - centroid).square()).sum().item() / mass)
        if mass else None
    )
    return {
        "nonzero": len(support),
        "support_indices": support,
        "support_time_span": (support[-1] - support[0]) / 55.0 if support else 0.0,
        "support_contiguous_runs": sum(
            i == 0 or value != support[i - 1] + 1 for i, value in enumerate(support)
        ),
        "weighted_time_centroid": centroid,
        "weighted_time_variance": variance,
        "adjacent_total_variation": float(torch.diff(row).abs().sum().item()),
    }


def _matrix_diagnostics(fixed: torch.Tensor, target: torch.Tensor,
                        *, atol: float, rtol: float) -> dict[str, Any]:
    # Conversion before multiplication/SVD isolates target mathematical
    # invariants from float32 reduction and residual-reconstruction error.
    source = fixed.detach().cpu().double()
    candidate = target.detach().cpu().double()
    source_gram, target_gram = source @ source.T, candidate @ candidate.T
    source_singular = torch.linalg.svdvals(source)
    target_singular = torch.linalg.svdvals(candidate)
    source_rank = int(torch.linalg.matrix_rank(source).item())
    target_rank = int(torch.linalg.matrix_rank(candidate).item())
    return {
        "arithmetic_dtype": "torch.float64",
        "atol": atol,
        "rtol": rtol,
        "fixed_row_gram": source_gram.tolist(),
        "target_row_gram": target_gram.tolist(),
        "row_gram_max_abs_difference": float((source_gram - target_gram).abs().max().item()),
        "row_gram_close": bool(torch.allclose(source_gram, target_gram, atol=atol, rtol=rtol)),
        "fixed_singular_values": source_singular.tolist(),
        "target_singular_values": target_singular.tolist(),
        "singular_values_max_abs_difference": float((source_singular - target_singular).abs().max().item()),
        "singular_values_close": bool(torch.allclose(source_singular, target_singular, atol=atol, rtol=rtol)),
        "fixed_rank": source_rank,
        "target_rank": target_rank,
        "rank_equal": source_rank == target_rank,
    }


@torch.no_grad()
def build_projection(fixed: torch.Tensor, kind: str, replicate: int
                     ) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build and audit one initialization, preserving fixed dtype/device.

    Row controls deliberately require the registered analytic float32 matrix;
    analytical/common_column also accept other finite nonnegative 9x56 matrices.
    ``fixed`` and every RNG state remain unchanged. Returned metadata is JSON-safe.
    """
    _validate_fixed(fixed)
    _validate_kind(kind, replicate)
    fixed = fixed.detach()
    fixed_cpu = fixed.cpu()
    fixed_hash = tensor_sha256(fixed)
    row_family = kind in ("registered_row", "independent_row")
    if row_family and fixed_hash != REGISTERED_FIXED_SHA256:
        raise RuntimeError("row controls require the exact registered float32 analytic matrix")
    salt: str | None = None
    nonces: list[int] = []
    common_nonce: int | None = None
    canonical = list(range(56))
    if kind == "analytical":
        rows = [canonical.copy() for _ in range(9)]
        selection = "identity_no_selection"
    elif kind == "registered_row":
        salt, nonces = REGISTERED_SALT, list(REGISTERED_NONCES)
        rows = [_sha_permutation(salt, row, nonce) for row, nonce in enumerate(nonces)]
        selection = "historical_registered_exact_nonces"
        if permutation_sha256(rows) != REGISTERED_PERMUTATION_SHA256:
            raise RuntimeError("historical registered permutation identity drift")
    elif kind == "independent_row":
        salt = INDEPENDENT_ROW_SALTS[replicate]
        rows, nonces = _select_independent_rows(fixed_cpu, salt)
        selection = "first_CPU_candidate_passing_all_historical_row_gates"
    else:
        salt, common_nonce = COMMON_COLUMN_SALTS[replicate], 0
        permutation = _sha_permutation(salt, None, common_nonce)
        # No support, correlation, fixed-point-count, row-statistic, or
        # residual-roundtrip selection. Never retry based on these diagnostics.
        if permutation == canonical:
            raise RuntimeError("first common-column permutation is identity; protocol must be reviewed")
        rows = [permutation.copy() for _ in range(9)]
        selection = "first_SHA_sort_nonce_0_no_data_or_decorrelation_selection"
    if any(sorted(row) != canonical for row in rows):
        raise RuntimeError("invalid permutation table")
    indices = torch.tensor(rows, dtype=torch.long, device=fixed.device)
    target = torch.gather(fixed, 1, indices)
    residual = target - fixed
    effective = fixed + residual
    row_records = []
    for row_index in range(9):
        record = _row_gate_record(fixed[row_index], target[row_index], indices[row_index])
        cpu_record = _row_gate_record(
            fixed_cpu[row_index], target[row_index].cpu(), indices[row_index].cpu()
        )
        record.update({
            "row": row_index,
            "nonce": nonces[row_index] if nonces else common_nonce,
            "fixed_locality": _locality(fixed[row_index]),
            "target_locality": _locality(target[row_index]),
            "CPU_historical_row_gates_pass": cpu_record["historical_row_gates_pass"],
            "CPU_statistics_bitwise_equal": cpu_record["statistics_bitwise_equal"],
        })
        row_records.append(record)
    if not all(row["row_multiset_bitwise_equal"] for row in row_records):
        raise RuntimeError("row multiset drift")
    if row_family and not all(
        row["historical_row_gates_pass"] and row["CPU_historical_row_gates_pass"]
        for row in row_records
    ):
        raise RuntimeError(
            "historical row gates failed on CPU or caller device; no device-dependent reselection allowed"
        )
    tolerance = 8.0 * torch.finfo(fixed.dtype).eps
    scale = max(1.0, float(fixed.abs().max().item()))
    effective_close = bool(torch.allclose(effective, target, atol=tolerance * scale, rtol=tolerance))
    if not effective_close:
        raise RuntimeError("effective initialization does not match intended target within dtype tolerance")
    mathematical = _matrix_diagnostics(fixed, target, atol=FLOAT64_ATOL, rtol=FLOAT64_RTOL)
    actual = _matrix_diagnostics(fixed, effective, atol=tolerance * scale * scale * 56, rtol=tolerance * 56)
    if kind == "common_column" and not all(
        mathematical[name] for name in ("row_gram_close", "singular_values_close", "rank_equal")
    ):
        raise RuntimeError("common-column float64 Gram/spectral invariant failed")
    hashes = {
        "fixed_projection_sha256": fixed_hash,
        "target_projection_sha256": tensor_sha256(target),
        "initial_residual_sha256": tensor_sha256(residual),
        "effective_projection_sha256": tensor_sha256(effective),
    }
    if kind == "registered_row" and (
        hashes["target_projection_sha256"] != REGISTERED_TARGET_SHA256
        or hashes["initial_residual_sha256"] != REGISTERED_RESIDUAL_SHA256
        or hashes["effective_projection_sha256"] != REGISTERED_TARGET_SHA256
    ):
        raise RuntimeError("historical registered matrix/residual identity drift")
    metadata = {
        "status": "PASS",
        "protocol": PROTOCOL,
        "kind": kind,
        "replicate": replicate,
        "projection_parameters": FULL_PROJECTION_PARAMETERS,
        "dtype": str(fixed.dtype),
        "device": str(fixed.device),
        "forward_dispatch_representation": LEARNED_FULL_REPRESENTATION,
        "selection_rule": selection,
        "selection_uses_data_or_metrics": False,
        "selection_device": "cpu" if row_family else "not_applicable",
        "salt": salt,
        "salt_sha256": hashlib.sha256(salt.encode("utf-8")).hexdigest() if salt else None,
        "row_nonces": nonces,
        "common_nonce": common_nonce,
        "search_nonce_exclusive_bound": MAX_SEARCH_NONCES if kind == "independent_row" else None,
        "permutations": rows,
        "permutation_sha256": permutation_sha256(rows),
        "permutation_convention": "target[k,t] = fixed[k, permutations[k][t]]",
        "common_permutation_shared_all_rows": all(row == rows[0] for row in rows),
        "common_permutation_fixed_points": row_records[0]["permutation_fixed_points"] if kind == "common_column" else None,
        "all_row_multisets_bitwise_equal": True,
        "all_row_statistics_bitwise_equal": all(all(row["statistics_bitwise_equal"].values()) for row in row_records),
        "effective_equals_target_bitwise": torch.equal(effective, target),
        "effective_equals_target_within_tolerance": effective_close,
        "effective_target_atol": tolerance * scale,
        "effective_target_rtol": tolerance,
        "effective_target_max_abs_difference": float((effective - target).abs().max().item()),
        "effective_minimum_weight": float(effective.min().item()),
        "target_float64_gram_spectral": mathematical,
        "actual_effective_gram_spectral": actual,
        "invariant_claim_boundary": (
            "Common-column target preserves WW^T, rank and singular values; "
            "not W^T W in original coordinates, data covariance or optimization geometry. "
            "Actual effective values can differ through finite-precision residual reconstruction."
        ),
        "rows": row_records,
        **hashes,
    }
    return target, metadata


def _model_inventory(model: torch.nn.Module) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    parameters = [(name, id(value), tuple(value.shape), value.requires_grad)
                  for name, value in model.named_parameters()]
    buffers = [(name, id(value), tuple(value.shape)) for name, value in model.named_buffers()]
    return parameters, buffers


@torch.no_grad()
def apply_initialization(model: torch.nn.Module, kind: str, replicate: int) -> dict[str, Any]:
    """Copy target-minus-fixed into the existing trainable residual, only.

    Call after construction and device placement, before optimizer/DDP setup.
    This does not establish inherited encode/forward equivalence: deployment
    must separately exercise the real encoder and all 504 gradient elements.
    """
    _validate_kind(kind, replicate)
    if getattr(model, "representation", None) != LEARNED_FULL_REPRESENTATION:
        raise ValueError("model must retain internal learned_full9_bg dispatch")
    residual = getattr(model, "full_projection_residual", None)
    if not isinstance(residual, torch.nn.Parameter) or tuple(residual.shape) != (9, 56):
        raise ValueError("model requires an existing 9x56 full_projection_residual Parameter")
    if residual.numel() != FULL_PROJECTION_PARAMETERS or not residual.requires_grad:
        raise ValueError("all 504 residual parameters must be trainable")
    before = _model_inventory(model)
    timestamps, delta = model._time_basis(residual.new_zeros(()))
    fixed = timestamps.unsqueeze(0) * model.value_layer.trilinear(delta)
    if fixed.device != residual.device or fixed.dtype != residual.dtype:
        raise ValueError("fixed basis must match residual dtype and device")
    target, metadata = build_projection(fixed, kind, replicate)
    previous = residual.detach().clone()
    try:
        residual.copy_(target - fixed)
        effective = fixed + residual
        if not torch.allclose(effective, target, atol=metadata["effective_target_atol"],
                              rtol=metadata["effective_target_rtol"]):
            raise RuntimeError("copied residual does not reconstruct target")
        if before != _model_inventory(model):
            raise RuntimeError("initialization changed parameter or buffer inventory")
        if model.representation != LEARNED_FULL_REPRESENTATION:
            raise RuntimeError("initialization changed internal model dispatch")
        if tensor_sha256(effective) != metadata["effective_projection_sha256"]:
            raise RuntimeError("effective projection differs from audited residual reconstruction")
    except Exception:
        residual.copy_(previous)
        raise
    metadata.update({
        "applied_to_model": True,
        "projection_requires_grad": bool(residual.requires_grad),
        "parameter_inventory_unchanged": True,
        "buffer_inventory_unchanged": True,
        "residual_definition_bitwise_equal": torch.equal(residual, target - fixed),
        "total_model_parameters": sum(value.numel() for value in model.parameters()),
    })
    return metadata
