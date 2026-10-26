"""Load the experiment's selected permutations without rerunning host-specific gates.

The original strict selector is preserved in initialization.py. Some CPU backends
use different float32 reduction orders. This loader verifies byte identities of
the published matrices instead of selecting a different permutation on that host.
"""
from importlib.resources import files
import json

import torch

from .initialization import _validate_fixed, _validate_kind, permutation_sha256, tensor_sha256


@torch.no_grad()
def build_published_projection(fixed, kind="analytical", replicate=0):
    _validate_fixed(fixed)
    _validate_kind(kind, replicate)
    if fixed.dtype != torch.float32:
        raise ValueError("published experiment matrices require float32")
    records = json.loads(files("structured_spike").joinpath("published_initializations.json").read_text())
    record = next(r for r in records if r["kind"] == kind and r["replicate"] == replicate)
    if tensor_sha256(fixed) != record["fixed_projection_sha256"]:
        raise RuntimeError("analytic matrix differs from published float32 bytes; use a matching runtime")
    rows = record["permutations"]
    if len(rows) != 9 or any(sorted(row) != list(range(56)) for row in rows):
        raise RuntimeError("invalid published permutation")
    if permutation_sha256(rows) != record["permutation_sha256"]:
        raise RuntimeError("published permutation hash mismatch")
    indices = torch.tensor(rows, device=fixed.device, dtype=torch.long)
    target = torch.gather(fixed.detach(), 1, indices)
    residual = target - fixed.detach()
    effective = fixed.detach() + residual
    for label, value in (("target_projection_sha256", target),
                         ("initial_residual_sha256", residual),
                         ("effective_projection_sha256", effective)):
        if tensor_sha256(value) != record[label]:
            raise RuntimeError("published matrix identity mismatch: " + label)
    return target, dict(record, loader="published_indices_exact_byte_checks",
                        structural_selection_repeated_on_this_host=False)
