import random

import numpy as np
import pytest
import torch

from structured_spike.initialization import tensor_sha256, _sha_permutation
from structured_spike.portable_initialization import build_published_projection


def fixed():
    t = torch.linspace(0, 1, 56)
    c = torch.linspace(0, 1, 9)
    return t[None] * torch.relu(1 - 8 * (t[None] - c[:, None]).abs())


ARMS = [("analytical", 0), ("registered_row", 0), ("independent_row", 1),
        ("independent_row", 2), ("common_column", 1), ("common_column", 2)]


@pytest.mark.parametrize("kind,replicate", ARMS)
def test_published_matrices(kind, replicate):
    w = fixed()
    before = w.clone()
    state = torch.get_rng_state().clone()
    py_state = random.getstate()
    np_state = np.random.get_state()
    target, meta = build_published_projection(w, kind, replicate)
    assert torch.equal(w, before)
    assert torch.equal(torch.get_rng_state(), state)
    assert random.getstate() == py_state
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert tensor_sha256(target) == meta["target_projection_sha256"]
    assert tensor_sha256(w + (target - w)) == meta["effective_projection_sha256"]
    assert torch.equal(w.sort(1).values, target.sort(1).values)
    assert not meta["structural_selection_repeated_on_this_host"]
    if kind in ("registered_row", "independent_row"):
        for row, nonce in enumerate(meta["row_nonces"]):
            assert meta["permutations"][row] == _sha_permutation(meta["salt"], row, nonce)
    if kind == "common_column":
        assert meta["permutations"] == [_sha_permutation(meta["salt"], None, 0)] * 9
        assert torch.allclose(w.double() @ w.double().T,
                              target.double() @ target.double().T, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("value,kind,replicate,error", [
    (torch.ones(9, 56), "analytical", 0, RuntimeError),
    (fixed().double(), "analytical", 0, ValueError),
    (fixed(), "independent_row", 0, ValueError),
    (fixed(), "common_column", True, ValueError),
])
def test_invalid_requests_fail(value, kind, replicate, error):
    with pytest.raises(error):
        build_published_projection(value, kind, replicate)
