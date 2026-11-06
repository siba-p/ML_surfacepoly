import numpy as np
import pytest

from mlsurfpoly.data import prepare_model_data
from mlsurfpoly.data.symmetry import RotFlipInvariants, findCanonicalForm
from mlsurfpoly.paths import SPLITS


def test_surface_canonical_form_is_invariant_under_all_symmetries():
    rng = np.random.default_rng(0)
    surface = (rng.random((20, 20)) > 0.5).astype(int)
    canon = findCanonicalForm(RotFlipInvariants(surface))[0]
    for op in (np.rot90, np.fliplr, np.flipud, lambda a: np.rot90(a, 2).T):
        other = findCanonicalForm(RotFlipInvariants(op(surface)))[0]
        assert np.array_equal(canon, other)


def test_polymer_canonical_form_is_invariant_under_reversal():
    seq = np.array([0, 1, 1, 0, 0, 1, 0, 0])
    a = findCanonicalForm(RotFlipInvariants(seq))[0]
    b = findCanonicalForm(RotFlipInvariants(seq[::-1]))[0]
    assert np.array_equal(a, b)


@pytest.fixture(scope="module")
def splits():
    return prepare_model_data("canonical")


def test_split_shapes_and_sizes(splits):
    xtr, ytr, xva, yva, xte, yte, delf = splits
    assert xtr.shape == (5797, 440) and ytr.shape == (5797, 100)
    assert xva.shape == (1656, 440) and xte.shape == (828, 440)
    assert delf.shape == (91 * 91,)


def test_no_pair_is_shared_between_splits(splits):
    xtr, _, xva, _, xte, _, _ = splits
    keys = [{r.astype(np.int8).tobytes() for r in x} for x in (xtr, xva, xte)]
    assert not (keys[0] & keys[1]) and not (keys[0] & keys[2]) and not (keys[1] & keys[2])


def test_pmf_unbound_state_is_zero(splits):
    ytr = splits[1]
    assert np.abs(ytr[:, -10:].mean(axis=1)).max() < 1e-6


@pytest.mark.skipif(not (SPLITS / "fdX_test.npy").exists(), reason="shipped splits not present")
def test_pipeline_reproduces_shipped_splits(splits):
    names = ["fdX_train", "fdY_train", "fdX_valid", "fdY_valid", "fdX_test", "fdY_test", "delF"]
    for name, arr in zip(names, splits):
        assert np.allclose(arr, np.load(SPLITS / f"{name}.npy"), atol=1e-6), name


def test_every_test_polymer_and_surface_is_seen_in_training(splits):
    """Documents what the random pair-level split measures (new combinations of known polymers and surfaces)."""
    xtr, _, _, _, xte, _, _ = splits
    for sl in (slice(0, 400), slice(400, 440)):
        seen = {r[sl].astype(np.int8).tobytes() for r in xtr}
        assert all(r[sl].astype(np.int8).tobytes() in seen for r in xte)
