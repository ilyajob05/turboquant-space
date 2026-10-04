"""FastScan flat indexes (review item 3.2) and RaBitQ two-stage search (3.3).

Acceptance: after re-ranking, the FastScan top-k must agree with the
per-pair estimate of the same space (recall@10 >= 0.99), i.e. LUT
quantization never changes the answer once candidates are re-scored.
"""
import gc

import numpy as np
import pytest

from vsq import RaBitQFastScan, RaBitQSpace, TurboQuantFastScan, TurboQuantSpace


def gaussian(n, dim, seed=0):
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)


def topk(d, k):
    return set(np.argsort(d, kind="stable")[:k].tolist())


@pytest.mark.parametrize("centering", ["none", "ivf"])
def test_tq4_fastscan_matches_per_pair_topk(centering):
    dim, n, k = 192, 3000, 10
    X = gaussian(n, dim, seed=1)
    Q = gaussian(20, dim, seed=2)
    space = TurboQuantSpace(dim, 4, centering=centering, n_clusters=16)
    if centering != "none":
        space.train(X)
    codes = space.encode_batch(X)
    index = TurboQuantFastScan(space, codes)
    assert len(index) == n
    hits = 0
    for q in Q:
        exact = space.distance_1_to_n(q, codes)
        approx = index.scan(q)
        assert np.max(np.abs(approx - exact) / exact) < 0.05
        ids, dists = index.search(q, k, rerank=4)
        assert list(dists) == sorted(dists)
        np.testing.assert_allclose(dists, exact[ids], rtol=1e-4)
        hits += len(topk(exact, k) & set(ids.tolist()))
    assert hits / (len(Q) * k) >= 0.99


def test_tq4_fastscan_rejects_other_layouts():
    X = gaussian(100, 64, seed=3)
    for kw in ({"bits": 8}, {"bits": 4, "qjl": True}):
        space = TurboQuantSpace(64, centering="none", **kw)
        with pytest.raises(ValueError):
            TurboQuantFastScan(space, space.encode_batch(X))


def test_fastscan_keeps_space_alive():
    X = gaussian(64, 64, seed=4)
    space = TurboQuantSpace(64, 4, centering="none")
    index = TurboQuantFastScan(space, space.encode_batch(X))
    del space
    gc.collect()
    ids, _ = index.search(X[0], 3)
    assert ids[0] == 0


@pytest.mark.parametrize("bits", [1, 4, 8])
def test_rabitq_two_stage_matches_full_scan(bits):
    dim, n, k = 256, 4000, 10
    X = gaussian(n, dim, seed=5)
    Q = gaussian(20, dim, seed=6)
    space = RaBitQSpace(dim, bits=bits)
    index = RaBitQFastScan(space, X)
    codes = index.codes()
    hits, refined = 0, 0
    for q in Q:
        exact = space.distance_1_to_n(q, codes)
        ids, dists, r = index.search(q, k)
        refined += r
        np.testing.assert_allclose(dists, exact[ids], rtol=1e-5)
        hits += len(topk(exact, k) & set(ids.tolist()))
        est, lower = index.scan(q)
        assert np.all(lower <= est + 1e-6)
    assert hits / (len(Q) * k) >= 0.99
    assert refined / (len(Q) * n) < 0.5  # stage 1 prunes most codes
