"""TurboQuant code format v2 and the review-2026-10-04 fixes.

Each test names the review item it guards (see .agent/planning/plan.md).
Distances are compared with float64 brute force; tolerances are relative
RMS errors that hold with wide margin for the sizes used here.
"""
import pickle

import numpy as np
import pytest

from vsq import TurboQuantSpace, detected_isa


def gaussian(n, dim, offset=0.5, seed=0):
    r = np.random.default_rng(seed)
    return (r.standard_normal((n, dim)) + offset).astype(np.float32)


def brute(Q, X):
    Q = Q.astype(np.float64)
    X = X.astype(np.float64)
    return (Q**2).sum(1)[:, None] + (X**2).sum(1)[None] - 2.0 * Q @ X.T


def rel_rms(got, ref):
    return float(np.sqrt(np.mean((got - ref) ** 2) / np.mean(ref**2)))


def trained_space(dim, bits=4, X=None, **kw):
    space = TurboQuantSpace(dim, bits, **kw)
    if space.centering() != "none":
        space.train(X if X is not None else gaussian(2000, dim, seed=1))
    return space


# ---------------------------------------------------------------- accuracy ---


@pytest.mark.parametrize("bits,tol", [(4, 0.02), (8, 0.002), (16, 1e-4)])
@pytest.mark.parametrize("centering", ["none", "mean", "ivf"])
def test_asymmetric_distance_tracks_l2(bits, tol, centering):
    dim = 200
    X = gaussian(1500, dim, seed=2)
    Q = gaussian(20, dim, seed=3)
    space = trained_space(dim, bits, X, centering=centering, n_clusters=16)
    codes = space.encode_batch(X)
    assert rel_rms(space.distance_m_to_n(Q, codes), brute(Q, X)) < tol


@pytest.mark.parametrize("bits,tol", [(4, 0.03), (8, 0.003), (16, 1e-4)])
@pytest.mark.parametrize("centering", ["none", "mean", "ivf"])
def test_symmetric_distance_tracks_l2(bits, tol, centering):
    # IVF pairs from different clusters go through the cross-term path.
    dim = 128
    X = gaussian(800, dim, seed=4)
    space = trained_space(dim, bits, X, centering=centering, n_clusters=8)
    codes = space.encode_batch(X)
    got = space.distance_m_to_n_symmetric(codes[:40], codes)
    assert rel_rms(got, brute(X[:40], X)) < tol


def test_batch_apis_agree_with_single_calls():
    dim = 96
    X = gaussian(300, dim, seed=5)
    Q = gaussian(7, dim, seed=6)
    space = trained_space(dim, 4, X, n_clusters=8)
    codes = space.encode_batch(X)
    np.testing.assert_array_equal(codes[3], space.encode(X[3]))
    m2n = space.distance_m_to_n(Q, codes)
    for i in range(len(Q)):
        np.testing.assert_allclose(m2n[i], space.distance_1_to_n(Q[i], codes), rtol=1e-6)
    assert abs(m2n[2, 9] - space.distance(Q[2], codes[9])) < 1e-4 * m2n[2, 9]
    sym = space.distance_m_to_n_symmetric(codes[:5], codes[:9])
    assert abs(sym[1, 7] - space.distance_symmetric(codes[1], codes[7])) < 1e-4 * sym[1, 7]


@pytest.mark.parametrize("bits", [4, 8, 16])
def test_simd_kernels_match_scalar(bits):
    dim = 160
    X = gaussian(400, dim, seed=7)
    Q = gaussian(5, dim, seed=8)
    fast = trained_space(dim, bits, X, n_clusters=8)
    ref = trained_space(dim, bits, X, n_clusters=8, isa="scalar")
    assert ref.kernel_isa() == "scalar"
    assert fast.kernel_isa() == detected_isa()
    codes = fast.encode_batch(X)
    np.testing.assert_array_equal(codes, ref.encode_batch(X))
    np.testing.assert_allclose(fast.distance_m_to_n(Q, codes), ref.distance_m_to_n(Q, codes),
                               rtol=2e-5, atol=1e-3)
    np.testing.assert_allclose(fast.distance_m_to_n_symmetric(codes[:10], codes),
                               ref.distance_m_to_n_symmetric(codes[:10], codes), rtol=2e-5, atol=1e-3)


def test_decode_reconstructs_vector():
    dim = 64
    X = gaussian(500, dim, seed=9)
    space = trained_space(dim, 8, X, centering="mean")
    x_hat = space.decode(space.encode(X[0]))
    assert np.linalg.norm(x_hat - X[0]) < 0.02 * np.linalg.norm(X[0])


# ------------------------------------------------------- 2.1 QJL is optional ---


def test_qjl_is_opt_in_and_costs_one_meta_float():
    dim = 128
    plain = TurboQuantSpace(dim, 4, centering="none")
    qjl = TurboQuantSpace(dim, 4, centering="none", qjl=True)
    assert not plain.qjl() and qjl.qjl()
    assert qjl.code_size_bytes() == plain.code_size_bytes() + 4


def test_qjl_variants_track_l2():
    dim = 128
    X = gaussian(600, dim, seed=10)
    Q = gaussian(10, dim, seed=11)
    for centering in ["none", "ivf"]:
        space = trained_space(dim, 4, X, centering=centering, qjl=True, n_clusters=8)
        codes = space.encode_batch(X)
        assert rel_rms(space.distance_m_to_n(Q, codes), brute(Q, X)) < 0.03


def test_mse_only_beats_qjl_at_equal_storage():
    # Review 2.1 was decided on real embeddings (.agent/planning/plan.md); this
    # is a synthetic guard: 4-bit MSE (+corrected scale) vs 3-bit MSE + QJL.
    dim = 256
    X = gaussian(2000, dim, seed=12)
    Q = gaussian(30, dim, seed=13)
    errs = {}
    for qjl in (False, True):
        space = TurboQuantSpace(dim, 4, centering="none", qjl=qjl)
        errs[qjl] = rel_rms(space.distance_m_to_n(Q, space.encode_batch(X)), brute(Q, X))
    assert errs[False] < errs[True]


# ---------------------------------------- 2.2 / 2.3 layout and storage cost ---


@pytest.mark.parametrize("bits", [4, 8, 16])
@pytest.mark.parametrize("centering,meta", [("none", 8), ("mean", 8), ("ivf", 14)])
def test_code_size_is_nominal_bits_plus_meta(bits, centering, meta):
    space = TurboQuantSpace(1536, bits, centering=centering)
    assert space.padded_dim() == 1536
    assert space.code_size_bytes() == 1536 * bits // 8 + meta


@pytest.mark.parametrize("bits", [0, 1, 2, 3, 5, 7, 9, 12, 32])
def test_rejects_unsupported_bits(bits):
    with pytest.raises(ValueError):
        TurboQuantSpace(64, bits)


def test_rejects_qjl_with_fp16_and_bad_options():
    with pytest.raises(ValueError):
        TurboQuantSpace(64, 16, qjl=True)
    for kw in ({"centering": "kmeans"}, {"estimator": "x"}, {"isa": "sse9"}, {"qjl_cross": "cos"},
               {"rotation_rounds": 0}, {"n_clusters": 0}, {"n_clusters": 70000}):
        with pytest.raises(ValueError):
            TurboQuantSpace(64, 4, **kw)
    with pytest.raises(ValueError):
        TurboQuantSpace(0, 4)


# ------------------------------------------------ 2.4 full symmetric (QJL) ---


@pytest.mark.parametrize("cross", ["linear", "arcsine"])
@pytest.mark.parametrize("centering", ["none", "ivf"])
def test_full_symmetric_qjl(cross, centering):
    dim = 128
    X = gaussian(500, dim, seed=14)
    space = trained_space(dim, 4, X, centering=centering, qjl=True, qjl_cross=cross, n_clusters=8)
    codes = space.encode_batch(X)
    full = space.distance_m_to_n_symmetric_full(codes[:20], codes)
    assert rel_rms(full, brute(X[:20], X)) < 0.03
    assert abs(full[3, 17] - space.distance_symmetric_full(codes[3], codes[17])) < 1e-4 * full[3, 17]
    assert np.all(full >= 0)


def test_full_symmetric_requires_qjl():
    space = TurboQuantSpace(64, 4, centering="none")
    c = space.encode(np.ones(64, np.float32))
    with pytest.raises(ValueError):
        space.distance_symmetric_full(c, c)


# ------------------------------------------- 2.5 rotation without pow2 pad ---


@pytest.mark.parametrize("dim,padded", [(768, 768), (1536, 1536), (3072, 3072), (100, 128),
                                        (960, 960), (8, 64)])
def test_padding_is_multiple_of_64_not_pow2(dim, padded):
    assert TurboQuantSpace(dim, 4, centering="none").padded_dim() == padded


def test_rotation_rounds_are_configurable():
    dim = 192
    X = gaussian(400, dim, seed=15)
    Q = gaussian(8, dim, seed=16)
    codes = {}
    for rounds in (1, 3, 5):
        space = TurboQuantSpace(dim, 4, centering="none", rotation_rounds=rounds)
        assert space.rotation_rounds() == rounds
        codes[rounds] = space.encode_batch(X)
        assert rel_rms(space.distance_m_to_n(Q, codes[rounds]), brute(Q, X)) < 0.03
    assert not np.array_equal(codes[1], codes[3])


def test_more_rounds_gaussianise_sparse_inputs():
    # One-hot inputs: a single round leaves heavy structure in the rotated
    # coordinates; three rounds must not be worse.
    dim = 192
    X = np.zeros((64, dim), np.float32)
    X[np.arange(64), np.arange(64) * 3] = 1.0
    Q = gaussian(8, dim, seed=17)
    errs = []
    for rounds in (1, 3):
        space = TurboQuantSpace(dim, 4, centering="none", rotation_rounds=rounds)
        errs.append(rel_rms(space.distance_m_to_n(Q, space.encode_batch(X)), brute(Q, X)))
    assert errs[1] <= errs[0] * 1.05


# ------------------------------------------------------------ 2.6 centering ---


def test_default_is_ivf_and_requires_training():
    space = TurboQuantSpace(32, 4)
    assert space.centering() == "ivf" and not space.trained()
    with pytest.raises(RuntimeError):
        space.encode(np.ones(32, np.float32))
    space.train(gaussian(1000, 32, seed=18))
    assert space.trained() and 1 <= space.n_clusters() <= 256


def test_ivf_clamps_clusters_to_training_size():
    space = TurboQuantSpace(16, 4, n_clusters=256)
    space.train(gaussian(390, 16, seed=19))  # 390 // 39 = 10 clusters
    assert space.n_clusters() == 10


def test_centering_improves_offset_data():
    dim = 128
    X = gaussian(3000, dim, offset=3.0, seed=20)
    Q = gaussian(30, dim, offset=3.0, seed=21)
    errs = {}
    for c in ("none", "mean", "ivf"):
        space = trained_space(dim, 4, X, centering=c, n_clusters=32)
        errs[c] = rel_rms(space.distance_m_to_n(Q, space.encode_batch(X)), brute(Q, X))
    assert errs["mean"] < errs["none"] and errs["ivf"] < errs["none"]


def test_set_centroids_roundtrip_and_validation():
    dim = 24
    C = gaussian(4, dim, seed=22)
    space = TurboQuantSpace(dim, 4, centering="ivf")
    space.set_centroids(C)
    np.testing.assert_array_equal(space.centroids(), C)
    with pytest.raises(ValueError):
        TurboQuantSpace(dim, 4, centering="mean").set_centroids(C)


@pytest.mark.parametrize("centering", ["none", "mean", "ivf"])
def test_vector_at_centroid_is_exact(centering):
    # Zero residual: f_mul = 0, so the estimate is exactly ||q - c||^2.
    dim = 40
    X = gaussian(400, dim, seed=23)
    space = trained_space(dim, 4, X, centering=centering, n_clusters=4)
    c = np.zeros(dim, np.float32) if centering == "none" else space.centroids()[0].copy()
    q = gaussian(1, dim, seed=24)[0]
    np.testing.assert_allclose(space.distance(q, space.encode(c)), float(np.sum((q - c) ** 2)), rtol=1e-4)


# ---------------------------------------------------- serialization (4) ---


def test_pickle_roundtrip_preserves_codes():
    dim = 72
    X = gaussian(300, dim, seed=25)
    space = trained_space(dim, 8, X, n_clusters=8, rotation_rounds=4, rot_seed=7)
    clone = pickle.loads(pickle.dumps(space))
    assert TurboQuantSpace.format_version() == 2
    np.testing.assert_array_equal(clone.encode_batch(X), space.encode_batch(X))
    np.testing.assert_array_equal(clone.centroids(), space.centroids())


# ------------------------------------------ 1.3 strided input is rejected ---


def test_strided_and_wrong_dtype_inputs_raise():
    dim = 32
    space = TurboQuantSpace(dim, 4, centering="none")
    X = gaussian(10, 2 * dim, seed=26)
    with pytest.raises(ValueError, match="C-contiguous"):
        space.encode_batch(X[:, ::2])
    with pytest.raises(ValueError, match="C-contiguous"):
        space.encode_batch(np.repeat(X[:, :dim], 2, axis=0)[::2])
    with pytest.raises(ValueError, match="float32"):
        space.encode(np.ones(dim, np.float64))
    codes = space.encode_batch(np.ascontiguousarray(X[:, :dim]))
    wide = np.zeros((codes.shape[0], 2 * codes.shape[1]), np.uint8)
    wide[:, ::2] = codes
    with pytest.raises(ValueError, match="C-contiguous"):
        space.distance_1_to_n(np.ones(dim, np.float32), wide[:, ::2])
    with pytest.raises(ValueError):
        space.distance(np.ones(dim, np.float32), codes[0][:-1])


def test_distances_are_never_negative():
    dim = 64
    X = gaussian(200, dim, seed=29)
    for bits in (4, 8, 16):
        space = trained_space(dim, bits, X, n_clusters=4)
        codes = space.encode_batch(X)
        assert np.all(space.distance_m_to_n(X[:20], codes) >= 0)
        assert np.all(space.distance_m_to_n_symmetric(codes[:20], codes) >= 0)
