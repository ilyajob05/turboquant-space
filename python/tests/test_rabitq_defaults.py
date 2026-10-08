"""RaBitQ with the default settings and the review-2026-10-04 fixes.

test_rabitq.py pins the legacy rotation + float query and checks bit-exact
agreement with a NumPy oracle; this file covers what changed by default:
BlockKac rotation, quantized 1-bit query (popcount), error bounds,
zero residuals, clamping and batch threading.
"""
import numpy as np
import pytest

from vsq import RaBitQSpace, detected_isa


def gaussian(n, dim, seed=0):
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)


def brute(Q, X):
    Q = Q.astype(np.float64)
    X = X.astype(np.float64)
    return (Q**2).sum(1)[:, None] + (X**2).sum(1)[None] - 2.0 * Q @ X.T


def rel_rms(got, ref):
    return float(np.sqrt(np.mean((got - ref) ** 2) / np.mean(ref**2)))


@pytest.mark.parametrize("dim,padded", [(1536, 1536), (768, 768), (100, 128), (8, 64)])
def test_default_rotation_pads_to_multiple_of_64(dim, padded):
    space = RaBitQSpace(dim)
    assert space.rotation() == "kac"
    assert space.padded_dim() == padded


def test_legacy_rotation_keeps_pow2_padding():
    assert RaBitQSpace(1536, rotation="legacy").padded_dim() == 2048


def test_default_is_four_bit_fixed_scale_float_query():
    space = RaBitQSpace(256)
    assert (space.bits(), space.encode_mode(), space.query_bits()) == (4, "fixed_scale", 0)


def test_one_bit_default_uses_quantized_query():
    space = RaBitQSpace(256, bits=1)
    assert space.query_bits() == 4
    assert space.distance_kernel() == "1-" + detected_isa() + "-q4"
    assert RaBitQSpace(256, bits=4).query_bits() == 0
    assert RaBitQSpace(256, bits=1, query_bits=0).distance_kernel() == "1-" + detected_isa()


def test_query_bits_validation():
    with pytest.raises(ValueError):
        RaBitQSpace(64, bits=4, query_bits=4)
    with pytest.raises(ValueError):
        RaBitQSpace(64, bits=1, query_bits=9)
    with pytest.raises(ValueError):
        RaBitQSpace(16, bits=1, rotation="legacy", query_bits=4)  # D = 16 is not a multiple of 64
    assert RaBitQSpace(16, bits=1, rotation="legacy").query_bits() == 0  # the default falls back


@pytest.mark.parametrize("bits,tol", [(1, 0.08), (4, 0.012), (8, 0.002)])
def test_distance_tracks_l2(bits, tol):
    dim = 256
    X = gaussian(1000, dim, seed=1)
    Q = gaussian(10, dim, seed=2)
    space = RaBitQSpace(dim, bits=bits)
    assert rel_rms(space.distance_m_to_n(Q, space.encode_batch(X)), brute(Q, X)) < tol


def test_quantized_query_is_close_to_float_query():
    dim = 512
    X = gaussian(500, dim, seed=3)
    Q = gaussian(5, dim, seed=4)
    q4 = RaBitQSpace(dim, bits=1, query_bits=4)
    qf = RaBitQSpace(dim, bits=1, query_bits=0)
    codes = q4.encode_batch(X)
    np.testing.assert_array_equal(codes, qf.encode_batch(X))  # same code, different query
    assert rel_rms(q4.distance_m_to_n(Q, codes), qf.distance_m_to_n(Q, codes)) < 0.03


@pytest.mark.parametrize("bits", [1, 4, 8])
def test_simd_matches_scalar(bits):
    dim = 192
    X = gaussian(50, dim, seed=5)
    q = gaussian(1, dim, seed=6)[0]
    space = RaBitQSpace(dim, bits=bits)
    codes = space.encode_batch(X)
    for i in range(len(X)):
        assert abs(space.distance(q, codes[i]) - space.distance_scalar(q, codes[i])) < 1e-3


@pytest.mark.parametrize("bits", [1, 4, 8])
def test_error_bound_covers_true_distance(bits):
    # eps0 = 1.9 is a ~94% two-sided interval in the paper; require >= 85%.
    dim = 256
    X = gaussian(400, dim, seed=7)
    Q = gaussian(5, dim, seed=8)
    space = RaBitQSpace(dim, bits=bits)
    codes = space.encode_batch(X)
    ref = brute(Q, X)
    inside = 0
    for i in range(len(Q)):
        for j in range(len(X)):
            d, lo, hi = space.distance_bound(Q[i], codes[j])
            assert lo <= d <= hi
            inside += lo <= ref[i, j] <= hi
    assert inside / ref.size > 0.85


def test_wider_eps_gives_wider_interval():
    space = RaBitQSpace(128)
    x, q = gaussian(2, 128, seed=9)
    c = space.encode(x)
    _, lo1, hi1 = space.distance_bound(q, c, eps0=1.0)
    _, lo3, hi3 = space.distance_bound(q, c, eps0=3.0)
    assert hi3 - lo3 > hi1 - lo1


@pytest.mark.parametrize("bits", [1, 4, 8])
def test_zero_residual_is_exact(bits):
    dim = 96
    c = gaussian(1, dim, seed=10)[0]
    q = gaussian(1, dim, seed=11)[0]
    space = RaBitQSpace(dim, centroid=c, bits=bits)
    np.testing.assert_allclose(space.distance(q, space.encode(c.copy())), float(np.sum((q - c) ** 2)),
                               rtol=1e-5)


def test_non_finite_input_raises():
    space = RaBitQSpace(16)
    x = np.ones(16, np.float32)
    x[3] = np.nan
    with pytest.raises(ValueError):
        space.encode(x)
    X = np.ones((100, 16), np.float32)
    X[57, 0] = np.inf
    with pytest.raises(ValueError):  # the first error inside the OpenMP loop is rethrown
        space.encode_batch(X)


@pytest.mark.parametrize("bits", [1, 4, 8])
def test_distances_are_never_negative(bits):
    dim = 64
    X = gaussian(300, dim, seed=12)
    space = RaBitQSpace(dim, bits=bits)
    assert np.all(space.distance_m_to_n(X[:30], space.encode_batch(X)) >= 0)


@pytest.mark.parametrize("threads", [1, 4])
def test_batch_threading_is_deterministic(threads):
    dim = 128
    X = gaussian(700, dim, seed=13)
    Q = gaussian(70, dim, seed=14)
    ref = RaBitQSpace(dim, bits=4, num_threads=1)
    space = RaBitQSpace(dim, bits=4, num_threads=threads)
    codes = space.encode_batch(X)
    np.testing.assert_array_equal(codes, ref.encode_batch(X))
    np.testing.assert_array_equal(space.distance_m_to_n(Q, codes), ref.distance_m_to_n(Q, codes))


def test_strided_input_raises():
    space = RaBitQSpace(32)
    X = gaussian(4, 64, seed=15)
    with pytest.raises(ValueError, match="C-contiguous"):
        space.encode_batch(X[:, ::2])
