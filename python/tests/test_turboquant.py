import numpy as np
import pytest

from turboquant import TurboQuantSpace as TQS

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.parametrize("dim", [32, 64, 128, 256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("bits", [4, 8])
def test_distance_approximates_l2(dim, bits):
    space = TQS(dim, bits_per_coord=bits)
    rng = np.random.default_rng(dim * 10 + bits)
    q = rng.standard_normal(dim).astype(np.float32)
    x = rng.standard_normal(dim).astype(np.float32)

    code = space.encode(x)
    brute = float(np.linalg.norm(q - x) ** 2)
    tq = space.distance(q, code)

    # TurboQuant is an approximation; tolerance loosens at low bit budgets.
    tol_rel = 0.25 if bits >= 8 else 0.5
    assert abs(tq - brute) < tol_rel * brute + 1.0


def test_self_distance_near_zero():
    dim = 512
    space = TQS(dim, bits_per_coord=8)
    rng = np.random.default_rng(7)
    x = rng.standard_normal(dim).astype(np.float32)
    code = space.encode(x)
    d = space.distance(x, code)
    brute = float(np.linalg.norm(x) ** 2)
    # self-distance should be much smaller than ||x||^2
    assert d < 0.1 * brute


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
@pytest.mark.parametrize("dim", [128, 512])
@pytest.mark.parametrize("bits", [4, 8])
def test_torch_zero_copy(dim, bits):
    import torch
    space = TQS(dim, bits_per_coord=bits)
    torch.manual_seed(dim * 100 + bits)
    q = torch.randn(dim, dtype=torch.float32)
    x = torch.randn(dim, dtype=torch.float32)

    code = space.encode(x.numpy())
    dist = space.distance(q.numpy(), code)
    brute = float(((q - x) ** 2).sum())

    tol_rel = 0.25 if bits >= 8 else 0.5
    assert abs(dist - brute) < tol_rel * brute + 1.0


def test_encode_into_inplace():
    space = TQS(256)
    x = np.random.randn(256).astype(np.float32)
    code = np.empty(space.code_size_bytes(), dtype=np.uint8)

    space.encode_into(x, code)
    code2 = space.encode(x)
    np.testing.assert_array_equal(code, code2)


def test_encode_batch():
    dim = 128
    n = 17
    space = TQS(dim, bits_per_coord=8)
    rng = np.random.default_rng(1)
    X = rng.standard_normal((n, dim)).astype(np.float32)

    codes = space.encode_batch(X)
    assert codes.shape == (n, space.code_size_bytes())

    for i in range(n):
        single = space.encode(X[i])
        np.testing.assert_array_equal(codes[i], single)


def test_distance_symmetric():
    dim = 256
    space = TQS(dim, bits_per_coord=8)
    rng = np.random.default_rng(2)
    x = rng.standard_normal(dim).astype(np.float32)
    y = rng.standard_normal(dim).astype(np.float32)

    ca = space.encode(x)
    cb = space.encode(y)
    d_sym = space.distance_symmetric(ca, cb)
    brute = float(np.linalg.norm(x - y) ** 2)
    assert d_sym >= 0.0
    # symmetric quantized distance is noisy, just sanity-check order of magnitude
    assert abs(d_sym - brute) < 0.5 * brute + 1.0


def test_distance_1_to_n():
    dim = 128
    n = 8
    space = TQS(dim, bits_per_coord=8)
    rng = np.random.default_rng(3)
    q = rng.standard_normal(dim).astype(np.float32)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    codes = space.encode_batch(X)

    dists = space.distance_1_to_n(q, codes)
    assert dists.shape == (n,)
    for i in range(n):
        assert abs(dists[i] - space.distance(q, codes[i])) < 1e-4


def test_distance_m_to_n():
    dim = 64
    m, n = 5, 7
    space = TQS(dim, bits_per_coord=8)
    rng = np.random.default_rng(4)
    Q = rng.standard_normal((m, dim)).astype(np.float32)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    codes = space.encode_batch(X)

    D = space.distance_m_to_n(Q, codes)
    assert D.shape == (m, n)
    for i in range(m):
        for j in range(n):
            assert abs(D[i, j] - space.distance(Q[i], codes[j])) < 1e-4


def test_distance_m_to_n_symmetric():
    dim = 64
    m, n = 4, 6
    space = TQS(dim, bits_per_coord=8)
    rng = np.random.default_rng(5)
    A = rng.standard_normal((m, dim)).astype(np.float32)
    B = rng.standard_normal((n, dim)).astype(np.float32)
    ca = space.encode_batch(A)
    cb = space.encode_batch(B)

    D = space.distance_m_to_n_symmetric(ca, cb)
    assert D.shape == (m, n)
    for i in range(m):
        for j in range(n):
            assert abs(D[i, j] - space.distance_symmetric(ca[i], cb[j])) < 1e-4
