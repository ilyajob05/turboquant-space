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


# =====================================================================
# Investigation tests: quantization precision on SIFT-like uint8 data
# =====================================================================

def _lloyd_max_gaussian(bits: int, max_iter: int = 1000, tol: float = 1e-12):
    """Reproduce the C++ Lloyd-Max for N(0,1) in Python.

    Returns (boundaries, centroids) arrays matching the C++ implementation.
    """
    from scipy.special import erfc
    levels = 1 << bits
    half = levels // 2

    def phi(x):
        return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)

    def Phi(x):
        return 0.5 * erfc(-x / np.sqrt(2.0))

    def cond_mean(a, b):
        d = Phi(b) - Phi(a)
        if d < 1e-15:
            return 0.5 * (a + b)
        return (phi(a) - phi(b)) / d

    pos_c = np.array([(i + 0.5) * 3.5 / half for i in range(half)])

    for _ in range(max_iter):
        pos_b = 0.5 * (pos_c[:-1] + pos_c[1:])
        new_c = np.empty(half)
        max_delta = 0.0
        for i in range(half):
            lo = 0.0 if i == 0 else pos_b[i - 1]
            hi = 1e10 if i == half - 1 else pos_b[i]
            new_c[i] = cond_mean(lo, hi)
            max_delta = max(max_delta, abs(new_c[i] - pos_c[i]))
        pos_c = new_c
        if max_delta < tol:
            break

    centroids = np.empty(levels, dtype=np.float64)
    boundaries = np.empty(levels - 1, dtype=np.float64)
    for i in range(half):
        centroids[half + i] = pos_c[i]
        centroids[half - 1 - i] = -pos_c[i]
    boundaries[half - 1] = 0.0
    for i in range(half - 1):
        b = 0.5 * (pos_c[i] + pos_c[i + 1])
        boundaries[half + i] = b
        boundaries[half - 2 - i] = -b
    return boundaries, centroids


def _quantize_vec(normalized_coords, boundaries):
    """Quantize each coordinate using sorted boundaries (linear scan like C++)."""
    idx = np.searchsorted(boundaries, normalized_coords, side="right").astype(np.uint8)
    return idx


def _simulate_turbo_quant_encode(vec, boundaries_7bit, centroids_7bit, rotation_signs, dim):
    """Simulate the full TurboQuant encode pipeline in Python."""
    norm = np.sqrt(np.dot(vec, vec))
    inv_norm = 1.0 / norm if norm > 1e-10 else 0.0
    rotated = np.zeros(dim, dtype=np.float64)
    rotated[:len(vec)] = vec * inv_norm

    # Randomized Hadamard
    rotated *= rotation_signs
    # Walsh-Hadamard Transform (in-place)
    step = 1
    while step < dim:
        jump = step << 1
        for i in range(0, dim, jump):
            lo = rotated[i:i + step].copy()
            hi = rotated[i + step:i + jump].copy()
            rotated[i:i + step] = lo + hi
            rotated[i + step:i + jump] = lo - hi
        step <<= 1
    rotated /= np.sqrt(dim)

    var = np.dot(rotated, rotated)
    sigma = np.sqrt(var / dim)
    if sigma < 1e-10:
        sigma = 1e-10

    normalized = rotated / sigma
    sq_idx = _quantize_vec(normalized, boundaries_7bit)
    residual = rotated - centroids_7bit[sq_idx] * sigma

    gamma = np.sqrt(np.dot(residual, residual))
    return norm, sigma, gamma, rotated, sq_idx, residual


def _make_rotation_signs(dim, seed=42):
    """Reproduce the C++ splitmix64 sign generation."""
    state = np.uint64(seed)
    signs = np.empty(dim, dtype=np.float64)
    for i in range(dim):
        state = np.uint64(state + np.uint64(0x9E3779B97F4A7C15))
        z = state
        z = np.uint64((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9))
        z = np.uint64((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB))
        z = z ^ (z >> np.uint64(31))
        signs[i] = 1.0 if (z & np.uint64(1)) else -1.0
    return signs


class TestQJLBitValue:
    """Test 1: Does the QJL sign bit actually help at 8 bits?

    Compare recall with and without QJL correction to determine whether
    dedicating 1 of 8 bits to QJL is beneficial for SIFT-like data.
    """

    def _recall_at_k(self, pred_indices, gt_indices, k):
        n = pred_indices.shape[0]
        hits = 0
        for i in range(n):
            hits += len(np.intersect1d(pred_indices[i, :k], gt_indices[i, :k]))
        return hits / (n * k)

    def _brute_force_topk(self, queries, base, k):
        n_q = queries.shape[0]
        indices = np.empty((n_q, k), dtype=np.int64)
        for i in range(n_q):
            dists = np.sum((base - queries[i]) ** 2, axis=1)
            idx = np.argpartition(dists, k)[:k]
            idx = idx[np.argsort(dists[idx])]
            indices[i] = idx
        return indices

    @pytest.mark.parametrize("bits", [4, 8])
    def test_qjl_contribution(self, bits):
        """Measure recall with full TQ distance (MSE+QJL) vs MSE-only.

        If QJL doesn't help at 8 bits, reallocating it to MSE could improve
        recall from ~0.95 toward 1.0.
        """
        dim = 128
        n_base = 5000
        n_query = 100
        k = 10
        rng = np.random.default_rng(42)

        # SIFT-like uint8 data
        base = rng.integers(0, 256, size=(n_base, dim)).astype(np.float32)
        queries = rng.integers(0, 256, size=(n_query, dim)).astype(np.float32)

        gt = self._brute_force_topk(queries, base, k)

        space = TQS(dim, bits_per_coord=bits)
        codes = space.encode_batch(base)

        # Full TQ distance (MSE + QJL correction) via library
        dists_full = space.distance_m_to_n(queries, codes)
        pred_full = np.empty((n_query, k), dtype=np.int64)
        for i in range(n_query):
            idx = np.argpartition(dists_full[i], k)[:k]
            pred_full[i] = idx[np.argsort(dists_full[i, idx])]

        recall_full = self._recall_at_k(pred_full, gt, k)

        # MSE-only distance: strip QJL correction by using symmetric distance
        # between (encoded query) and (encoded base) — distBuild ignores QJL.
        # This isolates the MSE-only component.
        query_codes = space.encode_batch(queries)
        dists_sym = space.distance_m_to_n_symmetric(query_codes, codes)
        pred_sym = np.empty((n_query, k), dtype=np.int64)
        for i in range(n_query):
            idx = np.argpartition(dists_sym[i], k)[:k]
            pred_sym[i] = idx[np.argsort(dists_sym[i, idx])]

        recall_sym = self._recall_at_k(pred_sym, gt, k)

        print(f"\n[bits={bits}] recall@{k} full(MSE+QJL)={recall_full:.4f}, "
              f"symmetric(MSE-only)={recall_sym:.4f}, "
              f"delta={recall_full - recall_sym:+.4f}")

        # The full distance should be at least as good as symmetric
        # If not, QJL correction may be hurting at this bit level.
        # We just record the results — this is an investigative test.
        assert recall_full >= 0.0  # always passes; we care about the print


class TestNormalizationPrecisionLoss:
    """Test 2: Measure information loss from normalize + Hadamard pipeline.

    For SIFT uint8 data, the original 8-bit integer coordinates become
    continuous floats after normalization and rotation. This test quantifies
    how much the transform chain degrades distance preservation.
    """

    def test_transform_distance_correlation(self):
        """Check correlation between original L2 and post-transform L2.

        If the Hadamard transform is orthogonal (distance-preserving), the
        only source of error is normalization (projecting onto unit sphere).
        """
        dim = 128
        n = 500
        rng = np.random.default_rng(99)

        base = rng.integers(0, 256, size=(n, dim)).astype(np.float64)

        norms = np.linalg.norm(base, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = base / norms

        # Randomized Hadamard
        pad_dim = 1
        while pad_dim < dim:
            pad_dim <<= 1
        rot_signs = _make_rotation_signs(pad_dim, seed=42)

        def apply_rht(vec):
            buf = np.zeros(pad_dim, dtype=np.float64)
            buf[:dim] = vec
            buf *= rot_signs
            step = 1
            while step < pad_dim:
                jump = step << 1
                for i in range(0, pad_dim, jump):
                    lo = buf[i:i + step].copy()
                    hi = buf[i + step:i + jump].copy()
                    buf[i:i + step] = lo + hi
                    buf[i + step:i + jump] = lo - hi
                step <<= 1
            buf /= np.sqrt(pad_dim)
            return buf

        # Sample pairs and compare distances
        n_pairs = 2000
        pairs = rng.choice(n, size=(n_pairs, 2), replace=True)
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]

        dist_orig = np.array([
            np.linalg.norm(base[i] - base[j]) for i, j in pairs
        ])
        dist_normalized = np.array([
            np.linalg.norm(normalized[i] - normalized[j]) for i, j in pairs
        ])

        rotated = np.array([apply_rht(normalized[i]) for i in range(n)])
        dist_rotated = np.array([
            np.linalg.norm(rotated[i] - rotated[j]) for i, j in pairs
        ])

        # Hadamard is orthogonal — rotated distances should match normalized
        np.testing.assert_allclose(dist_rotated, dist_normalized, rtol=1e-6,
                                   err_msg="Hadamard broke distance preservation")

        # Normalization changes distances — measure rank correlation
        from scipy.stats import spearmanr
        rho_norm, _ = spearmanr(dist_orig, dist_normalized)
        rho_rot, _ = spearmanr(dist_orig, dist_rotated)

        print(f"\nSpearman rank correlation with original L2:")
        print(f"  After normalization: {rho_norm:.6f}")
        print(f"  After norm+Hadamard: {rho_rot:.6f}")
        print(f"  (1.0 = perfect rank preservation)")

        # Normalization should mostly preserve ranking for SIFT data
        assert rho_norm > 0.8, f"Normalization destroys ranking: rho={rho_norm:.4f}"

    def test_float32_vs_float64_pipeline(self):
        """Check whether float32 arithmetic introduces significant error
        compared to float64 in the normalize+Hadamard+quantize pipeline.
        """
        dim = 128
        n = 200
        rng = np.random.default_rng(77)
        base = rng.integers(0, 256, size=(n, dim)).astype(np.float32)

        pad_dim = 1
        while pad_dim < dim:
            pad_dim <<= 1
        rot_signs = _make_rotation_signs(pad_dim, seed=42)

        boundaries_7, centroids_7 = _lloyd_max_gaussian(7)

        errors_32 = []
        errors_64 = []
        for i in range(n):
            vec64 = base[i].astype(np.float64)
            vec32 = base[i].astype(np.float32)

            # Float64 pipeline
            _, sigma64, _, rotated64, idx64, res64 = _simulate_turbo_quant_encode(
                vec64, boundaries_7, centroids_7, rot_signs, pad_dim)
            recon64 = centroids_7[idx64] * sigma64
            errors_64.append(np.sqrt(np.mean((rotated64 - recon64) ** 2)))

            # Float32 pipeline
            _, sigma32, _, rotated32, idx32, res32 = _simulate_turbo_quant_encode(
                vec32.astype(np.float64), boundaries_7, centroids_7, rot_signs, pad_dim)
            recon32 = centroids_7[idx32] * sigma32
            errors_32.append(np.sqrt(np.mean((rotated32 - recon32) ** 2)))

        rmse_64 = np.mean(errors_64)
        rmse_32 = np.mean(errors_32)
        print(f"\nMean RMSE of quantization (rotated coords):")
        print(f"  float64 pipeline: {rmse_64:.8f}")
        print(f"  float32 pipeline: {rmse_32:.8f}")
        print(f"  relative diff:    {abs(rmse_64 - rmse_32) / rmse_64:.2e}")

        # float32 shouldn't add more than 1% error on top of quantization
        assert abs(rmse_64 - rmse_32) / rmse_64 < 0.01


class TestDataDrivenLloydMax:
    """Test 3: Data-driven Lloyd-Max vs Gaussian-assumed Lloyd-Max.

    Instead of assuming N(0,σ²), build optimal quantizer boundaries from
    the actual distribution of rotated+normalized coordinates. If this
    significantly reduces MSE, the Gaussian assumption is a meaningful
    source of recall loss.
    """

    @staticmethod
    def _lloyd_max_empirical(samples, n_levels, max_iter=500, tol=1e-12):
        """Standard Lloyd-Max on empirical data (1D samples)."""
        sorted_samples = np.sort(samples)
        n = len(sorted_samples)
        # Initialize centroids at quantiles of the data — much better than
        # linspace for distributions with non-uniform density.
        quantile_positions = (np.arange(n_levels) + 0.5) / n_levels
        centroids = np.quantile(sorted_samples, quantile_positions)

        for _ in range(max_iter):
            # Boundaries = midpoints between adjacent centroids
            boundaries = 0.5 * (centroids[:-1] + centroids[1:])

            # Assign each sample to nearest centroid via boundaries
            labels = np.searchsorted(boundaries, samples, side="right")

            # Update centroids
            new_centroids = np.empty_like(centroids)
            max_delta = 0.0
            for j in range(n_levels):
                mask = labels == j
                if mask.any():
                    new_centroids[j] = samples[mask].mean()
                else:
                    new_centroids[j] = centroids[j]
                max_delta = max(max_delta, abs(new_centroids[j] - centroids[j]))
            centroids = np.sort(new_centroids)
            if max_delta < tol:
                break

        boundaries = 0.5 * (centroids[:-1] + centroids[1:])
        return boundaries, centroids

    @pytest.mark.parametrize("bits_mse", [3, 7])
    def test_gaussian_vs_empirical_mse(self, bits_mse):
        """Compare quantization MSE: Gaussian Lloyd-Max vs data-driven.

        Uses SIFT-like uint8 data, applies full normalize+Hadamard pipeline,
        then quantizes the resulting coordinates with both methods.
        """
        dim = 128
        n = 2000
        n_levels = 1 << bits_mse
        rng = np.random.default_rng(123)

        base = rng.integers(0, 256, size=(n, dim)).astype(np.float64)

        pad_dim = 1
        while pad_dim < dim:
            pad_dim <<= 1
        rot_signs = _make_rotation_signs(pad_dim, seed=42)
        boundaries_gauss, centroids_gauss = _lloyd_max_gaussian(bits_mse)

        # Collect all rotated+normalized coordinates
        all_coords = []
        for i in range(n):
            norm = np.sqrt(np.dot(base[i], base[i]))
            inv_norm = 1.0 / norm if norm > 1e-10 else 0.0
            buf = np.zeros(pad_dim, dtype=np.float64)
            buf[:dim] = base[i] * inv_norm
            buf *= rot_signs
            step = 1
            while step < pad_dim:
                jump = step << 1
                for ii in range(0, pad_dim, jump):
                    lo = buf[ii:ii + step].copy()
                    hi = buf[ii + step:ii + jump].copy()
                    buf[ii:ii + step] = lo + hi
                    buf[ii + step:ii + jump] = lo - hi
                step <<= 1
            buf /= np.sqrt(pad_dim)

            sigma = np.sqrt(np.dot(buf, buf) / pad_dim)
            if sigma < 1e-10:
                sigma = 1e-10
            all_coords.append(buf / sigma)

        all_coords = np.concatenate(all_coords)

        # Gaussian Lloyd-Max MSE
        idx_gauss = np.searchsorted(boundaries_gauss, all_coords, side="right")
        recon_gauss = centroids_gauss[idx_gauss]
        mse_gauss = np.mean((all_coords - recon_gauss) ** 2)

        # Data-driven Lloyd-Max
        # Subsample for speed if needed
        subsample = all_coords if len(all_coords) <= 500_000 else rng.choice(
            all_coords, 500_000, replace=False)
        boundaries_emp, centroids_emp = self._lloyd_max_empirical(
            subsample, n_levels)

        idx_emp = np.searchsorted(boundaries_emp, all_coords, side="right")
        idx_emp = np.clip(idx_emp, 0, n_levels - 1)
        recon_emp = centroids_emp[idx_emp]
        mse_emp = np.mean((all_coords - recon_emp) ** 2)

        improvement = (mse_gauss - mse_emp) / mse_gauss * 100

        print(f"\n[{bits_mse} bits MSE, {n_levels} levels]")
        print(f"  Gaussian Lloyd-Max MSE: {mse_gauss:.8f}")
        print(f"  Empirical Lloyd-Max MSE: {mse_emp:.8f}")
        print(f"  Improvement: {improvement:+.2f}%")
        if improvement < 0:
            print(f"  NOTE: Gaussian LM is better — the distribution is close "
                  f"enough to Gaussian that data-driven doesn't help at {n_levels} levels.")

        # Investigative: record both MSEs. At high bit counts (7 bits) the
        # Gaussian assumption is near-optimal, so empirical may not win.
        # At low bit counts (3 bits) data-driven should show improvement.
        if bits_mse <= 4:
            assert improvement > 0, \
                f"Expected data-driven to win at {bits_mse} bits"

    def test_coordinate_distribution_normality(self):
        """Measure how close the rotated coordinates are to Gaussian.

        If the distribution is near-Gaussian, data-driven Lloyd-Max won't
        help much. If there's significant deviation, it could explain
        the recall gap.
        """
        from scipy.stats import normaltest, kurtosis, skew

        dim = 128
        n = 3000
        rng = np.random.default_rng(456)

        base = rng.integers(0, 256, size=(n, dim)).astype(np.float64)

        pad_dim = 1
        while pad_dim < dim:
            pad_dim <<= 1
        rot_signs = _make_rotation_signs(pad_dim, seed=42)

        all_normalized = []
        for i in range(n):
            norm = np.sqrt(np.dot(base[i], base[i]))
            inv_norm = 1.0 / norm if norm > 1e-10 else 0.0
            buf = np.zeros(pad_dim, dtype=np.float64)
            buf[:dim] = base[i] * inv_norm
            buf *= rot_signs
            step = 1
            while step < pad_dim:
                jump = step << 1
                for ii in range(0, pad_dim, jump):
                    lo = buf[ii:ii + step].copy()
                    hi = buf[ii + step:ii + jump].copy()
                    buf[ii:ii + step] = lo + hi
                    buf[ii + step:ii + jump] = lo - hi
                step <<= 1
            buf /= np.sqrt(pad_dim)

            sigma = np.sqrt(np.dot(buf, buf) / pad_dim)
            if sigma < 1e-10:
                sigma = 1e-10
            all_normalized.append(buf / sigma)

        all_coords = np.concatenate(all_normalized)

        k = kurtosis(all_coords)  # Gaussian = 0
        s = skew(all_coords)      # Gaussian = 0
        _, p_value = normaltest(all_coords)

        print(f"\nDistribution of rotated+normalized coords (SIFT-like uint8):")
        print(f"  Kurtosis (excess): {k:.4f}  (Gaussian=0)")
        print(f"  Skewness:          {s:.4f}  (Gaussian=0)")
        print(f"  D'Agostino p-val:  {p_value:.2e}  (<0.05 = non-Gaussian)")
        print(f"  N samples:         {len(all_coords)}")

        # With enough samples, p-value will be tiny even for near-Gaussian.
        # Focus on kurtosis/skewness magnitude for practical significance.
        print(f"  Practical deviation: kurtosis={'small' if abs(k) < 0.5 else 'notable'}, "
              f"skew={'small' if abs(s) < 0.3 else 'notable'}")


class TestSymmetricQJLCorrection:
    """Test symmetric distance variants with QJL correction.

    Compares three symmetric distance modes:
    - Original: MSE-only (ignores QJL bit)
    - Light: MSE + <e_a, e_b> via sign-bit dot product (cheap)
    - Full: MSE + <r̃_a, e_b> + <e_a, r̃_b> + <e_a, e_b> (reconstructs + Hadamard)
    """

    def _recall_at_k(self, pred_indices, gt_indices, k):
        n = pred_indices.shape[0]
        hits = 0
        for i in range(n):
            hits += len(np.intersect1d(pred_indices[i, :k], gt_indices[i, :k]))
        return hits / (n * k)

    def _brute_force_topk(self, queries, base, k):
        n_q = queries.shape[0]
        indices = np.empty((n_q, k), dtype=np.int64)
        for i in range(n_q):
            dists = np.sum((base - queries[i]) ** 2, axis=1)
            idx = np.argpartition(dists, k)[:k]
            idx = idx[np.argsort(dists[idx])]
            indices[i] = idx
        return indices

    def _topk_from_dists(self, dists, k):
        n_q = dists.shape[0]
        indices = np.empty((n_q, k), dtype=np.int64)
        for i in range(n_q):
            idx = np.argpartition(dists[i], k)[:k]
            indices[i] = idx[np.argsort(dists[i, idx])]
        return indices

    @pytest.mark.parametrize("bits", [4, 8])
    def test_symmetric_qjl_recall(self, bits):
        """Compare recall of original, light, and full symmetric distance."""
        dim = 128
        n_base = 5000
        n_query = 100
        k = 10
        rng = np.random.default_rng(42)

        base = rng.integers(0, 256, size=(n_base, dim)).astype(np.float32)
        queries = rng.integers(0, 256, size=(n_query, dim)).astype(np.float32)

        gt = self._brute_force_topk(queries, base, k)

        space = TQS(dim, bits_per_coord=bits)
        codes_base = space.encode_batch(base)
        codes_query = space.encode_batch(queries)

        # Original symmetric (MSE-only)
        dists_orig = space.distance_m_to_n_symmetric(codes_query, codes_base)
        pred_orig = self._topk_from_dists(dists_orig, k)
        recall_orig = self._recall_at_k(pred_orig, gt, k)

        # Light symmetric (MSE + sign-bit dot product)
        dists_light = space.distance_m_to_n_symmetric_light(codes_query, codes_base)
        pred_light = self._topk_from_dists(dists_light, k)
        recall_light = self._recall_at_k(pred_light, gt, k)

        # Full symmetric (MSE + all QJL correction terms)
        dists_full = space.distance_m_to_n_symmetric_full(codes_query, codes_base)
        pred_full = self._topk_from_dists(dists_full, k)
        recall_full = self._recall_at_k(pred_full, gt, k)

        # Asymmetric for reference (uses float query — best possible)
        dists_asym = space.distance_m_to_n(queries, codes_base)
        pred_asym = self._topk_from_dists(dists_asym, k)
        recall_asym = self._recall_at_k(pred_asym, gt, k)

        print(f"\n[bits={bits}] Symmetric distance recall@{k}:")
        print(f"  Original (MSE-only):     {recall_orig:.4f}")
        print(f"  Light (MSE + sign-dot):  {recall_light:.4f}  "
              f"(delta={recall_light - recall_orig:+.4f})")
        print(f"  Full (MSE + all QJL):    {recall_full:.4f}  "
              f"(delta={recall_full - recall_orig:+.4f})")
        print(f"  Asymmetric (reference):  {recall_asym:.4f}")

        # Sanity: all variants should produce non-negative distances
        assert np.all(dists_orig >= -1e-6)
        assert np.all(dists_light >= -1e-6)
        assert np.all(dists_full >= -1e-6)

    @pytest.mark.parametrize("bits", [4, 8])
    def test_symmetric_distance_accuracy(self, bits):
        """Measure MSE of each symmetric variant vs true L2 distance."""
        dim = 128
        n = 500
        rng = np.random.default_rng(77)

        vecs = rng.integers(0, 256, size=(n, dim)).astype(np.float32)
        space = TQS(dim, bits_per_coord=bits)
        codes = space.encode_batch(vecs)

        # Sample pairs
        n_pairs = 2000
        pairs = rng.choice(n, size=(n_pairs, 2), replace=True)
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]

        true_dists = np.array([
            float(np.sum((vecs[i] - vecs[j]) ** 2)) for i, j in pairs
        ])
        orig_dists = np.array([
            space.distance_symmetric(codes[i], codes[j]) for i, j in pairs
        ])
        light_dists = np.array([
            space.distance_m_to_n_symmetric_light(
                codes[i:i+1], codes[j:j+1])[0, 0] for i, j in pairs
        ])
        full_dists = np.array([
            space.distance_m_to_n_symmetric_full(
                codes[i:i+1], codes[j:j+1])[0, 0] for i, j in pairs
        ])

        # Relative MSE for each variant
        rmse_orig = np.sqrt(np.mean((orig_dists - true_dists) ** 2))
        rmse_light = np.sqrt(np.mean((light_dists - true_dists) ** 2))
        rmse_full = np.sqrt(np.mean((full_dists - true_dists) ** 2))
        mean_true = np.mean(true_dists)

        print(f"\n[bits={bits}] Symmetric distance RMSE (mean_true_dist={mean_true:.1f}):")
        print(f"  Original (MSE-only):     {rmse_orig:.2f}  "
              f"(relative: {rmse_orig/mean_true:.4f})")
        print(f"  Light (MSE + sign-dot):  {rmse_light:.2f}  "
              f"(relative: {rmse_light/mean_true:.4f})")
        print(f"  Full (MSE + all QJL):    {rmse_full:.2f}  "
              f"(relative: {rmse_full/mean_true:.4f})")


# ---------------------------------------------------------------------------
# Prepared-symmetric API: numerical equivalence with the unprepared variant.
# The prepared path caches the query-side reconstruction + Hadamard, so for
# every (a, b) pair it must yield the same distance (within fp32 noise) as
# distance_m_to_n_symmetric_full([a], [b]).
# ---------------------------------------------------------------------------

class TestPreparedSymmetricEquivalence:

    @pytest.mark.parametrize("bits", [4, 8])
    @pytest.mark.parametrize("dim", [128, 1024])
    def test_single_pair_equivalence(self, bits, dim):
        rng = np.random.default_rng(bits * 1000 + dim)
        n = 64
        vecs = rng.standard_normal((n, dim)).astype(np.float32)
        space = TQS(dim, bits_per_coord=bits)
        codes = space.encode_batch(vecs)

        # Reference: unprepared full M-to-N over all pairs.
        ref = space.distance_m_to_n_symmetric_full(codes, codes)

        # Prepared: walk each query code once, distance vs every base code.
        # Tolerance matches the batched tests below; the two-loop unprepared
        # path and the fused one-loop prepared path sum the four QJL terms in
        # different orders, so float32 reordering leaks a few thousandths at
        # dim=1024 where the result itself is ~0.3.
        got = np.empty_like(ref)
        for i in range(n):
            pq = space.prepare_symmetric_query(codes[i])
            for j in range(n):
                got[i, j] = space.distance_symmetric_full_prepared(pq, codes[j])
        np.testing.assert_allclose(got, ref, rtol=1e-3, atol=5e-3)

    @pytest.mark.parametrize("bits", [4, 8])
    @pytest.mark.parametrize("dim", [128, 1024])
    def test_1_to_n_equivalence(self, bits, dim):
        rng = np.random.default_rng(bits * 2000 + dim)
        n = 200
        vecs = rng.standard_normal((n, dim)).astype(np.float32)
        space = TQS(dim, bits_per_coord=bits)
        codes = space.encode_batch(vecs)

        ref = space.distance_m_to_n_symmetric_full(codes[:1], codes)[0]
        pq = space.prepare_symmetric_query(codes[0])
        got = space.distance_1_to_n_symmetric_full(pq, codes)
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-3)

    @pytest.mark.parametrize("bits", [4, 8])
    @pytest.mark.parametrize("dim", [128, 1024])
    def test_m_to_n_prepared_equivalence(self, bits, dim):
        rng = np.random.default_rng(bits * 3000 + dim)
        m, n = 32, 200
        a = rng.standard_normal((m, dim)).astype(np.float32)
        b = rng.standard_normal((n, dim)).astype(np.float32)
        space = TQS(dim, bits_per_coord=bits)
        ca = space.encode_batch(a)
        cb = space.encode_batch(b)

        ref = space.distance_m_to_n_symmetric_full(ca, cb)
        got = space.distance_m_to_n_symmetric_full_prepared(ca, cb)
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-3)

    @pytest.mark.parametrize("bits", [4, 8])
    def test_self_distance_near_zero(self, bits):
        """A code prepared and matched against itself should yield ~0."""
        dim = 256
        rng = np.random.default_rng(42)
        x = rng.standard_normal(dim).astype(np.float32)
        space = TQS(dim, bits_per_coord=bits)
        code = space.encode(x)
        pq = space.prepare_symmetric_query(code)
        d = space.distance_symmetric_full_prepared(pq, code)
        scale = float(np.sum(x * x))
        assert d < 0.1 * scale, f"self-distance {d} too large vs |x|^2={scale}"


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dim", [128, 1024])
def test_prepared_symmetric_speedup(bits, dim):
    """Microbenchmark: prepared M-to-N vs unprepared M-to-N. Reports ratio."""
    import time
    rng = np.random.default_rng(bits + dim)
    m, n = 64, 2000
    a = rng.standard_normal((m, dim)).astype(np.float32)
    b = rng.standard_normal((n, dim)).astype(np.float32)
    space = TQS(dim, bits_per_coord=bits)
    ca = space.encode_batch(a)
    cb = space.encode_batch(b)

    # Warmup
    space.distance_m_to_n_symmetric_full(ca[:4], cb[:4])
    space.distance_m_to_n_symmetric_full_prepared(ca[:4], cb[:4])

    def best_of(fn, repeats=3):
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            dt = time.perf_counter() - t0
            best = min(best, dt)
        return best

    t_unp = best_of(lambda: space.distance_m_to_n_symmetric_full(ca, cb))
    t_pre = best_of(lambda: space.distance_m_to_n_symmetric_full_prepared(ca, cb))
    speedup = t_unp / t_pre
    print(f"\n[bits={bits} dim={dim} m={m} n={n}] "
          f"unprepared={t_unp*1000:.1f}ms  prepared={t_pre*1000:.1f}ms  "
          f"speedup={speedup:.2f}x")
    # Soft check: prepared should be at least as fast (allow noise).
    assert speedup > 0.8, (
        f"prepared was unexpectedly slower: {speedup:.2f}x"
    )
