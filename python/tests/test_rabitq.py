"""RaBitQ against a float32 NumPy oracle.

The oracle repeats the C++ SRHT (splitmix64 LSB, unnormalized FWHT, scale
1/sqrt(D)). 1-bit sign slots must match that oracle byte for byte. 4-bit and
8-bit slots follow Extended RaBitQ (arXiv:2409.09913, Algorithm 1): the same
sweep over critical rescales, then the centered-grid inner product. The
oracle does not import TurboQuant's Lloyd-Max helpers.
"""

import numpy as np
import pytest

from turboquant import RaBitQSpace

MASK64 = (1 << 64) - 1


def padded_dim(dim):
    n = 4 if dim < 4 else dim
    p = 1
    while p < n:
        p <<= 1
    return p


def splitmix_signs(d, seed):
    state = seed & MASK64
    signs = np.empty(d, np.float32)
    for i in range(d):
        state = (state + 0x9E3779B97F4A7C15) & MASK64
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        z = (z ^ (z >> 31)) & MASK64
        signs[i] = np.float32(1.0 if (z & 1) else -1.0)
    return signs


def wht_f32(data):
    y = np.array(data, dtype=np.float32, copy=True)
    d = y.shape[0]
    step = 1
    while step < d:
        jump = step << 1
        for i in range(0, d, jump):
            for j in range(step):
                a = y[i + j]
                b = y[i + step + j]
                y[i + j] = a + b
                y[i + step + j] = a - b
        step <<= 1
    y *= np.float32(1.0 / np.sqrt(np.float32(d)))
    return y


def f32(v):
    return np.float32(v)


def residual_unit(vec, centroid):
    d = vec.shape[0]
    D = padded_dim(d)
    buf = np.zeros(D, np.float32)
    acc = f32(0.0)
    for i in range(d):
        v = f32(f32(vec[i]) - f32(centroid[i]))
        buf[i] = v
        acc = f32(acc + v * v)
    norm = f32(np.sqrt(acc))
    if norm > 0:
        inv = f32(f32(1.0) / norm)
        for i in range(d):
            buf[i] = f32(buf[i] * inv)
    return buf, norm


def oracle_encode(vec, seed, centroid):
    D = padded_dim(vec.shape[0])
    unit, norm = residual_unit(vec, centroid)
    rot = wht_f32(unit * splitmix_signs(D, seed))
    inv = f32(1.0 / np.sqrt(f32(D)))
    bits = np.zeros((D + 7) // 8, np.uint8)
    dot = f32(0.0)
    for i in range(D):
        positive = bool(rot[i] >= 0)
        if positive:
            bits[i >> 3] = np.uint8(bits[i >> 3] | np.uint8(1 << (i & 7)))
        s = inv if positive else f32(-inv)
        dot = f32(dot + s * rot[i])
    meta = np.array([norm, dot], np.float32).tobytes()
    return np.frombuffer(bits.tobytes() + meta, dtype=np.uint8).copy()


def oracle_distance(query, slot, seed, centroid):
    D = padded_dim(query.shape[0])
    sign_bytes = (D + 7) // 8
    bits = slot[:sign_bytes]
    meta = np.frombuffer(slot[sign_bytes:].tobytes(), dtype=np.float32)
    xnorm = f32(meta[0])
    dot_factor = f32(meta[1])
    q_unit, qnorm = residual_unit(query, centroid)
    rot = wht_f32(q_unit * splitmix_signs(D, seed))
    inv = f32(1.0 / np.sqrt(f32(D)))
    cube = f32(0.0)
    if qnorm > 0:
        for i in range(D):
            bit = (int(bits[i >> 3]) >> (i & 7)) & 1
            s = inv if bit else f32(-inv)
            cube = f32(cube + s * rot[i])
    ip = f32(cube / dot_factor)
    return float(f32(xnorm * xnorm + qnorm * qnorm - f32(2.0) * xnorm * qnorm * ip))


def test_slot_size_and_padding():
    space6 = RaBitQSpace(6, rot_seed=7)
    space8 = RaBitQSpace(8, rot_seed=7)
    assert space6.padded_dim() == 8
    assert space8.padded_dim() == 8
    assert space6.code_size_bytes() == (8 + 7) // 8 + 8
    assert space8.code_size_bytes() == space6.code_size_bytes()


@pytest.mark.parametrize("dim", [6, 8])
def test_encode_bits_match_oracle(dim):
    seed = 42
    rng = np.random.default_rng(dim)
    centroid = rng.standard_normal(dim).astype(np.float32)
    x = rng.standard_normal(dim).astype(np.float32)
    space = RaBitQSpace(dim, rot_seed=seed, centroid=centroid)
    got = space.encode(x)
    expect = oracle_encode(x, seed, centroid)
    np.testing.assert_array_equal(got, expect)


@pytest.mark.parametrize("dim", [8, 128])
def test_distance_matches_oracle(dim):
    seed = 42
    rng = np.random.default_rng(1000 + dim)
    centroid = rng.standard_normal(dim).astype(np.float32) * np.float32(0.1)
    space = RaBitQSpace(dim, rot_seed=seed, centroid=centroid)
    for k in range(32):
        x = rng.standard_normal(dim).astype(np.float32)
        q = rng.standard_normal(dim).astype(np.float32)
        code = space.encode(x)
        got = space.distance(q, code)
        expect = oracle_distance(q, code, seed, centroid)
        np.testing.assert_allclose(got, expect, rtol=1e-5, atol=1e-5)

@pytest.mark.parametrize("bits", [1, 4, 8])
@pytest.mark.parametrize("dim", [4, 6, 8, 128])
def test_selected_kernel_matches_scalar(bits, dim):
    """The ISA kernel and the scalar kernel estimate the same squared L2."""
    rng = np.random.default_rng(5000 + bits * 1000 + dim)
    centroid = rng.standard_normal(dim).astype(np.float32) * np.float32(0.1)
    space = RaBitQSpace(dim, rot_seed=11, centroid=centroid, bits=bits)
    for _ in range(8):
        x = rng.standard_normal(dim).astype(np.float32)
        q = rng.standard_normal(dim).astype(np.float32)
        code = space.encode(x)
        fast = space.distance(q, code)
        slow = space.distance_scalar(q, code)
        np.testing.assert_allclose(fast, slow, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("bits", [1, 4, 8])
@pytest.mark.parametrize("dim", [8, 128])
def test_distance_1_to_n_matches_single(bits, dim):
    """One prepared query scanned over slots matches distance() per slot."""
    rng = np.random.default_rng(7000 + bits * 100 + dim)
    centroid = rng.standard_normal(dim).astype(np.float32) * np.float32(0.1)
    space = RaBitQSpace(dim, rot_seed=11, centroid=centroid, bits=bits)
    n = 16
    base = rng.standard_normal((n, dim)).astype(np.float32)
    query = rng.standard_normal(dim).astype(np.float32)
    codes = np.vstack([np.asarray(space.encode(base[i]), dtype=np.uint8) for i in range(n)])
    batch = np.asarray(space.distance_1_to_n(query, codes), dtype=np.float64)
    single = np.array([space.distance(query, codes[i]) for i in range(n)], dtype=np.float64)
    assert batch.shape == (n,)
    np.testing.assert_allclose(batch, single, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("bits", [1, 4, 8])
def test_distance_m_to_n_matches_rows(bits):
    """Each row of distance_m_to_n is distance_1_to_n for that query."""
    dim = 32
    rng = np.random.default_rng(8000 + bits)
    space = RaBitQSpace(dim, rot_seed=13, bits=bits)
    n, m = 12, 3
    base = rng.standard_normal((n, dim)).astype(np.float32)
    queries = rng.standard_normal((m, dim)).astype(np.float32)
    codes = np.vstack([np.asarray(space.encode(base[i]), dtype=np.uint8) for i in range(n)])
    matrix = np.asarray(space.distance_m_to_n(queries, codes), dtype=np.float64)
    assert matrix.shape == (m, n)
    for i in range(m):
        row = np.asarray(space.distance_1_to_n(queries[i], codes), dtype=np.float64)
        np.testing.assert_allclose(matrix[i], row, rtol=1e-5, atol=1e-5)


def test_distance_batch_rejects_bad_code_bytes():
    space = RaBitQSpace(16, rot_seed=1, bits=4)
    query = np.ones(16, np.float32)
    bad = np.zeros(space.code_size_bytes() + 1, np.uint8)
    with pytest.raises(Exception):
        space.distance_1_to_n(query, bad)


def test_distance_kernel_matches_this_machine():
    import platform

    name = RaBitQSpace(32, bits=1).distance_kernel()
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        assert name == "1-neon"
    elif machine in ("x86_64", "amd64"):
        assert name in ("1-avx2", "1-scalar")
    else:
        assert name == "1-scalar"
    assert RaBitQSpace(32, bits=4).distance_kernel().startswith("4-")
    assert RaBitQSpace(32, bits=8).distance_kernel().startswith("8-")


def test_wrong_code_size_is_rejected():
    space = RaBitQSpace(16, rot_seed=1)
    q = np.ones(16, np.float32)
    bad = np.zeros(4, np.uint8)
    with pytest.raises(Exception) as exc:
        space.distance(q, bad)
    assert "code" in str(exc.value).lower() or "size" in str(exc.value).lower() or "byte" in str(exc.value).lower()


def test_zero_residual_is_rejected():
    dim = 8
    c = np.arange(dim, dtype=np.float32)
    space = RaBitQSpace(dim, rot_seed=3, centroid=c)
    with pytest.raises(Exception) as exc:
        space.encode(c.copy())
    assert "padded_dim=8" in str(exc.value)


def test_query_at_centroid_is_squared_norm():
    dim = 8
    seed = 9
    c = np.zeros(dim, np.float32)
    space = RaBitQSpace(dim, rot_seed=seed, centroid=c)
    x = np.arange(1, dim + 1, dtype=np.float32)
    code = space.encode(x)
    got = space.distance(c, code)
    expect = oracle_distance(c, code, seed, c)
    np.testing.assert_allclose(got, expect, rtol=1e-5, atol=1e-5)
    # The query is the centroid, so the estimate is ||x - c||^2, not a
    # self-distance tolerance borrowed from TurboQuant.
    brute = float(np.dot(x, x))
    assert abs(got - brute) < 1e-4 * brute + 1e-4


def _extended_codes(rot, bits):
    """Mirror of RaBitQSpace::quantizeExtended. `rot` is float32, length D."""
    levels = 1 << bits
    center = np.float32(127.5 if bits == 8 else 7.5)
    start_pos = levels >> 1
    start_neg = start_pos - 1
    dim = int(rot.shape[0])
    code = np.empty(dim, np.int32)
    dot = 0.0
    normsq = 0.0
    events = []
    for i in range(dim):
        oi = rot[i]
        code[i] = start_pos if oi >= 0 else start_neg
        value = float(code[i]) - float(center)
        dot += value * float(oi)
        normsq += value * value
        if oi > 0:
            for k in range(start_pos + 1, levels):
                num = np.float32(np.float32(np.float32(k) - np.float32(0.5)) - center)
                events.append((np.float32(num / oi), i))
        elif oi < 0:
            for m in range(start_neg - 1, -1, -1):
                num = np.float32(np.float32(np.float32(m) + np.float32(0.5)) - center)
                events.append((np.float32(num / oi), i))
    events.sort(key=lambda item: (float(item[0]), item[1]))

    current = code.copy()
    best_step = -1
    best_score = -1.0
    have = False
    if dot > 0.0 and normsq > 0.0:
        best_score = (dot * dot) / normsq
        have = True
    for step, (_t, index) in enumerate(events):
        delta = 1 if rot[index] > 0 else -1
        value = float(current[index]) - float(center)
        dot += float(delta) * float(rot[index])
        normsq += 2.0 * value * float(delta) + float(delta) * float(delta)
        current[index] += delta
        if dot > 0.0 and normsq > 0.0:
            score = (dot * dot) / normsq
            if (not have) or score > best_score:
                best_score = score
                best_step = step
                have = True
    chosen = code.copy()
    if best_step >= 0:
        for step in range(best_step + 1):
            index = events[step][1]
            chosen[index] += 1 if rot[index] > 0 else -1
    acc = np.float32(0.0)
    for i in range(dim):
        acc = np.float32(acc + np.float32(np.float32(chosen[i]) - center) * rot[i])
    return chosen.astype(np.uint8), acc


def oracle_encode_extended(vec, seed, centroid, bits):
    dim_pad = padded_dim(vec.shape[0])
    unit, norm = residual_unit(vec, centroid)
    rot = wht_f32(unit * splitmix_signs(dim_pad, seed))
    codes, dot = _extended_codes(rot, bits)
    if bits == 4:
        packed = np.zeros(dim_pad // 2, np.uint8)
        for i in range(dim_pad):
            shift = 4 if (i & 1) else 0
            packed[i >> 1] = np.uint8(packed[i >> 1] | np.uint8(int(codes[i]) << shift))
        payload = packed.tobytes()
    else:
        payload = codes.tobytes()
    meta = np.array([norm, dot], np.float32).tobytes()
    return np.frombuffer(payload + meta, dtype=np.uint8).copy()


def _unpack_codes(slot, bits, dim_pad):
    if bits == 8:
        return slot[:dim_pad].astype(np.float32)
    codes = np.empty(dim_pad, np.float32)
    for i in range(dim_pad):
        byte = int(slot[i >> 1])
        codes[i] = (byte >> 4) if (i & 1) else (byte & 0x0F)
    return codes


def oracle_distance_extended(query, slot, seed, centroid, bits):
    dim_pad = padded_dim(query.shape[0])
    payload = dim_pad if bits == 8 else dim_pad // 2
    codes = _unpack_codes(slot, bits, dim_pad)
    meta = np.frombuffer(slot[payload:].tobytes(), dtype=np.float32)
    xnorm = f32(meta[0])
    dot_factor = f32(meta[1])
    center = f32(127.5 if bits == 8 else 7.5)
    q_unit, qnorm = residual_unit(query, centroid)
    rot = wht_f32(q_unit * splitmix_signs(dim_pad, seed)) if qnorm > 0 else q_unit
    acc = f32(0.0)
    sum_q = f32(0.0)
    if qnorm > 0:
        for i in range(dim_pad):
            acc = f32(acc + f32(codes[i]) * rot[i])
            sum_q = f32(sum_q + rot[i])
    ip = f32(f32(acc - f32(center * sum_q)) / dot_factor)
    return float(f32(xnorm * xnorm + qnorm * qnorm - f32(2.0) * xnorm * qnorm * ip))


def test_extended_slot_size():
    space4 = RaBitQSpace(128, rot_seed=1, bits=4)
    space8 = RaBitQSpace(128, rot_seed=1, bits=8)
    assert space4.bits() == 4
    assert space8.bits() == 8
    assert space4.code_size_bytes() == 128 // 2 + 8
    assert space8.code_size_bytes() == 128 + 8
    assert RaBitQSpace(6, bits=4).code_size_bytes() == 4 + 8
    assert RaBitQSpace(6, bits=8).code_size_bytes() == 8 + 8


def test_invalid_bits_rejected():
    with pytest.raises(Exception) as exc:
        RaBitQSpace(8, bits=3)
    assert "bits" in str(exc.value).lower()


def test_explicit_one_bit_matches_default():
    rng = np.random.default_rng(5)
    x = rng.standard_normal(16).astype(np.float32)
    default = RaBitQSpace(16, rot_seed=42)
    explicit = RaBitQSpace(16, rot_seed=42, bits=1)
    np.testing.assert_array_equal(default.encode(x), explicit.encode(x))


@pytest.mark.parametrize("dim,bits", [(8, 4), (8, 8), (16, 4), (16, 8)])
def test_extended_encode_matches_oracle(dim, bits):
    seed = 42
    rng = np.random.default_rng(2000 + dim * 10 + bits)
    centroid = rng.standard_normal(dim).astype(np.float32)
    space = RaBitQSpace(dim, rot_seed=seed, centroid=centroid, bits=bits)
    payload = padded_dim(dim) if bits == 8 else padded_dim(dim) // 2
    for _ in range(4):
        x = rng.standard_normal(dim).astype(np.float32)
        got = space.encode(x)
        expect = oracle_encode_extended(x, seed, centroid, bits)
        # Grid indices are exact. norm and <y, o'> are float32 sums; the
        # extension is built with -ffast-math, so a reduction can differ
        # from the left-to-right oracle by about one ulp.
        np.testing.assert_array_equal(got[:payload], expect[:payload])
        got_meta = np.frombuffer(got[payload:].tobytes(), dtype=np.float32)
        exp_meta = np.frombuffer(expect[payload:].tobytes(), dtype=np.float32)
        np.testing.assert_allclose(got_meta, exp_meta, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dim,bits", [(8, 4), (8, 8), (16, 4), (128, 4), (128, 8)])
def test_extended_distance_matches_oracle(dim, bits):
    seed = 42
    rng = np.random.default_rng(3000 + dim * 10 + bits)
    centroid = rng.standard_normal(dim).astype(np.float32) * np.float32(0.1)
    space = RaBitQSpace(dim, rot_seed=seed, centroid=centroid, bits=bits)
    # D < 16 stays on the scalar tail for 4-bit NEON. Wider D exercises vmla.
    repeats = 8 if dim >= 128 else 4
    for _ in range(repeats):
        x = rng.standard_normal(dim).astype(np.float32)
        q = rng.standard_normal(dim).astype(np.float32)
        code = space.encode(x)
        got = space.distance(q, code)
        expect = oracle_distance_extended(q, code, seed, centroid, bits)
        np.testing.assert_allclose(got, expect, rtol=1e-4, atol=1e-3)
