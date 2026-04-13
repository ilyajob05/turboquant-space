import numpy as np
from turboquant import TurboQuantSpace as TQS

for bits in [4, 8]:
    dim = 128
    rng = np.random.default_rng(77)
    vecs = rng.integers(0, 256, size=(10, dim)).astype(np.float32)
    space = TQS(dim, bits_per_coord=bits)
    codes = space.encode_batch(vecs)

    for i, j in [(0, 1), (2, 3), (4, 5)]:
        d_true = float(np.sum((vecs[i] - vecs[j]) ** 2))
        d_orig = space.distance_symmetric(codes[i], codes[j])
        d_light = space.distance_m_to_n_symmetric_light(
            codes[i:i+1], codes[j:j+1])[0, 0]
        d_full = space.distance_m_to_n_symmetric_full(
            codes[i:i+1], codes[j:j+1])[0, 0]
        print(f"bits={bits} pair=({i},{j}): true={d_true:.0f} "
              f"orig={d_orig:.0f} light={d_light:.0f} full={d_full:.0f}")
